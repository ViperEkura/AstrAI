"""Registry and capability-based selection for operator implementations.

Families provide call axes and a fallback. Implementations may be registered
later by external packages. Explicit/context selection is strict; process
selection is soft. Unfaithful implementations require explicit selection.
"""

import contextvars
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

Axes = Mapping[str, Any]
Call = Tuple[Tuple, Dict[str, Any]]
_BOTH_MODES = frozenset(("train", "infer"))


def _fmt(value: Any) -> str:
    return str(value)


class Spec:
    """Composable, self-describing predicate over a family's axes dict."""

    __slots__ = ("_fn", "_desc")

    def __init__(self, fn: Callable[[Axes], bool], desc: str):
        self._fn = fn
        self._desc = desc

    def matches(self, ax: Axes) -> bool:
        return bool(self._fn(ax))

    @property
    def description(self) -> str:
        return self._desc

    def __and__(self, other: "Spec") -> "Spec":
        return Spec(
            lambda ax: self._fn(ax) and other._fn(ax),
            f"({self._desc} and {other._desc})",
        )

    def __or__(self, other: "Spec") -> "Spec":
        return Spec(
            lambda ax: self._fn(ax) or other._fn(ax),
            f"({self._desc} or {other._desc})",
        )

    def __invert__(self) -> "Spec":
        return Spec(lambda ax: not self._fn(ax), f"not({self._desc})")

    @classmethod
    def always(cls) -> "Spec":
        return cls(lambda ax: True, "always")

    @classmethod
    def of(cls, fn: Callable[[Axes], bool], desc: str) -> "Spec":
        return cls(fn, desc)


class Axis:
    """Named-axis predicate builder: ``axis("dtype").in_(torch.bfloat16)``.

    Axis names belong to each family; the core never defines or inspects
    them beyond the predicate the builder closes over.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str):
        self._name = name

    def in_(self, *values: Any) -> Spec:
        rendered = ", ".join(_fmt(v) for v in values)
        return Spec(
            lambda ax: ax.get(self._name) in values,
            f"{self._name} in {{{rendered}}}",
        )

    def eq(self, value: Any) -> Spec:
        return Spec(
            lambda ax: ax.get(self._name) == value, f"{self._name}=={_fmt(value)}"
        )

    def is_(self, value: Any) -> Spec:
        return Spec(
            lambda ax: ax.get(self._name) is value, f"{self._name} is {_fmt(value)}"
        )

    def none(self) -> Spec:
        return Spec(lambda ax: ax.get(self._name) is None, f"{self._name} is None")

    def not_none(self) -> Spec:
        return Spec(
            lambda ax: ax.get(self._name) is not None, f"{self._name} is not None"
        )

    def truthy(self) -> Spec:
        return Spec(lambda ax: bool(ax.get(self._name)), self._name)

    def falsy(self) -> Spec:
        return Spec(lambda ax: not ax.get(self._name), f"!{self._name}")


def axis(name: str) -> Axis:
    """Entry point for named-axis predicates; see ``Axis``."""
    return Axis(name)


def tensor_axes(x: torch.Tensor, **extra: Any) -> Dict[str, Any]:
    """Tensor-derived axes shared by most families; opt-in, extendable."""
    return {
        "dtype": x.dtype,
        "device_cuda": x.is_cuda,
        "grad_enabled": torch.is_grad_enabled(),
        **extra,
    }


@dataclass(frozen=True)
class ImplRecord:
    """One decision-table row: an implementation plus its capability."""

    family: str
    name: str
    obj: Any
    spec: Spec
    available: Callable[[], bool] = lambda: True
    priority: int = 100
    faithful: bool = True
    modes: FrozenSet[str] = _BOTH_MODES


@dataclass
class OpFamily:
    name: str
    axes: Callable[..., Axes]
    provider: Callable[[], List[ImplRecord]]
    fallback: Callable[[], ImplRecord]


_FAMILIES: Dict[str, OpFamily] = {}
_EXTERNAL: Dict[str, Dict[str, ImplRecord]] = {}
_ENV_ALIASES: Dict[str, str] = {}

# Priority-sorted record lists, cached per family until invalidated: the
# providers return constant record sets (availability is a per-call check
# on each record, not part of the sort), so resolving an op must not
# rebuild and re-sort the list every call. register_family and loader-side
# kernel imports invalidate.
_family_records: Dict[str, List[ImplRecord]] = {}
_cache_revision = 0

_current_overrides: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "astrai_op_overrides", default={}
)

_selection_lock = threading.Lock()
_selection: Optional[Dict[str, str]] = None  # None = not seeded yet
_warned: set = set()


def register_family(
    name: str,
    axes: Callable[..., Axes],
    provider: Callable[[], List[ImplRecord]],
    fallback: Callable[[], ImplRecord],
) -> None:
    """Register (or replace) a family. Availability changes are honored
    without re-registration: each record's ``available`` runs per resolve,
    and loader-side kernel imports call :func:`invalidate`.
    ``axes`` mirrors the op call signature and snapshots that family's
    decision axes; unregistered handles are probed through the same args.
    """
    _FAMILIES[name] = OpFamily(name, axes, provider, fallback)
    invalidate(name)


def register_impl(record: ImplRecord) -> None:
    """Add a third-party implementation to an existing family."""
    fam = _family(record.family)
    if not record.modes or not record.modes <= _BOTH_MODES:
        raise ValueError("modes must contain train, infer, or both")
    if any(row.name == record.name for row in _records(fam)):
        raise ValueError(
            f"{record.family} implementation {record.name!r} already exists"
        )
    _EXTERNAL.setdefault(record.family, {})[record.name] = record
    invalidate(record.family)


def unregister_impl(family: str, name: str) -> None:
    """Remove a third-party implementation."""
    del _EXTERNAL[family][name]
    invalidate(family)


def invalidate(family: Optional[str] = None) -> None:
    """Drop the cached record lists — call when a provider's record set or
    a kernel module's availability changed (family=None drops all)."""
    global _cache_revision
    _cache_revision += 1
    if family is None:
        _family_records.clear()
    else:
        _family_records.pop(family, None)


def _records(fam: OpFamily) -> List[ImplRecord]:
    records = _family_records.get(fam.name)
    if records is None:
        records = sorted(
            [*fam.provider(), *_EXTERNAL.get(fam.name, {}).values()],
            key=lambda r: r.priority,
        )
        _family_records[fam.name] = records
    return records


def register_env_alias(family: str, varname: str) -> None:
    """Legacy single-value env var for a family (e.g. attention →
    ASTR_BACKEND); an ASTR_OPS entry wins when both are set."""
    _ENV_ALIASES[family] = varname


def _family(name: str) -> OpFamily:
    fam = _FAMILIES.get(name)
    if fam is None:
        raise KeyError(f"no operator family registered under {name!r}")
    return fam


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        logger.warning(message)


def set_override(family: str, handle: Any) -> contextvars.Token:
    overrides = dict(_current_overrides.get())
    overrides[family] = handle
    return _current_overrides.set(overrides)


def reset_override(token: contextvars.Token) -> None:
    _current_overrides.reset(token)


def get_override(family: str) -> Optional[Any]:
    return _current_overrides.get().get(family)


@contextmanager
def op_backend(**handles: Any):
    """Select implementations per family for the enclosed scope::

        with op_backend(attention="torch_native", rotary="torch"):
            engine.generate(...)

    String handles are validated eagerly against the family's currently
    available implementations; object handles pass through unchecked.
    """
    for family, handle in handles.items():
        if isinstance(handle, str):
            fam = _FAMILIES.get(family)
            if fam is None:
                raise ValueError(f"unknown operator family {family!r}")
            record = _record_for_handle(fam, handle, _records(fam))
            if record is None or not record.available():
                raise ValueError(f"Unknown {family} implementation: {handle!r}")
    tokens = [set_override(f, h) for f, h in handles.items()]
    try:
        yield
    finally:
        for token in reversed(tokens):
            reset_override(token)


def parse_selections(raw: str) -> Dict[str, str]:
    """Parse an ASTR_OPS-style ``family=impl`` comma list (the seed format).

    Malformed entries warn once and are dropped.
    """
    parsed: Dict[str, str] = {}
    for item in raw.split(","):
        key_part, sep, value = item.strip().partition("=")
        key_part, value = key_part.strip(), value.strip()
        if not sep or not key_part or not value:
            _warn_once(f"ASTR_OPS: ignoring malformed entry {item!r}")
            continue
        parsed[key_part] = value
    return parsed


def _seed_selection() -> Dict[str, str]:
    """The process selection tier, seeded once from the deprecated
    ASTR_OPS / legacy-alias variables and owned at runtime by set_op()."""
    global _selection
    with _selection_lock:
        if _selection is None:
            merged: Dict[str, str] = {}
            raw = os.environ.get("ASTR_OPS", "").strip()
            if raw:
                merged.update(parse_selections(raw))
            for fam, varname in _ENV_ALIASES.items():
                raw = os.environ.get(varname, "").strip()
                if raw:
                    merged.setdefault(fam, raw.lower())
            for fam in [f for f in merged if f not in _FAMILIES and f != "profile"]:
                _warn_once(f"ASTR_OPS: unknown operator family {fam!r}; dropping it")
                merged.pop(fam)
            _selection = merged
        return _selection


def set_op(family: str, impl: Optional[str] = None) -> None:
    """Set (or, with ``impl=None``, clear) the process-level implementation
    for a family — the runtime replacement for the ``ASTR_OPS`` /
    ``ASTR_BACKEND`` variables. Like those variables the selection is soft:
    an incapable pick falls through to the chain."""
    overrides = dict(_seed_selection())
    if impl is None:
        overrides.pop(family, None)
    else:
        overrides[family] = impl
    global _selection
    with _selection_lock:
        _selection = overrides


def env_overrides() -> Dict[str, str]:
    """The active process-level selections (family or ``profile``)."""
    return _seed_selection()


def env_selection(family: str) -> Optional[str]:
    return env_overrides().get(family)


@dataclass(frozen=True)
class Resolution:
    record: ImplRecord
    origin: str


class ExplicitSelectionError(RuntimeError):
    """An explicitly selected implementation cannot handle the call."""


def _record_for_handle(
    fam: OpFamily, handle: Any, records: List[ImplRecord]
) -> Optional[ImplRecord]:
    if isinstance(handle, str):
        return next((r for r in records if r.name == handle), None)
    return next((r for r in records if r.obj is handle), None)


def _adhoc_record(family: str, handle: Any, args: Tuple, kwargs: Dict) -> ImplRecord:
    """Wrap an unregistered object; capability probes its own method on
    the original call arguments."""
    supports = getattr(handle, "supports_call", None)
    if supports is not None:
        spec = Spec.of(
            lambda ax: bool(supports(*args, **kwargs)),
            f"{type(handle).__name__}.supports_call",
        )
    else:
        spec = Spec.always()
    return ImplRecord(
        family,
        type(handle).__name__,
        handle,
        spec,
        available=getattr(handle, "available", lambda: True),
        modes=frozenset(getattr(handle, "modes", _BOTH_MODES)),
    )


def _mode_allowed(record: ImplRecord, mode: Optional[str]) -> bool:
    return mode in record.modes if mode is not None else record.modes == _BOTH_MODES


def _capable(record: ImplRecord, ax: Axes) -> bool:
    return (
        _mode_allowed(record, ax.get("mode"))
        and record.available()
        and record.spec.matches(ax)
    )


def _fallback(fam: OpFamily, ax: Axes, origin: str) -> Resolution:
    record = fam.fallback()
    if not _capable(record, ax):
        raise RuntimeError(f"No {fam.name} implementation can handle this call")
    return Resolution(record, origin)


def resolve(
    family: str, *args: Any, explicit: Optional[Any] = None, **kwargs: Any
) -> Resolution:
    """Resolve one family for one call (explicit-strict / implicit-loose).

    ``args``/``kwargs`` mirror the op call: the family's ``axes`` extractor
    snapshots the decision axes from them, and unregistered handles are
    probed through their own ``supports_call`` with the same arguments.
    """
    fam = _family(family)
    ax = fam.axes(*args, **kwargs)
    records = _records(fam)

    handle: Optional[Any] = None
    origin = "chain"
    if explicit is not None:
        handle, origin = explicit, "explicit"
    elif get_override(family) is not None:
        handle, origin = get_override(family), "context"
    else:
        env_name = env_selection(family)
        if env_name is not None:
            handle, origin = env_name, "env"

    if handle is not None:
        record = _record_for_handle(fam, handle, records)
        if record is None and not isinstance(handle, str):
            record = _adhoc_record(family, handle, args, kwargs)
        if record is None:
            if origin in ("explicit", "context"):
                raise ValueError(f"Unknown {family} implementation: {handle!r}")
            _warn_once(f"ASTR_OPS: {family}={handle!r} is not registered; ignoring")
        else:
            if _capable(record, ax):
                return Resolution(record, origin)
            if origin in ("explicit", "context"):
                raise ExplicitSelectionError(
                    f"Explicitly-set backend {type(record.obj).__name__} cannot "
                    f"handle this {family} call; required: {record.spec.description}"
                )

    if handle is None and env_overrides().get("profile") == "reference":
        return _fallback(fam, ax, "profile")

    for record in records:
        if record.faithful and _capable(record, ax):
            return Resolution(record, "chain")
    return _fallback(fam, ax, "fallback")


def cache_token(family: str) -> Optional[int]:
    """Revision for caching a builtin decision, or None for dynamic selections.

    External implementations can have runtime predicates. Context and process
    overrides must be resolved on each call. Builtin callers include their own
    input metadata and mutable settings in the remaining cache key.
    """
    if (
        _EXTERNAL.get(family)
        or get_override(family) is not None
        or env_selection(family) is not None
        or env_overrides().get("profile") is not None
    ):
        return None
    return _cache_revision


def resolve_plan(calls: Mapping[str, Call]) -> Dict[str, Resolution]:
    """Resolve several families at once (one decision snapshot)."""
    return {
        family: resolve(family, *args, **kwargs)
        for family, (args, kwargs) in calls.items()
    }


def _describe_axes(ax: Axes) -> str:
    return " ".join(f"{key}={ax[key]}" for key in sorted(ax) if not key.startswith("_"))


def explain(
    family: str, *args: Any, explicit: Optional[Any] = None, **kwargs: Any
) -> str:
    """Human-readable decision trace for one family call."""
    fam = _family(family)
    ax = fam.axes(*args, **kwargs)
    records = _records(fam)
    lines = [f"[{family}] {_describe_axes(ax)}"]
    for record in records:
        if not _mode_allowed(record, ax.get("mode")):
            lines.append(f"  {record.name}: SKIP mode {ax.get('mode')}")
        elif not record.available():
            lines.append(f"  {record.name}: SKIP unavailable")
        elif not record.faithful:
            lines.append(f"  {record.name}: SKIP not faithful (explicit-only)")
        elif record.spec.matches(ax):
            lines.append(f"  {record.name}: MATCH ({record.spec.description})")
        else:
            lines.append(f"  {record.name}: reject ({record.spec.description})")
    try:
        resolution = resolve(family, *args, explicit=explicit, **kwargs)
        lines.append(f"  => {resolution.record.name} (origin={resolution.origin})")
    except (ExplicitSelectionError, ValueError, RuntimeError) as exc:
        lines.append(f"  => ERROR: {exc}")
    return "\n".join(lines)


def explain_plan(calls: Mapping[str, Call]) -> str:
    return "\n".join(
        explain(family, *args, **kwargs) for family, (args, kwargs) in calls.items()
    )


__all__ = [
    "Axes",
    "Call",
    "ExplicitSelectionError",
    "ImplRecord",
    "OpFamily",
    "Resolution",
    "Spec",
    "Axis",
    "axis",
    "env_overrides",
    "env_selection",
    "invalidate",
    "parse_selections",
    "set_op",
    "explain",
    "explain_plan",
    "get_override",
    "op_backend",
    "register_env_alias",
    "register_family",
    "register_impl",
    "reset_override",
    "resolve",
    "resolve_plan",
    "set_override",
    "tensor_axes",
    "unregister_impl",
]
