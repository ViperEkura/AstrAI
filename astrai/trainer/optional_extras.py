"""Checkpoint extras from optional components.

Two kinds of state ride along in a checkpoint, and they need different
mechanics:

* **Run-owned state** — the optimizer moments, the scheduler position, and the
  fp8 delayed-scaling rings.  It is declared once in
  :data:`EXTRAS` as ``(key, snapshot, restore)``; the checkpoint callback
  snapshots through it and the builder restores through it, so neither has to
  know which components exist.
* **Component-owned tensor state** — the PPO critic, its optimizer, the frozen
  DPO/GRPO reference.  It is declared once in :data:`COMPONENT_EXTRAS` and
  snapshotted generically by the checkpoint callback, but each component
  restores its own entry: the build order differs (the critic exists before
  its optimizer, the frozen reference only after the strategy is built).

Adding a component = one :class:`Extra` row (run-owned) or one
:class:`ComponentExtra` row (component-owned); the checkpoint callback and
``TrainContextBuilder`` stay untouched.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple

import torch

from astrai.extension.policy.quantization import autocast
from astrai.trainer.components import load_optimizer_state_dict

logger = logging.getLogger(__name__)


def _fp8_snapshot(_context: Any) -> Optional[Dict]:
    sd = autocast.fp8_state_dict()
    return sd if sd["entries"] else None


def _fp8_restore(_context: Any, sd: Dict) -> None:
    autocast.fp8_load_state_dict(sd)


def _optimizer_snapshot(context: Any) -> Optional[Dict]:
    return context.optimizer.state_dict() if context.optimizer is not None else None


def _optimizer_restore(context: Any, state: Dict) -> None:
    # A full resume keeps the saved parameter-group options (betas, weight
    # decay); a partial one keeps the configured ones and only reuses the
    # moments, which is what load_optimizer_state_dict exists for.
    full_resume = (
        context.checkpoint is not None and "scheduler" in context.checkpoint.extra
    )
    if full_resume:
        context.optimizer.load_state_dict(state)
    else:
        load_optimizer_state_dict(context.optimizer, state)


def _scheduler_snapshot(context: Any) -> Optional[Dict]:
    return context.scheduler.state_dict() if context.scheduler is not None else None


def _scheduler_restore(context: Any, state: Dict) -> None:
    if context.scheduler is None:
        return
    context.scheduler.load_state_dict(state)
    # The saved schedule position wins over the configured warmup start.
    for group, lr in zip(
        context.optimizer.param_groups,
        context.scheduler.get_last_lr(),
        strict=True,
    ):
        group["lr"] = lr


@dataclass(frozen=True)
class Extra:
    """One run-owned state that must survive a resume.

    Attributes:
        key: Checkpoint extra key (also the ``<key>.pt`` file stem).
        snapshot: Reads the state off the context; ``None`` means "nothing to
            persist under this configuration".
        restore: Writes it back.  Order in :data:`EXTRAS` is the restore order,
            because the scheduler fix-up reads the optimizer's groups.
        resumable: Whether a resume must carry this entry.  A missing resumable
            entry is refused; a missing optional one is simply not restored.
        purpose: One-line reason the state must survive a resume.
    """

    key: str
    snapshot: Callable[[Any], Any]
    restore: Callable[[Any, Any], None]
    resumable: bool = False
    purpose: str = ""


EXTRAS: Tuple[Extra, ...] = (
    Extra(
        key="optimizer",
        snapshot=_optimizer_snapshot,
        restore=_optimizer_restore,
        resumable=True,
        purpose="the optimizer's moments",
    ),
    Extra(
        key="scheduler",
        snapshot=_scheduler_snapshot,
        restore=_scheduler_restore,
        resumable=True,
        purpose="the LR schedule position",
    ),
    Extra(key="fp8_state", snapshot=_fp8_snapshot, restore=_fp8_restore),
)

EXTRA_KEYS: Tuple[str, ...] = tuple(entry.key for entry in EXTRAS)


def snapshot_extras(context: Any) -> Dict[str, Any]:
    """Snapshot every run-owned component that has state worth persisting."""
    extra: Dict[str, Any] = {}
    for entry in EXTRAS:
        state = entry.snapshot(context)
        if state is not None:
            extra[entry.key] = state
    return extra


def restore_extras(context: Any, extra: Dict[str, Any]) -> None:
    """Restore whatever of the declared run-owned extras is present."""
    for entry in EXTRAS:
        if entry.key in extra:
            entry.restore(context, extra[entry.key])


def missing_resumable_extras(extra: Dict[str, Any], options: Dict[str, bool]) -> list:
    """Declared entries a resume cannot do without, that this one lacks."""
    return [
        entry.key
        for entry in EXTRAS
        if entry.resumable and options.get(entry.key, True) and entry.key not in extra
    ]


ExtraKind = Literal["module", "optimizer"]


@dataclass(frozen=True)
class ComponentExtra:
    """One strategy-owned tensor state that must survive a resume.

    Attributes:
        key: Checkpoint extra key (also the ``<key>.pt`` file stem).
        attribute: Attribute on the strategy holding the source object.
        kind: Snapshot semantics.  ``"module"`` stores a detached CPU copy —
            checkpoints load on CPU, so writing device tensors would bake CUDA
            device indices into the file.  ``"optimizer"`` stores
            ``state_dict()`` verbatim, which is what
            ``Optimizer.load_state_dict`` expects.
        save_flag: Optional config flag gating the save.
        allow_missing_flag: Optional config flag that downgrades a missing
            entry on resume from an error to a warning.
        purpose: One-line reason the state must survive a resume, quoted by
            the shared error/warning message.
    """

    key: str
    attribute: str
    kind: ExtraKind = "module"
    save_flag: Optional[str] = None
    allow_missing_flag: Optional[str] = None
    purpose: str = ""


COMPONENT_EXTRAS: Tuple[ComponentExtra, ...] = (
    ComponentExtra(
        key="value_model",
        attribute="critic",
        purpose="the PPO critic regresses against rollout-pinned returns",
    ),
    ComponentExtra(
        key="value_optimizer",
        attribute="critic_optimizer",
        kind="optimizer",
        purpose="the critic optimizer's moments",
    ),
    ComponentExtra(
        key="reference_model",
        attribute="ref_model",
        save_flag="save_reference_model",
        allow_missing_flag="allow_reference_reanchor",
        purpose="the policy is regularised towards it (KL / DPO anchor)",
    ),
)


def _index_by_key(
    entries: Iterable[ComponentExtra],
) -> Dict[str, ComponentExtra]:
    by_key: Dict[str, ComponentExtra] = {}
    for entry in entries:
        if entry.key in by_key:
            raise ValueError(f"duplicate component extra key: {entry.key!r}")
        by_key[entry.key] = entry
    return by_key


_BY_KEY = _index_by_key(COMPONENT_EXTRAS)


def component_extra(key: str) -> ComponentExtra:
    """Look up one declared entry; an unknown key is a programming error."""
    entry = _BY_KEY.get(key)
    if entry is None:
        raise ValueError(
            f"unknown component extra {key!r}; declared: {sorted(_BY_KEY)}"
        )
    return entry


def component_extra_keys(*attributes: str) -> Tuple[str, ...]:
    """Declared extra keys owned by the given strategy attributes."""
    wanted = set(attributes)
    return tuple(e.key for e in COMPONENT_EXTRAS if e.attribute in wanted)


def component_optimizers(strategy):
    """Enumerate all strategy optimizers before any rank starts snapshotting."""
    return tuple(
        component
        for entry in COMPONENT_EXTRAS
        if entry.kind == "optimizer"
        and (component := getattr(strategy, entry.attribute, None)) is not None
    )


def snapshot_component_extras(strategy, config: Any = None) -> Dict[str, Any]:
    """Collect every declared component extra the strategy currently owns."""
    extra: Dict[str, Any] = {}
    for entry in COMPONENT_EXTRAS:
        if entry.save_flag is not None and not getattr(config, entry.save_flag, True):
            continue
        component = getattr(strategy, entry.attribute, None)
        if component is None:
            continue
        state_dict = component.state_dict()
        if entry.kind == "module":
            state_dict = {
                key: value.detach().cpu() for key, value in state_dict.items()
            }
        extra[entry.key] = state_dict
    return extra


def require_component_extras(
    extra: Dict[str, Any],
    keys: Iterable[str],
    *,
    strategy: str,
    requirement: str,
) -> None:
    """Fail fast when a resume needs declared extras the checkpoint lacks."""
    missing = [key for key in keys if key not in extra]
    if missing:
        raise ValueError(
            f"{strategy} resume requires {requirement} in the checkpoint; "
            f"missing extras: {', '.join(missing)}"
        )


def load_component_extra(
    extra: Dict[str, Any],
    key: str,
    target: Any,
    config: Any = None,
) -> bool:
    """Restore one declared extra into its component, enforcing the policy.

    Returns ``True`` when the state was loaded, ``False`` when a declared
    escape flag allowed a missing entry (a warning is logged).  Raises
    ``ValueError`` when the entry is missing and nothing allows that.
    """
    entry = component_extra(key)
    state_dict = extra.get(key)
    if state_dict is not None:
        target.load_state_dict(state_dict)
        return True

    allow_flag = entry.allow_missing_flag
    if allow_flag is not None and getattr(config, allow_flag, False):
        logger.warning(
            "resume checkpoint has no %r extra: keeping the reconstructed %s "
            "(%s=True) — %s",
            key,
            entry.attribute,
            allow_flag,
            entry.purpose,
        )
        return False

    hints = []
    if entry.save_flag is not None:
        hints.append(f"save checkpoints with {entry.save_flag}=True")
    if allow_flag is not None:
        hints.append(f"set {allow_flag}=True to accept the reconstructed state")
    hint_text = f" ({'; '.join(hints)})" if hints else ""
    raise ValueError(
        f"resume checkpoint has no {key!r} extra and {entry.purpose} must "
        f"survive a resume{hint_text}"
    )
