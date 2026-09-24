"""Checkpoint extras from optional components.

Two kinds of state ride along in a checkpoint, and they need different
mechanics:

* **Global accelerator state** — the fp8 delayed-scaling rings and the RNG
  states.  It is context-free and restored centrally, so it goes through the
  ``_EXTRAS`` registry: one ``(key, provider, restorer)`` triple per
  component, and neither the checkpoint callback nor the context builder
  imports the component or knows it exists.
* **Component-owned tensor state** — the PPO critic, its optimizer, the frozen
  DPO/GRPO reference.  It is declared once in :data:`COMPONENT_EXTRAS` and
  snapshotted generically by the checkpoint callback, but each component
  restores its own entry: the build order differs (the critic exists before
  its optimizer, the frozen reference only after the strategy is built).  The
  table is still the single source for the key names, the owning strategy
  attribute, and what a missing entry means on resume — so a new component
  needs one table row, not edits in three call sites.

Adding a component = one registry triple (global state) or one table row
(component-owned state); the checkpoint callback and ``TrainContextBuilder``
stay untouched.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import torch

logger = logging.getLogger(__name__)


def _quantize_mod():
    try:
        from astrai.extension import quantize
    except ImportError:  # extension not built: fp8 was never in use either
        return None
    return quantize


def _fp8_extra() -> dict | None:
    mod = _quantize_mod()
    if mod is None:
        return None
    sd = mod.fp8_state_dict()
    return sd if sd["entries"] else None


def _fp8_restore(sd: dict) -> None:
    mod = _quantize_mod()
    if mod is not None:
        mod.fp8_load_state_dict(sd)


def _rng_extra() -> dict:
    """Snapshot every RNG the training loop draws from.

    Covers python's ``random``, torch CPU and (when initialized) CUDA
    generators, and numpy if installed.  Without this, a resumed run's
    dropout masks, init draws, and rollout sampling diverge from the
    uninterrupted run from the very first step.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
    except ImportError:
        return state
    state["numpy"] = np.random.get_state()
    return state


def _rng_restore(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        # Tolerate a device-count change across the resume: restore what
        # fits rather than failing the whole load.
        torch.cuda.set_rng_state_all(state["torch_cuda"][: torch.cuda.device_count()])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


# (checkpoint key, snapshot provider, restorer) — provider returns None when
# the component has nothing to persist under the current configuration.
_EXTRAS: tuple[tuple[str, Any, Any], ...] = (
    ("fp8_state", _fp8_extra, _fp8_restore),
    ("rng_state", _rng_extra, _rng_restore),
)


def checkpoint_extras() -> dict[str, Any]:
    """Snapshot every optional component that has state worth persisting."""
    extra: dict[str, Any] = {}
    for name, provider, _ in _EXTRAS:
        sd = provider()
        if sd is not None:
            extra[name] = sd
    return extra


def restore_checkpoint_extras(extra: dict[str, Any]) -> None:
    """Restore whatever of the known optional extras is present in a loaded
    checkpoint; unknown keys are ignored."""
    for name, _, restorer in _EXTRAS:
        if name in extra:
            restorer(extra[name])


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


COMPONENT_EXTRAS: tuple[ComponentExtra, ...] = (
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
) -> dict[str, ComponentExtra]:
    by_key: dict[str, ComponentExtra] = {}
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


def component_extra_keys(*attributes: str) -> tuple[str, ...]:
    """Declared extra keys owned by the given strategy attributes."""
    wanted = set(attributes)
    return tuple(e.key for e in COMPONENT_EXTRAS if e.attribute in wanted)


def snapshot_component_extras(strategy, config: Any = None) -> dict[str, Any]:
    """Collect every declared component extra the strategy currently owns."""
    extra: dict[str, Any] = {}
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
    extra: dict[str, Any],
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
    extra: dict[str, Any],
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
