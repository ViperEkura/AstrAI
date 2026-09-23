"""Checkpoint extras from optional components.

State that belongs to optional accelerators (the fp8 delayed-scaling rings
today) flows through here, so neither the checkpoint callback nor the context
builder imports those components or knows they exist: the checkpoint path
asks a generic registry, and the trainer's single guarded import lives in
this module (the extension is a compiled artifact a CPU-only run may not
have — bf16 training itself never needs it).

Adding a component = one ``(key, provider, restorer)`` triple below; the
checkpoint callback and ``TrainContextBuilder`` stay untouched.
"""

from typing import Any


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


# (checkpoint key, snapshot provider, restorer) — provider returns None when
# the component has nothing to persist under the current configuration.
_EXTRAS: tuple[tuple[str, Any, Any], ...] = (("fp8_state", _fp8_extra, _fp8_restore),)


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
