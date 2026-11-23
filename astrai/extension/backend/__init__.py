"""Backend selection, fallbacks, and execution policies."""

from astrai.extension.backend.attention import (
    AttentionDispatcher,
    AttentionSpec,
    attention,
)
from astrai.extension.backend.newton_schulz import newton_schulz, symm_out, syrk_out
from astrai.extension.backend.rotary import apply_rotary_emb
from astrai.extension.backend.swiglu import swiglu

__all__ = [
    "apply_rotary_emb",
    "newton_schulz",
    "symm_out",
    "syrk_out",
    "swiglu",
    "AttentionDispatcher",
    "AttentionSpec",
    "attention",
]
