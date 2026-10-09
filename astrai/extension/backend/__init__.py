"""Backend selection, fallbacks, and execution policies."""

from astrai.extension.backend.attention import (
    ATTN_BACKEND,
    AttentionBackend,
    AttentionBackendFactory,
    CudaBackend,
    FlashAttnBackend,
    TorchNativeBackend,
    attention,
    attn_backend,
    get_backend,
)
from astrai.extension.backend.newton_schulz import newton_schulz, symm_out, syrk_out
from astrai.extension.backend.rotary import apply_rotary_emb

__all__ = [
    "ATTN_BACKEND",
    "AttentionBackend",
    "AttentionBackendFactory",
    "CudaBackend",
    "FlashAttnBackend",
    "TorchNativeBackend",
    "apply_rotary_emb",
    "newton_schulz",
    "symm_out",
    "syrk_out",
    "attention",
    "attn_backend",
    "get_backend",
]
