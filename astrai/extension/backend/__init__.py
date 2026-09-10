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
from astrai.extension.backend.rotary import apply_rotary_emb

__all__ = [
    "ATTN_BACKEND",
    "AttentionBackend",
    "AttentionBackendFactory",
    "CudaBackend",
    "FlashAttnBackend",
    "TorchNativeBackend",
    "apply_rotary_emb",
    "attention",
    "attn_backend",
    "get_backend",
]
