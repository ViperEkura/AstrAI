"""CUDA attention kernel wrappers with torch fallback.

Public API:
    - ``attn_decode`` — single-query decode attention
    - ``attn_prefill`` — multi-query prefill attention
    - ``attn_paged_decode`` — paged decode attention (direct page-table access)

Interface (shared by all wrappers):
    causal_offset: -1 = non-causal; >=0 = absolute position of first Q token
    mask:       2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool, True = keep)
    scale:      0.0 = auto (1/sqrt(head_dim)); >0 = explicit
    layout:     "bhld" (default) or "blhd"

Causal and mask can coexist — both are applied simultaneously.

Each wrapper dispatches to its compiled CUDA kernel (``astrai.extension.attn_*``)
when available, otherwise falls back to ``torch.nn.functional.scaled_dot_product_attention``.
"""

from astrai.extension.loader import KERNEL_NAMES, is_available
from astrai.extension.ops import attn_decode, attn_paged_decode, attn_prefill

__all__ = [
    "attn_decode",
    "attn_paged_decode",
    "attn_prefill",
    "is_available",
    "KERNEL_NAMES",
]
