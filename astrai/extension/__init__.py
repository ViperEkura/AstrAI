"""CUDA attention kernel wrappers with torch fallback.

Public API:
    - ``attn_decode`` — single-query decode attention
    - ``attn_prefill`` — multi-query prefill attention
    - ``attn_paged_decode`` — paged decode attention (direct page-table access)

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
