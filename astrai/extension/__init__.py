"""CUDA attention kernel wrappers with torch fallback.

Public API:
    - ``gqa_decode_attn`` — single-query decode attention
    - ``gqa_prefill_attn`` — multi-query prefill attention

Each wrapper dispatches to its compiled CUDA kernel (``astrai.extension.gqa_*``)
when available, otherwise falls back to ``torch.nn.functional.scaled_dot_product_attention``.
"""

from astrai.extension.loader import KERNEL_NAMES, is_available
from astrai.extension.ops import gqa_decode_attn, gqa_prefill_attn

__all__ = [
    "gqa_decode_attn",
    "gqa_prefill_attn",
    "is_available",
    "KERNEL_NAMES",
]
