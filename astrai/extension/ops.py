"""GQA attention wrapper functions — one entry point per compiled kernel.

Each wrapper dispatches to its CUDA kernel (loaded in ``loader.py``) when
available, otherwise falls back to ``torch`` SDPA.

Add new kernel wrappers here; split into per-variant files only if this file
grows large.
"""

import torch
import torch.nn.functional as F

from astrai.extension.loader import _available, _modules


def _expand_kv_heads(
    k: torch.Tensor, v: torch.Tensor, q_head: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand K/V heads to match Q heads for GQA fallback."""
    kv_head = k.size(1)
    if kv_head == q_head:
        return k, v
    group = q_head // kv_head
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    return k, v


def _torch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    """Reference attention via ``scaled_dot_product_attention``."""
    k, v = _expand_kv_heads(k, v, q.size(1))
    attn_mask = mask[:, None, None, :] if mask is not None else None
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal and mask is None, scale=scale
    )


def attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
    causal_offset: int = 0,
    scale: float | None = None,
) -> torch.Tensor:
    if _available["attn_decode"]:
        return _modules["attn_decode"].attn_decode(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            causal_offset=causal_offset,
            scale=scale,
        )
    return _torch_fallback(q, k, v, mask, is_causal, scale)


def attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
    causal_offset: int = 0,
    scale: float | None = None,
) -> torch.Tensor:
    if _available["attn_prefill"]:
        return _modules["attn_prefill"].attn_prefill(
            q,
            k,
            v,
            mask=mask,
            is_causal=is_causal,
            causal_offset=causal_offset,
            scale=scale,
        )
    return _torch_fallback(q, k, v, mask, is_causal, scale)
