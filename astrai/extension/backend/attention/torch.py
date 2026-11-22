"""Torch reference attention backend."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.backend.attention.common import repeat_kv


def _mask_view(mask: Optional[Tensor]) -> Optional[Tensor]:
    """Attention masks use batch/key, batch/query/key or batch/head/query/key."""
    if mask is None:
        return None
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise ValueError("attention mask must be boolean or floating point")
    if mask.ndim == 2:
        return mask[:, None, None, :]
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim != 4:
        raise ValueError("attention mask must be 2D, 3D or 4D")
    return mask


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    q_pos = torch.arange(q_len, device=device) + kv_len - q_len
    k_pos = torch.arange(kv_len, device=device)
    return q_pos[:, None] >= k_pos[None, :]


def _combine_mask(allowed: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return allowed
    if mask.dtype == torch.bool:
        return allowed & mask
    return mask.masked_fill(~allowed, float("-inf"))


def _packed_mask(mask: Optional[Tensor], allowed: Tensor, max_q: int) -> Tensor:
    """Apply the paged mask's broadcast and out-of-extent rules."""
    if mask is None:
        return allowed
    max_kv = allowed.size(-1)
    missing = False if mask.dtype == torch.bool else float("-inf")
    mask_rows = mask.size(-2)
    mask = mask[..., :max_q, :max_kv]
    if mask_rows != 1 and mask.size(-2) < max_q:
        mask = F.pad(mask, (0, 0, 0, max_q - mask.size(-2)), value=missing)
    if mask.size(-1) < max_kv:
        mask = F.pad(mask, (0, max_kv - mask.size(-1)), value=missing)
    return _combine_mask(allowed, mask)


def forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
) -> Tensor:
    n_rep = q.size(2) // k.size(2)
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    mask = _mask_view(attn_mask)
    sdpa_causal = is_causal
    if is_causal and (mask is not None or q.size(1) != k.size(1)):
        causal = _causal_mask(q.size(1), k.size(1), q.device)
        mask = _combine_mask(causal, mask)
        sdpa_causal = False
    return (
        F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            mask,
            is_causal=sdpa_causal,
            scale=scale,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
