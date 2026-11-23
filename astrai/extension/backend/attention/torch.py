"""Torch reference attention backend."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from astrai.extension.backend.attention.common import _cache_view, repeat_kv


def _mask_view(mask):
    """Attention masks use batch/key, batch/query/key or batch/head/query/key."""
    if mask is None:
        return None
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise ValueError("attention mask must be boolean or floating point")
    if mask.ndim not in (2, 3, 4):
        raise ValueError("attention mask must be 2D, 3D or 4D")
    if mask.ndim == 2:
        return mask[:, None, None, :]
    if mask.ndim == 3:
        return mask[:, None, :, :]
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
    # Reference execution can run without the CUDA extension.
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("dense attention requires 4D Q/K/V")
    if (
        k.shape != v.shape
        or q.size(0) != k.size(0)
        or q.size(-1) != k.size(-1)
        or k.size(2) == 0
        or q.size(2) == 0
        or q.size(2) % k.size(2)
    ):
        raise ValueError("incompatible dense Q/K/V shapes or heads")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
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


def forward_varlen(q, k, v, qs, ks, is_causal, scale):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("varlen attention supports forward only")
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError("expected matching THD Q/K/V layouts")
    if ks is None and q.size(0) != k.size(0):
        raise ValueError("shared offsets require equal Q and KV token counts")
    ks = qs if ks is None else ks
    for ptr in (qs, ks):
        if (
            ptr.ndim != 1
            or ptr.numel() < 2
            or ptr.dtype != torch.int32
            or ptr.device != q.device
            or not ptr.is_contiguous()
        ):
            raise ValueError("expected contiguous int32 segment offsets on Q's device")
    if qs.numel() != ks.numel():
        raise ValueError("Q and KV segment counts must match")
    qoffs, koffs = qs.tolist(), ks.tolist()
    for offsets, tokens in ((qoffs, q.size(0)), (koffs, k.size(0))):
        if (
            offsets[0] != 0
            or offsets[-1] != tokens
            or any(a > b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("invalid segment offsets")
    result = []
    for i in range(len(qoffs) - 1):
        qr = q[qoffs[i] : qoffs[i + 1]]
        kr = k[koffs[i] : koffs[i + 1]]
        vr = v[koffs[i] : koffs[i + 1]]
        if not qr.size(0) or not kr.size(0):
            result.append(torch.zeros_like(qr))
        else:
            result.append(
                forward(
                    qr[None],
                    kr[None],
                    vr[None],
                    is_causal=is_causal,
                    scale=scale,
                )[0]
            )
    return torch.cat(result)


def forward_cache(
    q,
    kc,
    vc,
    k,
    v,
    past,
    pages,
    qs,
    mask,
    causal,
    workspace=(None, None, None),
    *,
    dense=forward,
):
    if (
        q.ndim not in (3, 4)
        or k.ndim != q.ndim
        or k.shape != v.shape
        or q.shape[:-2] != k.shape[:-2]
        or kc.ndim != 4
        or kc.shape != vc.shape
        or kc.size(1) <= 0
        or k.shape[-2:] != kc.shape[-2:]
        or q.size(-1) != k.size(-1)
        or k.size(-2) == 0
        or q.size(-2) == 0
        or q.size(-2) % k.size(-2)
        or not kc.is_contiguous()
        or not vc.is_contiguous()
        or any(t.device != q.device or t.dtype != q.dtype for t in (k, v, kc, vc))
    ):
        raise ValueError("incompatible cache and appended Q/K/V tensors")
    if (
        past.ndim != 1
        or past.numel() == 0
        or pages.ndim != 2
        or pages.size(0) != past.numel()
    ):
        raise ValueError("expected past lengths [B] and page table [B, blocks]")
    if qs is not None:
        if q.ndim != 3 or qs.ndim != 1 or qs.numel() != past.numel() + 1:
            raise ValueError("expected THD queries and [B+1] offsets")
    elif q.size(0) != past.numel():
        raise ValueError("expected BSHD queries or one THD row per request")
    for tensor in (past, pages, qs):
        if tensor is not None and (
            tensor.dtype != torch.int32
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("expected contiguous int32 metadata on Q's device")
    q, k, v, qs, original = _cache_view(q, k, v, qs)
    offsets = list(range(past.numel() + 1)) if qs is None else qs.tolist()
    lengths = past.tolist()
    size = kc.size(1)
    if (
        offsets[0] != 0
        or offsets[-1] != q.size(0)
        or any(a > b for a, b in zip(offsets, offsets[1:]))
    ):
        raise ValueError("invalid query offsets")
    mask = _mask_view(mask)
    if mask is not None and (
        mask.device != q.device
        or mask.size(0) not in (1, past.numel())
        or mask.size(1) not in (1, q.size(1))
        or not 0 < mask.size(-1) <= pages.size(1) * size
        or not 0 < mask.size(-2) <= (q.size(0) if qs is not None else 1)
    ):
        raise ValueError("invalid request-local paged mask")
    slots = []
    for i, n in enumerate(lengths):
        length = n + offsets[i + 1] - offsets[i]
        if n < 0 or length > pages.size(1) * size:
            raise ValueError("request exceeds cache capacity")
        active = pages[i, : (length + size - 1) // size]
        if any(page < 0 or page >= kc.size(0) for page in active.tolist()):
            raise ValueError("page index outside cache")
        pos = torch.arange(length, device=q.device)
        slots.append(active[pos // size].long() * size + pos % size)
    keys = kc.view(-1, *kc.shape[-2:])
    values = vc.view_as(keys)
    output = []
    for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
        sl = slots[i]
        keys[sl[lengths[i] :]] = k[start:end]
        values[sl[lengths[i] :]] = v[start:end]
        if start == end:
            output.append(q[start:end])
            continue
        local = None
        if mask is not None:
            item = mask[:1] if mask.size(0) == 1 else mask[i : i + 1]
            allowed = torch.ones(
                (1, 1, end - start, sl.numel()), dtype=torch.bool, device=q.device
            )
            local = _packed_mask(item, allowed, end - start)
        output.append(
            dense(
                q[None, start:end],
                keys[sl][None],
                values[sl][None],
                local,
                causal,
            )[0]
        )
    return torch.cat(output).reshape(original)
