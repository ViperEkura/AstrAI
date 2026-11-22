"""Attention contracts for non-native execution.

CUDA validates while packing its native arguments. Torch/Flash reuse these
checks instead; backend selection never duplicates tensor validation.
"""

import math

import torch


def dense(q, k, v, scale=None):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("dense attention requires 4D Q, K and V")
    if (
        k.shape != v.shape
        or q.size(0) != k.size(0)
        or q.size(-1) != k.size(-1)
        or k.size(2) == 0
        or q.size(2) == 0
        or q.size(2) % k.size(2)
        or any(t.device != q.device or t.dtype != q.dtype for t in (k, v))
    ):
        raise ValueError("incompatible dense Q/K/V shapes, heads, dtype or device")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")


def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k=None, scale=None):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError("attention_varlen supports forward only")
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError("expected THD inputs with matching K/V shapes")
    if (
        k.size(1) == 0
        or q.size(1) == 0
        or q.size(1) % k.size(1)
        or q.size(2) != k.size(2)
        or q.size(2) == 0
        or not q.is_floating_point()
        or any(t.dtype != q.dtype or t.device != q.device for t in (k, v))
    ):
        raise ValueError("incompatible varlen heads, dtype or device")
    qs = cu_seqlens_q
    ks = qs if cu_seqlens_k is None else cu_seqlens_k
    if cu_seqlens_k is None and q.size(0) != k.size(0):
        raise ValueError("shared offsets require equal Q and KV token counts")
    for ptr in (qs, ks):
        if (
            ptr.device != q.device
            or ptr.dtype != torch.int32
            or ptr.ndim != 1
            or ptr.numel() < 2
            or not ptr.is_contiguous()
        ):
            raise ValueError(
                "expected contiguous int32 segment offsets on the token device"
            )
    if qs.numel() != ks.numel():
        raise ValueError("Q and KV segment counts must match")
    scale = q.size(-1) ** -0.5 if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    return qs, ks, scale


def cache_layout(q, cu_seqlens_q):
    if q.ndim not in (3, 4) or (q.ndim == 3 and cu_seqlens_q is None):
        raise ValueError("expected BSHD queries or THD queries with offsets")
    if q.ndim == 4 and cu_seqlens_q is not None:
        raise ValueError("regular queries do not take offsets")


def cache(q, k_cache, v_cache, k_new, v_new, cache_seqlens, page_table, cu_seqlens_q):
    if q.ndim not in (3, 4) or k_new.ndim != q.ndim or v_new.ndim != q.ndim:
        raise ValueError("expected matching THD or BSHD Q/K/V")
    if (
        cache_seqlens.ndim != 1
        or cache_seqlens.numel() == 0
        or page_table.ndim != 2
        or page_table.size(0) != cache_seqlens.numel()
    ):
        raise ValueError("expected past lengths [B] and page table [B, blocks]")
    if (
        k_cache.ndim != 4
        or k_cache.shape != v_cache.shape
        or k_cache.size(1) <= 0
        or not k_cache.is_contiguous()
        or not v_cache.is_contiguous()
    ):
        raise ValueError("expected contiguous [pages, page_size, heads, dim] caches")
    if (
        k_new.shape != v_new.shape
        or k_new.shape[:-2] != q.shape[:-2]
        or k_new.shape[-2:] != k_cache.shape[-2:]
        or q.size(-1) != k_cache.size(-1)
        or k_new.size(-2) == 0
        or q.size(-2) == 0
        or q.size(-2) % k_new.size(-2)
        or not q.is_floating_point()
        or any(
            t.device != q.device or t.dtype != q.dtype
            for t in (k_new, v_new, k_cache, v_cache)
        )
    ):
        raise ValueError("incompatible cache and appended Q/K/V tensors")
    original = q.shape
    qs = cu_seqlens_q
    metadata = [cache_seqlens, page_table]
    if qs is not None:
        if q.ndim != 3 or qs.ndim != 1 or qs.numel() != cache_seqlens.numel() + 1:
            raise ValueError("expected THD queries and [B+1] offsets")
        metadata.append(qs)
    else:
        if q.ndim not in (3, 4) or q.size(0) != cache_seqlens.numel():
            raise ValueError("expected BSHD queries or one THD row per request")
        if q.ndim == 4:
            if q.size(1) == 0:
                raise ValueError("expected nonempty regular queries")
            if q.size(1) > 1:
                qs = torch.arange(
                    q.size(0) + 1, device=q.device, dtype=torch.int32
                ) * q.size(1)
            q, k_new, v_new = (t.flatten(0, 1) for t in (q, k_new, v_new))
    if any(
        t.dtype != torch.int32 or t.device != q.device or not t.is_contiguous()
        for t in metadata
    ):
        raise ValueError("expected contiguous int32 metadata on the query device")
    return q, k_new, v_new, qs, original


def mask_view(mask):
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


def segment_offsets(q, k, qs, ks):
    qoffs, koffs = qs.tolist(), ks.tolist()
    for offsets, tokens in ((qoffs, q.size(0)), (koffs, k.size(0))):
        if (
            offsets[0] != 0
            or offsets[-1] != tokens
            or any(a > b for a, b in zip(offsets, offsets[1:]))
        ):
            raise ValueError("invalid segment offsets")
    return qoffs, koffs


def cache_slots(q, kc, past, pages, qs, mask):
    offsets = list(range(past.numel() + 1)) if qs is None else qs.tolist()
    lengths = past.tolist()
    size = kc.size(1)
    if (
        offsets[0] != 0
        or offsets[-1] != q.size(0)
        or any(a > b for a, b in zip(offsets, offsets[1:]))
    ):
        raise ValueError("invalid query offsets")
    mask = mask_view(mask)
    if mask is not None and (
        mask.device != q.device
        or mask.size(0) not in (1, len(lengths))
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
    return offsets, lengths, slots, mask
