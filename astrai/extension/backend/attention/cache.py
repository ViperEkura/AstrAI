"""Append new K/V and attend over a batch-ordered physical page table."""

from typing import Optional

import torch
from torch import Tensor

from astrai.extension.backend.attention import torch as reference
from astrai.extension.kernel.attention import ROLLOUT_KV_SPLIT_SIZE
from astrai.extension.runtime.context import op_context
from astrai.extension.runtime.loader import get_module, is_available
from astrai.extension.runtime.operation import ExplicitSelectionError


def _reference(q, kc, vc, k, v, past, pages, qs, mask, causal):
    offsets = list(range(past.numel() + 1)) if qs is None else qs.tolist()
    lengths = past.tolist()
    size = kc.size(1)
    if (
        offsets[0] != 0
        or offsets[-1] != q.size(0)
        or any(a > b for a, b in zip(offsets, offsets[1:]))
    ):
        raise ValueError("invalid query offsets")
    mask = reference._mask_view(mask)
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
        if ((active < 0) | (active >= kc.size(0))).any():
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
            local = reference._packed_mask(item, allowed, end - start)
        output.append(
            reference.forward(
                q[None, start:end],
                keys[sl][None],
                values[sl][None],
                None,
                0,
                local,
                causal,
            )[0]
        )
    return torch.cat(output)


def _cache_impl(
    q, kc, vc, k, v, past, pages, qs, mask, causal, workspace=(None, None, None)
):
    choice, origin = op_context.snapshot().choice("attention")
    native = (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.size(-1) in (32, 64, 128, 256)
        and (mask is None or (mask.dtype == torch.bool and mask.stride(-1) == 1))
        and is_available("attention")
    )
    if (
        native
        and choice in (None, "auto", "cuda", "cuda_consistent")
        and (choice != "cuda_consistent" or q.dtype == torch.bfloat16)
    ):
        split = ROLLOUT_KV_SPLIT_SIZE if choice == "cuda_consistent" else 0
        return get_module("attention")._cache_forward(
            q, kc, vc, k, v, past, pages, qs, mask, causal, split, *workspace
        )
    if choice == "flash" and q.is_cuda and mask is None:
        from astrai.extension.backend.attention.common import _flash_attn

        if hasattr(_flash_attn, "flash_attn_varlen_func"):
            # Optional FA2 accepts paged cache blocks in multiples of 256.
            if qs is None and kc.size(1) % 256 == 0:
                return _flash_attn.flash_attn_with_kvcache(
                    q[:, None],
                    kc,
                    vc,
                    k=k[:, None],
                    v=v[:, None],
                    cache_seqlens=past,
                    block_table=pages,
                    causal=causal,
                )[:, 0]
    if origin == "context" and choice not in ("auto", "torch", "torch_native"):
        raise ExplicitSelectionError(
            f"{choice!r} cannot handle this cache attention call"
        )
    return _reference(q, kc, vc, k, v, past, pages, qs, mask, causal)


def attention_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    *,
    k_new: Tensor,
    v_new: Tensor,
    cache_seqlens: Tensor,
    page_table: Tensor,
    cu_seqlens_q: Optional[Tensor] = None,
    is_causal: bool = False,
) -> Tensor:
    """Append at each request's past length and return the attended queries."""
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
        if q.ndim != 4 or q.size(0) != cache_seqlens.numel() or q.size(1) == 0:
            raise ValueError("expected BSHD queries")
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
    return _cache_impl(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens,
        page_table,
        qs,
        None,
        is_causal,
    ).reshape(original)
