"""Select one attention forward implementation for each call."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from astrai.extension.backend.attention import cuda, flash
from astrai.extension.backend.attention import torch as native
from astrai.extension.backend.attention.common import AttentionSpec
from astrai.extension.runtime.operation import ExplicitSelectionError, OpDispatcher


class AttentionDispatcher(OpDispatcher[AttentionSpec]):
    @property
    def operation(self) -> str:
        return "attention"

    def build_spec(self, q, k, v, mask, is_causal, scale=None, *, k_cache=None):
        mode = "cache" if k_cache is not None else "varlen" if q.ndim == 3 else "dense"
        needs_backward = torch.is_grad_enabled() and any(
            t.requires_grad for t in (q, k, v)
        )
        return AttentionSpec(
            device=q.device,
            dtype=q.dtype,
            phase="train" if needs_backward else "infer",
            mode=mode,
            needs_backward=needs_backward,
            head_dim=q.size(-1) if q.ndim else 0,
            mask_kind="none" if mask is None else str(mask.dtype),
            mask_last_stride=0 if mask is None or not mask.ndim else mask.stride(-1),
            is_causal=is_causal,
            scale=scale,
        )

    def select(self, spec, choice, origin, *args, **kwargs):
        name = {
            "dense": "forward",
            "varlen": "forward_varlen",
            "cache": "forward_cache",
        }[spec.mode]
        if choice in ("torch", "torch_native"):
            return getattr(native, name)
        if choice == "cuda":
            if cuda.supports(spec) and not spec.needs_backward:
                return getattr(cuda, name)
        elif choice == "cuda_consistent":
            if (
                spec.mode != "varlen"
                and (spec.mode == "dense" or not spec.needs_backward)
                and spec.dtype == torch.bfloat16
                and cuda.supports(spec)
            ):
                return (
                    cuda.forward_consistent
                    if spec.mode == "dense"
                    else cuda.forward_cache_consistent
                )
        elif choice == "flash":
            if (
                spec.mode != "varlen"
                and (spec.mode == "dense" or not spec.needs_backward)
                and flash.supports(spec)
            ):
                return getattr(flash, name)
        elif choice is not None and origin in ("explicit", "context"):
            raise ValueError(f"Unknown attention implementation: {choice!r}")
        if choice is not None and origin in ("explicit", "context"):
            raise ExplicitSelectionError(
                f"Explicitly-set attention implementation {choice!r} cannot handle this {spec.mode} call"
            )
        # Preserve dense training policy; packed and cache calls prefer native CUDA.
        if spec.mode != "dense" and cuda.supports(spec) and not spec.needs_backward:
            return getattr(cuda, name)
        if spec.mode == "dense" and flash.supports(spec):
            return flash.forward
        return getattr(native, name)


_attention_dispatcher = AttentionDispatcher()


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Optional[Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_k: Optional[Tensor] = None,
    k_cache: Optional[Tensor] = None,
    v_cache: Optional[Tensor] = None,
    cache_seqlens: Optional[Tensor] = None,
    page_table: Optional[Tensor] = None,
    workspace: Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]] = (
        None,
        None,
        None,
    ),
) -> Tensor:
    """Attend over dense or packed inputs, optionally appending K/V to a cache.

    Without a cache, BSHD selects dense attention and THD requires prefix sums.
    Cache inputs are BSHD, packed THD with offsets, or one THD row per request.
    Cache attention uses the default scale; varlen currently supports forward only.
    The inference runner owns graph scratch in ``workspace``.
    """
    q_shape, k_shape, v_shape = q.shape, k.shape, v.shape
    rank = len(q_shape)
    if rank not in (3, 4) or len(k_shape) != rank or len(v_shape) != rank:
        raise ValueError("attention requires matching 3D or 4D Q/K/V layouts")
    device, dtype = q.device, q.dtype
    if (
        k_shape != v_shape
        or q_shape[-1] != k_shape[-1]
        or q_shape[-1] == 0
        or k_shape[-2] == 0
        or q_shape[-2] == 0
        or q_shape[-2] % k_shape[-2]
        or not dtype.is_floating_point
        or k.device != device
        or v.device != device
        or k.dtype != dtype
        or v.dtype != dtype
    ):
        raise ValueError("incompatible Q/K/V shapes, heads, dtype or device")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    if mask is not None:
        if mask.dtype != torch.bool and not mask.is_floating_point():
            raise ValueError("attention mask must be boolean or floating point")
        if mask.ndim not in (2, 3, 4) or mask.device != device:
            raise ValueError("attention mask must be 2D, 3D or 4D on the query device")

    if (
        k_cache is not None
        or v_cache is not None
        or cache_seqlens is not None
        or page_table is not None
    ):
        if (
            k_cache is None
            or v_cache is None
            or cache_seqlens is None
            or page_table is None
        ):
            raise ValueError(
                "cache requires k_cache, v_cache, cache_seqlens and page_table"
            )
        if cu_seqlens_k is not None:
            raise ValueError("cache attention does not accept cu_seqlens_k")
        if scale is not None:
            raise ValueError("cache attention uses the default scale")
        past_shape, page_shape = cache_seqlens.shape, page_table.shape
        if (
            len(past_shape) != 1
            or past_shape[0] == 0
            or len(page_shape) != 2
            or page_shape[0] != past_shape[0]
        ):
            raise ValueError("expected past lengths [B] and page table [B, blocks]")
        cache_shape = k_cache.shape
        if (
            len(cache_shape) != 4
            or cache_shape != v_cache.shape
            or cache_shape[1] <= 0
            or not k_cache.is_contiguous()
            or not v_cache.is_contiguous()
        ):
            raise ValueError(
                "expected contiguous [pages, page_size, heads, dim] caches"
            )
        if (
            k_shape[:-2] != q_shape[:-2]
            or k_shape[-2:] != cache_shape[-2:]
            or k_cache.device != device
            or v_cache.device != device
            or k_cache.dtype != dtype
            or v_cache.dtype != dtype
        ):
            raise ValueError("incompatible cache and appended Q/K/V tensors")
        if cu_seqlens_q is not None:
            if rank == 4:
                raise ValueError("regular queries do not take offsets")
            if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() != past_shape[0] + 1:
                raise ValueError("expected THD queries and [B+1] offsets")
        elif q_shape[0] != past_shape[0]:
            raise ValueError("expected BSHD queries or one THD row per request")
        if rank == 4 and q_shape[1] == 0:
            raise ValueError("expected nonempty regular queries")
        if mask is not None:
            batch = past_shape[0]
            max_q = (
                q_shape[1]
                if rank == 4
                else q_shape[0]
                if cu_seqlens_q is not None
                else 1
            )
            mask_heads = mask.size(1) if mask.ndim == 4 else 1
            mask_rows = mask.size(-2) if mask.ndim >= 3 else 1
            if (
                mask.size(0) not in (1, batch)
                or mask_heads not in (1, q_shape[-2])
                or not 0 < mask.size(-1) <= page_shape[1] * cache_shape[1]
                or not 0 < mask_rows <= max_q
            ):
                raise ValueError("invalid request-local paged mask")
    else:
        if any(t is not None for t in workspace):
            raise ValueError("attention workspace requires a cache")
        if rank == 4:
            if cu_seqlens_q is not None or cu_seqlens_k is not None:
                raise ValueError("dense attention does not take sequence offsets")
            if q_shape[0] != k_shape[0]:
                raise ValueError("dense Q/K/V batch sizes must match")
        else:
            if cu_seqlens_q is None:
                raise ValueError("3D attention requires cu_seqlens_q without a cache")
            if mask is not None:
                raise ValueError("varlen attention does not accept an explicit mask")
            if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
                raise RuntimeError("varlen attention supports forward only")
            if cu_seqlens_k is None and q_shape[0] != k_shape[0]:
                raise ValueError("shared offsets require equal Q and KV token counts")
            if (
                cu_seqlens_k is not None
                and cu_seqlens_q.numel() != cu_seqlens_k.numel()
            ):
                raise ValueError("Q and KV segment counts must match")
    for ptr in (cu_seqlens_q, cu_seqlens_k):
        if ptr is not None:
            if ptr.ndim != 1 or ptr.numel() < 2:
                raise ValueError("expected segment offsets [B+1]")
    for tensor in (cu_seqlens_q, cu_seqlens_k, cache_seqlens, page_table):
        if tensor is not None and (
            tensor.dtype != torch.int32
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError("expected contiguous int32 metadata on the query device")
    selected = _attention_dispatcher.dispatch(
        q, k, v, mask, is_causal, scale=scale, k_cache=k_cache
    )
    if k_cache is not None:
        return selected(
            q,
            k_cache,
            v_cache,
            k,
            v,
            cache_seqlens,
            page_table,
            cu_seqlens_q,
            mask,
            is_causal,
            workspace,
        )
    if q.ndim == 3:
        return selected(q, k, v, cu_seqlens_q, cu_seqlens_k, is_causal, scale)
    return selected(q, k, v, mask, is_causal, scale=scale)
