"""Select one attention forward implementation for each call."""

from __future__ import annotations

import math
from typing import Optional

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

    def build_spec(self, q, k, v, mask, is_causal, scale=None, *, mode="dense"):
        return AttentionSpec(
            device=q.device,
            dtype=q.dtype,
            phase="train" if torch.is_grad_enabled() and mode == "dense" else "infer",
            mode=mode,
            needs_backward=torch.is_grad_enabled()
            and any(t.requires_grad for t in (q, k, v)),
            head_dim=q.size(-1),
            mask_kind="none" if mask is None else str(mask.dtype),
            mask_last_stride=0 if mask is None else mask.stride(-1),
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
                and spec.dtype == torch.bfloat16
                and cuda.supports(spec)
            ):
                return (
                    cuda.forward_consistent
                    if spec.mode == "dense"
                    else cuda.forward_cache_consistent
                )
        elif choice == "flash":
            if spec.mode != "varlen" and flash.supports(spec):
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
) -> Tensor:
    """Attend over dense [batch, sequence, heads, head_dim] tensors."""
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
    selected = _attention_dispatcher.dispatch(q, k, v, mask, is_causal, scale=scale)
    return selected(q, k, v, mask, is_causal, scale=scale)


def attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Optional[Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """Attend within each paired segment, using bottom-right causal alignment."""
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
    selected = _attention_dispatcher.dispatch(
        q, k, v, None, is_causal, scale=scale, mode="varlen"
    )
    return selected(q, k, v, qs, ks, is_causal, scale)


def _cache_attention(
    q, kc, vc, k, v, past, pages, qs, mask, causal, workspace=(None, None, None)
):
    """Tensor-only internal entry for model masks and graph workspace."""
    selected = _attention_dispatcher.dispatch(q, k, v, mask, causal, mode="cache")
    return selected(q, kc, vc, k, v, past, pages, qs, mask, causal, workspace)


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
    return _cache_attention(
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
