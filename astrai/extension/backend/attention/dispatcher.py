"""Select one attention forward implementation for each call."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from astrai.extension.backend.attention import cuda, flash
from astrai.extension.backend.attention import torch as native
from astrai.extension.backend.attention.common import AttentionSpec
from astrai.extension.kernel.attention import _validate_cache_layout
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
    selected = _attention_dispatcher.dispatch(
        q, k, v, None, is_causal, scale=scale, mode="varlen"
    )
    return selected(q, k, v, cu_seqlens_q, cu_seqlens_k, is_causal, scale)


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
    _validate_cache_layout(q, cu_seqlens_q)
    selected = _attention_dispatcher.dispatch(
        q, k_new, v_new, None, is_causal, mode="cache"
    )
    return selected(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens,
        page_table,
        cu_seqlens_q,
        None,
        is_causal,
    )
