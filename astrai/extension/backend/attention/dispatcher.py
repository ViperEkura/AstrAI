"""Select one attention forward implementation for each call."""

from __future__ import annotations

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
    # Only reject ambiguous routing options here; each implementation checks tensors.
    if any(t is not None for t in (k_cache, v_cache, cache_seqlens, page_table)):
        if any(t is None for t in (k_cache, v_cache, cache_seqlens, page_table)):
            raise ValueError(
                "cache requires k_cache, v_cache, cache_seqlens and page_table"
            )
        if cu_seqlens_k is not None or scale is not None:
            raise ValueError(
                "cache attention does not accept cu_seqlens_k or custom scale"
            )
    else:
        if any(t is not None for t in workspace):
            raise ValueError("attention workspace requires a cache")
        if q.ndim == 4 and (cu_seqlens_q is not None or cu_seqlens_k is not None):
            raise ValueError("dense attention does not take sequence offsets")
        if q.ndim == 3:
            if cu_seqlens_q is None:
                raise ValueError("3D attention requires cu_seqlens_q without a cache")
            if mask is not None:
                raise ValueError("varlen attention does not accept an explicit mask")
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
