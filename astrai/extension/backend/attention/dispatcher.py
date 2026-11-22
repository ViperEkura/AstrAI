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

    def build_spec(self, q, k, v, mask, is_causal, scale=None):
        return AttentionSpec(
            device=q.device,
            dtype=q.dtype,
            phase="train",
            needs_backward=torch.is_grad_enabled()
            and any(t.requires_grad for t in (q, k, v)),
            head_dim=q.size(-1),
            mask_kind="none" if mask is None else str(mask.dtype),
            mask_last_stride=0 if mask is None else mask.stride(-1),
            is_causal=is_causal,
            scale=scale,
        )

    def select(self, spec, choice, origin, *args, **kwargs):
        if choice in ("torch", "torch_native"):
            return native.forward
        if choice == "cuda":
            if cuda.supports(spec) and not spec.needs_backward:
                return cuda.forward
        elif choice == "cuda_consistent":
            if spec.dtype == torch.bfloat16 and cuda.supports(spec):
                return cuda.forward_consistent
        elif choice == "flash":
            if flash.supports(spec):
                return flash.forward
        elif choice is not None and origin in ("explicit", "context"):
            raise ValueError(f"Unknown attention implementation: {choice!r}")
        if choice is not None and origin in ("explicit", "context"):
            raise ExplicitSelectionError(
                f"Explicitly-set attention implementation {choice!r} cannot handle this call"
            )
        if flash.supports(spec):
            return flash.forward
        return native.forward


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
