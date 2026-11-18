"""Select one attention forward implementation for each call."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from torch import Tensor

from astrai.extension.backend.attention import cuda, flash, varlen
from astrai.extension.backend.attention import torch as native
from astrai.extension.backend.attention.common import AttentionSpec, _attention_spec
from astrai.extension.runtime.operation import ExplicitSelectionError, OpDispatcher

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


class AttentionDispatcher(OpDispatcher[AttentionSpec]):
    @property
    def operation(self) -> str:
        return "attention"

    def build_spec(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
        scale: Optional[float] = None,
    ) -> AttentionSpec:
        return _attention_spec(q, k, v, kv_cache, attn_mask, is_causal, fwd, scale)

    def select(
        self,
        spec: AttentionSpec,
        choice: Optional[str],
        origin: str,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
        scale: Optional[float] = None,
    ):
        if choice in ("torch", "torch_native"):
            return native.forward
        if choice == "cuda":
            if cuda.supports(spec):
                return cuda.forward
        elif choice == "flash":
            if flash.supports(spec):
                return flash.forward
        elif choice == "varlen":
            if varlen.supports(spec):
                return varlen.forward
        elif choice is not None and origin in ("explicit", "context"):
            raise ValueError(f"Unknown attention implementation: {choice!r}")
        if choice is not None and origin in ("explicit", "context"):
            raise ExplicitSelectionError(
                f"Explicitly-set attention implementation {choice!r} cannot handle this call"
            )
        if cuda.supports(spec):
            return cuda.forward
        if flash.supports(spec):
            return flash.forward
        if varlen.supports(spec):
            return varlen.forward
        return native.forward


_attention_dispatcher = AttentionDispatcher()


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"] = None,
    layer_id: int = 0,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    fwd: Optional[str] = None,
    implementation: Optional[str] = None,
    *,
    scale: Optional[float] = None,
) -> Tensor:
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    args = (q, k, v, kv_cache, layer_id, attn_mask, is_causal, fwd)
    selected = _attention_dispatcher.dispatch(
        *args, explicit=implementation, scale=scale
    )
    return selected(*args, scale=scale)
