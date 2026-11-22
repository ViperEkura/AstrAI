"""Select one attention forward implementation for each call."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from torch import Tensor

from astrai.extension.backend.attention import cuda, flash
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
        elif choice == "cuda_consistent":
            if cuda.supports_consistent(spec):
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
        if cuda.supports(spec):
            return cuda.forward
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
    return _dispatch(q, k, v, None, 0, mask, is_causal, None, scale, None)


def paged_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    kv_cache: "KVCache",
    layer_id: int,
    mask: Optional[Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
    implementation: Optional[str] = None,
) -> Tensor:
    """Append this layer's new K/V and attend over the bound request cache."""
    from astrai.model.kv_cache import DecodeKVCache, PrefillKVCache

    if isinstance(kv_cache, DecodeKVCache):
        mode = "decode"
    elif isinstance(kv_cache, PrefillKVCache):
        mode = "prefill"
    else:
        raise TypeError("paged attention requires a PrefillKVCache or DecodeKVCache")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("paged attention requires 3D Q, K and V")
    if not 0 <= layer_id < kv_cache.k_buffer.size(0):
        raise ValueError("layer_id is outside the KV cache")
    count = kv_cache.out_cache_loc.numel()
    if q.size(0) != count or k.size(0) != count or v.size(0) != count:
        raise ValueError("Q, K and V must each have one token per cache write location")
    return _dispatch(
        q, k, v, kv_cache, layer_id, mask, is_causal, mode, scale, implementation
    )


def _dispatch(
    q, k, v, kv_cache, layer_id, mask, is_causal, mode, scale, implementation
):
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")
    args = (q, k, v, kv_cache, layer_id, mask, is_causal, mode)
    selected = _attention_dispatcher.dispatch(
        *args, explicit=implementation, scale=scale
    )
    return selected(*args, scale=scale)
