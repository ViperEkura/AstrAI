"""Optional FlashAttention strategy."""

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

import astrai.extension.backend.attention as _attention
from astrai.extension.runtime.dispatch import Axes

from . import (
    ATTN_BACKEND,
    AttentionBackend,
    AttentionBackendFactory,
    _axes,
    _flash_attn,
)

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


@AttentionBackendFactory.register(ATTN_BACKEND.FLASH.value)
class FlashAttnBackend(AttentionBackend):
    """FlashAttention backend via the optional ``flash-attn`` package.

    Decode (q_len=1, contiguous cache): writes K/V to the pool, gathers
    flat K/V via the ``req_to_token`` page table, and calls
    ``flash_attn_varlen_func`` over the ragged batch
    (``qo_indptr``/``kv_indptr``).

    Prefill: packed 3-D calls share the ``flash_attn_varlen_func`` path;
    dense 4-D calls go through ``flash_attn_func`` (mask-free only).
    """

    supports_scale = True
    priority = 10

    @classmethod
    def available(cls) -> bool:
        return _attention.flash_attn_available()

    @classmethod
    def supports_axes(cls, ax: Axes) -> bool:
        if not _attention.flash_attn_available():
            return False
        if ax["has_mask"]:
            return False
        if ax["dtype"] not in (torch.float16, torch.bfloat16):
            return False
        if ax["fwd"] is not None:
            return ax["ndim"] == 3 and hasattr(_flash_attn, "flash_attn_varlen_func")
        # Dense training cannot apply a custom mask; causal is a flag.
        return not ax["has_mask"]

    def supports_call(
        self,
        q: Tensor,
        kv_cache: Optional["KVCache"],
        attn_mask: Optional[Tensor],
        is_causal: bool,
        fwd: Optional[str],
        *,
        scale: Optional[float] = None,
    ) -> bool:
        return self.supports_axes(_axes(q, kv_cache, attn_mask, is_causal, fwd, scale))

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: Optional["KVCache"],
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        fwd: Optional[str] = None,
        *,
        scale: Optional[float] = None,
    ) -> Tensor:
        self._check_fwd(fwd)
        if attn_mask is not None:
            raise ValueError("FlashAttnBackend cannot handle a custom attention mask")
        # Decode is always packed; prefill/training split by layout.
        if fwd == "decode" or q.ndim == 3:
            return self._forward_packed(q, k, v, kv_cache, layer_id, is_causal, scale)
        return self._forward_dense(q, k, v, attn_mask, is_causal, scale)

    def _forward_dense(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> Tensor:
        if attn_mask is not None:
            raise ValueError(
                "FlashAttnBackend cannot handle a custom attention mask; "
                "use a causal mask or select TorchNativeBackend."
            )
        fa = _flash_attn
        if fa is None:
            raise RuntimeError(
                "FlashAttnBackend requires the optional 'flash-attn' package. "
                "Install with `pip install flash-attn`."
            )
        out = fa.flash_attn_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            causal=is_causal,
            softmax_scale=scale,
        )
        return out.contiguous()

    def _forward_packed(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: "KVCache",
        layer_id: int,
        is_causal: bool,
        scale: Optional[float],
    ) -> Tensor:
        fa = _flash_attn
        if fa is None or not hasattr(fa, "flash_attn_varlen_func"):
            raise RuntimeError("packed inference requires flash_attn_varlen_func")
        kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
        kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v
        page_table = kv_cache.req_to_token[
            kv_cache.req_pool_indices, : kv_cache.max_len
        ]
        positions = torch.arange(kv_cache.max_len, device=q.device)
        indices = page_table[positions.unsqueeze(0) < kv_cache.seq_lens.unsqueeze(1)]
        k_flat = kv_cache.k_buffer[layer_id, indices].contiguous()
        v_flat = kv_cache.v_buffer[layer_id, indices].contiguous()
        out = fa.flash_attn_varlen_func(
            q.contiguous(),
            k_flat,
            v_flat,
            kv_cache.qo_indptr,
            kv_cache.kv_indptr,
            int((kv_cache.qo_indptr[1:] - kv_cache.qo_indptr[:-1]).max()),
            int(kv_cache.seq_lens.max()),
            dropout_p=0.0,
            causal=is_causal,
            softmax_scale=scale,
        )
        return out
