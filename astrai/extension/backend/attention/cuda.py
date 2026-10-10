"""Native CUDA attention backend for packed KV-cache inference."""

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from astrai.extension.kernel.attention import attn_paged_decode, attn_paged_prefill
from astrai.extension.runtime.dispatch import Axes
from astrai.extension.runtime.loader import is_available

from . import ATTN_BACKEND, AttentionBackend, AttentionBackendFactory, _axes

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


def _validate_paged_mask(
    q: Tensor, kv_cache: "KVCache", mask: Tensor, fwd: Optional[str]
) -> None:
    """Check host-visible mask metadata before writing the KV cache."""
    if mask.ndim not in (2, 3, 4):
        raise ValueError("paged mask must be 2D, 3D or 4D")
    if mask.device != q.device or mask.dtype != torch.bool:
        raise ValueError("paged mask must be boolean on Q's device")
    if mask.stride(-1) != 1:
        raise ValueError("paged mask key axis must be contiguous")
    if mask.size(0) not in (1, kv_cache.req_pool_indices.numel()):
        raise ValueError("paged mask batch mismatch")
    if mask.ndim == 4 and mask.size(1) not in (1, q.size(1)):
        raise ValueError("paged mask head mismatch")
    if not 0 < mask.size(-1) <= kv_cache.req_to_token.size(1):
        raise ValueError("paged mask key axis exceeds cache capacity")
    if mask.ndim >= 3:
        max_rows = 1 if fwd == "decode" else q.size(0)
        if not 0 < mask.size(-2) <= max_rows:
            raise ValueError("paged mask query axis must use request-local rows")


@AttentionBackendFactory.register(ATTN_BACKEND.CUDA.value)
class CudaBackend(AttentionBackend):
    """CUDA kernel backend with direct KV cache access.

    Decode path: writes K/V to the flat pool, then calls
    ``attn_paged_decode`` with req_to_token + kv_indptr.

    Prefill path: writes K/V to the flat pool, then calls
    ``attn_paged_prefill`` with ragged-batch support via qo_indptr +
    kv_indptr.

    ``kv_cache is None`` (training) raises — the per-call fallback to
    torch SDPA for training / fp32 / unsupported head_dim happens in the
    ``attention()`` entry point.

    Raises ``RuntimeError`` if the required kernel is not available.
    """

    # Head dims supported by the CUDA kernels (single source of truth).
    HEAD_DIMS = (32, 64, 128, 256)
    supports_scale = True
    priority = 0
    modes = frozenset(("infer",))

    @classmethod
    def available(cls) -> bool:
        return torch.cuda.is_available() and is_available("attention")

    @classmethod
    def supports_axes(cls, ax: Axes) -> bool:
        mask = ax["_call"][2]
        # The CUDA kernels take one precision per build — bf16 today, and the
        # instantiated set lives in csrc/include/api/attention_dtypes.h.
        return (
            ax["fwd"] in ("prefill", "decode")
            and ax["has_cache"]
            and (ax.get("scale") is None or ax["scale"] > 0)
            and (
                mask is None
                or (
                    mask.dtype == torch.bool
                    and 2 <= mask.ndim <= 4
                    and mask.stride(-1) == 1
                )
            )
            and ax["ndim"] == 3
            and ax["dtype"] == torch.bfloat16
            and ax["head_dim"] in cls.HEAD_DIMS
            and is_available("attention")
        )

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

    @staticmethod
    def supports_graph() -> bool:
        return True

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
        if kv_cache is None:
            raise RuntimeError("CudaBackend does not support training (kv_cache=None)")
        if attn_mask is not None:
            _validate_paged_mask(q, kv_cache, attn_mask, fwd)
        if fwd == "decode":
            return self._decode(
                q, k, v, kv_cache, layer_id, attn_mask, is_causal, scale
            )
        return self._prefill(q, k, v, kv_cache, layer_id, attn_mask, is_causal, scale)

    def _decode(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: "KVCache",
        layer_id: int,
        attn_mask: Optional[Tensor],
        is_causal: bool,
        scale: Optional[float],
    ) -> Tensor:
        kv_indptr = kv_cache.kv_indptr

        out = attn_paged_decode(
            q,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_indptr,
            new_k=k,
            new_v=v,
            mask=attn_mask,
            is_causal=is_causal,
            scale=scale,
            o_part_buf=kv_cache.decode_o_part,
            ml_part_buf=kv_cache.decode_ml_part,
            out_buf=kv_cache.decode_out,
        )
        return out

    def _prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        kv_cache: "KVCache",
        layer_id: int,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> Tensor:
        loc = kv_cache.out_cache_loc
        kv_cache.k_buffer[layer_id, loc] = k
        kv_cache.v_buffer[layer_id, loc] = v

        out = attn_paged_prefill(
            q,
            kv_cache.k_buffer[layer_id],
            kv_cache.v_buffer[layer_id],
            kv_cache.req_to_token,
            kv_cache.req_pool_indices,
            kv_cache.kv_indptr,
            kv_cache.qo_indptr,
            kv_cache.q_tile_to_batch,
            kv_cache.q_tile_to_index,
            attn_mask,
            is_causal=is_causal,
            scale=scale,
        )
        return out
