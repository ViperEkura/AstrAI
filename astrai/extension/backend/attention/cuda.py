"""Native CUDA attention for paged inference and consistent BF16 replay."""

from functools import partial
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from astrai.extension.backend.attention import torch as torch_attention
from astrai.extension.backend.attention.common import AttentionSpec, _check_fwd
from astrai.extension.kernel.attention import (
    ROLLOUT_KV_SPLIT_SIZE,
    attn_paged_decode,
    attn_paged_prefill,
    attn_prefill,
)
from astrai.extension.runtime.loader import is_available

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


HEAD_DIMS = (32, 64, 128, 256)


def supports(spec: AttentionSpec) -> bool:
    return (
        spec.device.type == "cuda"
        and not spec.needs_backward
        and spec.fwd in ("prefill", "decode")
        and spec.kv_cache_kind != "none"
        and (spec.scale is None or spec.scale > 0)
        and (
            spec.mask_kind == "none"
            or (
                spec.mask_kind == str(torch.bool)
                and 2 <= spec.mask_ndim <= 4
                and spec.mask_last_stride == 1
            )
        )
        and spec.ndim == 3
        and spec.dtype in (torch.bfloat16, torch.float16)
        and spec.head_dim in HEAD_DIMS
        and is_available("attention")
    )


def supports_consistent(spec: AttentionSpec) -> bool:
    if spec.dtype != torch.bfloat16:
        return False
    if spec.kv_cache_kind != "none":
        return supports(spec)
    return (
        spec.device.type == "cuda"
        and spec.ndim == 4
        and spec.head_dim in HEAD_DIMS
        and (spec.scale is None or spec.scale > 0)
        and (
            spec.mask_kind == "none"
            or (spec.mask_kind == str(torch.bool) and spec.mask_last_stride == 1)
        )
        and is_available("attention")
    )


_fixed_prefill = partial(attn_prefill, kv_split_size=ROLLOUT_KV_SPLIT_SIZE)


class _ConsistentPrefill(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, is_causal, scale):
        ctx.save_for_backward(q, k, v, mask)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return _fixed_prefill(q, k, v, mask, is_causal, scale=scale)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, mask = ctx.saved_tensors
        # Differentiate scaled dot-product attention with the same mask and
        # scale. The reference backward is independent of the tile schedule
        # and does not differentiate intermediate BF16 rounding.
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_(True) for t in (q, k, v))
            result = torch_attention.forward(
                *inputs, None, 0, mask, ctx.is_causal, scale=ctx.scale
            )
            grads = torch.autograd.grad(result, inputs, grad_output)
        return *grads, None, None, None


def forward_consistent(
    q,
    k,
    v,
    kv_cache,
    layer_id,
    attn_mask=None,
    is_causal=False,
    fwd=None,
    *,
    scale=None,
):
    if kv_cache is not None:
        return _fixed_paged_attention(
            q, k, v, kv_cache, layer_id, attn_mask, is_causal, fwd, scale=scale
        )
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _ConsistentPrefill.apply(q, k, v, attn_mask, is_causal, scale)
    return _fixed_prefill(q, k, v, attn_mask, is_causal, scale=scale)


def _paged_attention(
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
    prefill,
    decode,
) -> Tensor:
    _check_fwd(fwd)
    if kv_cache is None:
        raise RuntimeError("Paged CUDA attention requires a KV cache")
    if attn_mask is not None:
        _validate_paged_mask(q, kv_cache, attn_mask, fwd)
    paged = (
        q,
        kv_cache.k_buffer[layer_id],
        kv_cache.v_buffer[layer_id],
        kv_cache.req_to_token,
        kv_cache.req_pool_indices,
        kv_cache.kv_indptr,
    )
    if fwd == "decode":
        return decode(
            *paged,
            new_k=k,
            new_v=v,
            mask=attn_mask,
            is_causal=is_causal,
            scale=scale,
            o_part_buf=kv_cache.decode_o_part,
            ml_part_buf=kv_cache.decode_ml_part,
            out_buf=kv_cache.decode_out,
        )

    loc = kv_cache.out_cache_loc
    kv_cache.k_buffer[layer_id, loc] = k
    kv_cache.v_buffer[layer_id, loc] = v
    return prefill(
        *paged,
        kv_cache.qo_indptr,
        kv_cache.q_tile_to_batch,
        kv_cache.q_tile_to_index,
        attn_mask,
        is_causal=is_causal,
        scale=scale,
    )


forward = partial(
    _paged_attention, prefill=attn_paged_prefill, decode=attn_paged_decode
)
_fixed_paged_attention = partial(
    _paged_attention,
    prefill=partial(attn_paged_prefill, kv_split_size=ROLLOUT_KV_SPLIT_SIZE),
    decode=partial(attn_paged_decode, kv_split_size=ROLLOUT_KV_SPLIT_SIZE),
)
