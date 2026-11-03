"""Optional FlashAttention-3 backend for Hopper attention layouts."""

import functools
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from astrai.extension.backend.attention.common import AttentionSpec, _check_fwd
from astrai.model.masking import PackedDocumentMask

try:
    from flash_attn_3 import flash_attn_interface as _flash3
except ImportError:
    _flash3 = None

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


@functools.lru_cache(maxsize=None)
def available(device: torch.device) -> bool:
    """FA3 targets Hopper; probe the installed extension once per device."""
    if (
        _flash3 is None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability(device)[0] != 9
    ):
        return False
    try:
        x = torch.zeros(1, 1, 1, 64, device=device, dtype=torch.bfloat16)
        return bool(torch.isfinite(_flash3.flash_attn_func(x, x, x, causal=True)).all())
    except Exception:
        return False


def supports(spec: AttentionSpec) -> bool:
    if (
        spec.device.type != "cuda"
        or spec.dtype not in (torch.float16, torch.bfloat16)
        or not available(spec.device)
    ):
        return False
    if spec.ndim == 4 and spec.fwd is None:
        return spec.mask_kind in ("none", "packed_doc")
    if spec.ndim == 3 and spec.fwd in ("prefill", "decode"):
        return spec.mask_kind == "none" and spec.kv_cache_kind != "none"
    return False


def forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"],
    layer_id: int,
    attn_mask: Optional[Tensor | PackedDocumentMask] = None,
    is_causal: bool = False,
    fwd: Optional[str] = None,
    *,
    scale: Optional[float] = None,
) -> Tensor:
    _check_fwd(fwd)
    if q.ndim == 4 and fwd is None and kv_cache is None:
        return forward_4d(q, k, v, attn_mask, is_causal, scale)
    if q.ndim == 3 and fwd is not None and kv_cache is not None:
        return forward_3d(q, k, v, kv_cache, layer_id, attn_mask, is_causal, scale)
    raise ValueError("FA3 requires 4D training or 3D packed inference inputs")


def forward_4d(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor | PackedDocumentMask],
    is_causal: bool,
    scale: Optional[float],
) -> Tensor:
    """Run dense or document-packed training attention."""
    if _flash3 is None:
        raise RuntimeError("FA3 requires the optional flash_attn_3 package")
    if isinstance(attn_mask, PackedDocumentMask):
        batch, seq_len = q.shape[:2]
        out = _flash3.flash_attn_varlen_func(
            q.reshape(batch * seq_len, *q.shape[2:]).contiguous(),
            k.reshape(batch * seq_len, *k.shape[2:]).contiguous(),
            v.reshape(batch * seq_len, *v.shape[2:]).contiguous(),
            attn_mask.cu_seqlens,
            attn_mask.cu_seqlens,
            attn_mask.max_seqlen,
            attn_mask.max_seqlen,
            causal=True,
            softmax_scale=scale,
        )
        return out.reshape(q.shape).contiguous()
    if attn_mask is not None:
        raise ValueError("FA3 cannot handle a dense custom attention mask")
    return _flash3.flash_attn_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        causal=is_causal,
        softmax_scale=scale,
    ).contiguous()


def forward_3d(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: "KVCache",
    layer_id: int,
    attn_mask: Optional[Tensor],
    is_causal: bool,
    scale: Optional[float],
) -> Tensor:
    """Gather paged KV into request order and run packed inference attention."""
    if _flash3 is None:
        raise RuntimeError("FA3 requires the optional flash_attn_3 package")
    if attn_mask is not None:
        raise ValueError("FA3 packed inference cannot handle a custom mask")
    kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
    kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v
    page_table = kv_cache.req_to_token[kv_cache.req_pool_indices, : kv_cache.max_len]
    positions = torch.arange(kv_cache.max_len, device=q.device)
    indices = page_table[positions.unsqueeze(0) < kv_cache.seq_lens.unsqueeze(1)]
    k_flat = kv_cache.k_buffer[layer_id, indices].contiguous()
    v_flat = kv_cache.v_buffer[layer_id, indices].contiguous()
    return _flash3.flash_attn_varlen_func(
        q.contiguous(),
        k_flat,
        v_flat,
        kv_cache.qo_indptr,
        kv_cache.kv_indptr,
        int((kv_cache.qo_indptr[1:] - kv_cache.qo_indptr[:-1]).max()),
        int(kv_cache.seq_lens.max()),
        causal=is_causal,
        softmax_scale=scale,
    )
