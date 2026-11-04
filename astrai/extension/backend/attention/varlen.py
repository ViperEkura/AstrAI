"""Built-in PyTorch variable-length attention for packed documents."""

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.varlen import varlen_attn

from astrai.attention_layout import VarlenAttentionLayout
from astrai.extension.backend.attention.common import AttentionSpec, _check_fwd

if TYPE_CHECKING:
    from astrai.model.kv_cache import KVCache


def supports(spec: AttentionSpec) -> bool:
    if (
        spec.device.type != "cuda"
        or spec.dtype not in (torch.float16, torch.bfloat16)
        or not torch.backends.cuda.is_flash_attention_available()
    ):
        return False
    if spec.mask_kind == "varlen":
        return spec.ndim == 4 and spec.fwd is None and spec.kv_cache_kind == "none"
    return (
        spec.mask_kind == "none"
        and spec.ndim == 3
        and spec.fwd in ("prefill", "decode")
        and spec.kv_cache_kind != "none"
    )


def forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: Optional["KVCache"],
    layer_id: int,
    attn_mask: Optional[Tensor | VarlenAttentionLayout] = None,
    is_causal: bool = False,
    fwd: Optional[str] = None,
    *,
    scale: Optional[float] = None,
) -> Tensor:
    _check_fwd(fwd)
    if isinstance(attn_mask, VarlenAttentionLayout):
        if q.ndim != 4 or kv_cache is not None or fwd is not None:
            raise ValueError("varlen training requires 4D QKV without a KV cache")
        return forward_4d(q, k, v, attn_mask, scale)
    if q.ndim == 3 and kv_cache is not None and fwd is not None:
        if attn_mask is not None:
            raise ValueError("varlen inference cannot use a custom attention mask")
        return forward_3d(q, k, v, kv_cache, layer_id, is_causal, scale)
    raise ValueError("varlen attention requires segment or KV-cache metadata")


def forward_4d(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    layout: VarlenAttentionLayout,
    scale: Optional[float],
) -> Tensor:
    """Unpad a training batch, attend within each document, and restore rows."""
    batch, seq_len, q_heads, head_dim = q.shape
    indices = layout.indices
    cu = layout.cu_seqlens
    if indices.device != q.device or cu.device != q.device:
        raise ValueError("varlen offsets and QKV must be on the same device")
    q_flat = q.reshape(batch * seq_len, q_heads, head_dim)
    k_flat = k.reshape(batch * seq_len, *k.shape[2:])
    v_flat = v.reshape(batch * seq_len, *v.shape[2:])
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = varlen_attn(
            q_flat.index_select(0, indices),
            k_flat.index_select(0, indices),
            v_flat.index_select(0, indices),
            cu,
            cu,
            layout.max_seqlen,
            layout.max_seqlen,
            scale=scale,
            window_size=(-1, 0),
            enable_gqa=q_heads != k.size(2),
        )
    return torch.zeros_like(q_flat).index_copy(0, indices, out).reshape(q.shape)


def forward_3d(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    kv_cache: "KVCache",
    layer_id: int,
    is_causal: bool,
    scale: Optional[float],
) -> Tensor:
    """Gather paged KV into request order for packed inference."""
    kv_cache.k_buffer[layer_id, kv_cache.out_cache_loc] = k
    kv_cache.v_buffer[layer_id, kv_cache.out_cache_loc] = v
    page_table = kv_cache.req_to_token[kv_cache.req_pool_indices, : kv_cache.max_len]
    positions = torch.arange(kv_cache.max_len, device=q.device)
    indices = page_table[positions.unsqueeze(0) < kv_cache.seq_lens.unsqueeze(1)]
    k_flat = kv_cache.k_buffer[layer_id, indices].contiguous()
    v_flat = kv_cache.v_buffer[layer_id, indices].contiguous()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return varlen_attn(
            q.contiguous(),
            k_flat,
            v_flat,
            kv_cache.qo_indptr,
            kv_cache.kv_indptr,
            int((kv_cache.qo_indptr[1:] - kv_cache.qo_indptr[:-1]).max()),
            int(kv_cache.seq_lens.max()),
            scale=scale,
            window_size=(-1, 0) if is_causal else (-1, -1),
            enable_gqa=q.size(1) != k_flat.size(1),
        )
