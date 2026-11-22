"""Stateless adapters for the three compiled attention entry points.

Backend policy, fallback and autograd live in backend/attention.
Scratch belongs to the caller and is never retained by these wrappers.
"""

from typing import Optional, Tuple

from torch import Tensor

from astrai.extension.runtime.loader import get_module

ROLLOUT_KV_SPLIT_SIZE = 256


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
    kv_split_size: int = 0,
) -> Tensor:
    return get_module("attention").attention(
        q, k, v, mask, scale, is_causal, kv_split_size
    )


def attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    is_causal: bool,
    scale: float,
) -> Tensor:
    return get_module("attention").attention_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, is_causal, scale
    )


def attention_with_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    cache_seqlens: Tensor,
    page_table: Tensor,
    cu_seqlens_q: Optional[Tensor],
    mask: Optional[Tensor],
    is_causal: bool,
    *,
    workspace: Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]] = (
        None,
        None,
        None,
    ),
    kv_split_size: int = 0,
) -> Tensor:
    return get_module("attention").attention_with_kvcache(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        cache_seqlens,
        page_table,
        cu_seqlens_q,
        mask,
        is_causal,
        kv_split_size,
        *workspace,
    )
