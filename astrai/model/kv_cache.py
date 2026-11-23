"""Device KV storage, logical-to-physical page mappings and per-step metadata."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


class KVStorage:
    """Per-layer K/V storage indexed by physical page and in-page position."""

    def __init__(
        self,
        size: int,
        n_layers: int,
        page_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.size = size
        self.k_buffer = torch.empty(
            (n_layers, size // page_size, page_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_buffer = torch.empty(
            (n_layers, size // page_size, page_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        # Storage never changes shape; create layer views once for every step.
        self.layer_views = tuple(
            (self.k_buffer[i].detach(), self.v_buffer[i].detach())
            for i in range(n_layers)
        )


@dataclass
class BaseKVCache:
    """Layer buffers and workspace-backed metadata shared by a model forward."""

    k_buffer: Tensor
    v_buffer: Tensor
    page_table: Tensor
    cache_seqlens: Tensor
    layer_views: Optional[Tuple[Tuple[Tensor, Tensor], ...]] = field(
        default=None, kw_only=True, repr=False
    )

    def layer(self, index: int) -> Tuple[Tensor, Tensor]:
        if self.layer_views is not None and not torch.is_grad_enabled():
            return self.layer_views[index]
        return self.k_buffer[index], self.v_buffer[index]


@dataclass
class PrefillKVCache(BaseKVCache):
    cu_seqlens_q: Tensor


@dataclass
class DecodeKVCache(BaseKVCache):
    decode_o_part: Tensor
    decode_ml_part: Tensor
    decode_out: Tensor


KVCache = Union[PrefillKVCache, DecodeKVCache]
