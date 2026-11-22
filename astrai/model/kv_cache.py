"""Device KV storage, logical-to-physical page mappings and per-step metadata."""

import threading
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor


class RequestPageTable:
    """Physical page IDs indexed by request slot and logical page."""

    def __init__(self, size: int, max_pages: int, device: torch.device):
        self.size = size
        self.max_pages = max_pages
        self.page_table = torch.zeros(
            (size, max_pages), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))
        self._lock = threading.Lock()

    def alloc(self, num_reqs: int) -> Optional[List[int]]:
        with self._lock:
            if num_reqs > len(self.free_slots):
                return None
            slots = self.free_slots[:num_reqs]
            self.free_slots = self.free_slots[num_reqs:]
            return slots

    def free(self, req_indices: List[int]):
        with self._lock:
            self.free_slots.extend(req_indices)

    def write(self, indices, values):
        self.page_table[indices] = values


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


@dataclass
class BaseKVCache:
    """Layer buffers and workspace-backed metadata shared by a model forward."""

    k_buffer: Tensor
    v_buffer: Tensor
    page_table: Tensor
    cache_seqlens: Tensor


@dataclass
class PrefillKVCache(BaseKVCache):
    cu_seqlens_q: Tensor


@dataclass
class DecodeKVCache(BaseKVCache):
    decode_o_part: Tensor
    decode_ml_part: Tensor
    decode_out: Tensor


KVCache = Union[PrefillKVCache, DecodeKVCache]
