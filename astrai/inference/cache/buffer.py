"""Physical KV cache buffers.

Layer 1 — ``KVStorage``:   flat token-level K/V GPU buffers [n_layers, size, n_kv_heads, head_dim]
Layer 2 — ``ReqToTokenPool``: index table [req_idx, pos] → physical token slot
Layer 3 — ``BaseKVCache``:    shared fields for all cache modes
         ``PrefillKVCache``:  prefill-specific layout
         ``DecodeKVCache``:   decode-specific layout
         ``KVCache``:         union type for backward compatibility

These classes have no knowledge of tasks, allocation policies, or scheduling.
They are the "dumb" physical storage layer.
"""

import threading
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor


class ReqToTokenPool:
    """Maps [req_idx, pos] → physical token slot in KV storage.

    Each row is one request; each column is a sequence position.  The value
    at [req_idx, pos] is the flat index into the KV storage buffers.
    """

    def __init__(self, size: int, max_context_len: int, device: torch.device):
        self.size = size
        self.max_context_len = max_context_len
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
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
        self.req_to_token[indices] = values


class KVStorage:
    """Token-level KV cache storage.

    Buffers: ``[n_layers, size, n_kv_heads, head_dim]``.  Each token occupies
    one slot indexed by ``ReqToTokenPool``.
    """

    def __init__(
        self,
        size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.size = size
        self.k_buffer = torch.empty(
            (n_layers, size, n_kv_heads, head_dim), device=device, dtype=dtype
        )
        self.v_buffer = torch.empty(
            (n_layers, size, n_kv_heads, head_dim), device=device, dtype=dtype
        )


@dataclass
class BaseKVCache:
    """Shared fields for all KV cache modes.

    The attention layer does raw buffer indexing — no methods, no abstraction.
    """

    k_buffer: Tensor
    v_buffer: Tensor
    req_to_token: Tensor
    req_pool_indices: Tensor
    seq_lens: Tensor
    max_len: int
    kv_indptr: Tensor  # Always present in both modes


@dataclass
class PrefillKVCache(BaseKVCache):
    """Prefill-specific KV cache layout.

    Handles packed ragged batching where prompts have variable lengths.
    """

    out_cache_loc: Tensor  # [total_q_tokens] - flattened write locations
    qo_indptr: Tensor  # [B+1] - prefix sum of q_lens for unpacking
    q_tile_to_batch: Tensor  # [num_q_tiles] - maps Q tiles to batch indices
    q_tile_to_index: Tensor  # [num_q_tiles] - maps Q tiles to local indices


@dataclass
class DecodeKVCache(BaseKVCache):
    """Decode-specific KV cache layout.

    Single-token incremental generation with split-KV partial results.
    """

    out_cache_loc: Tensor  # [B] - one write position per request
    qo_indptr: Tensor  # [B+1] - sequential [0, 1, 2, ..., B]
    decode_o_part: Tensor  # [B, max_q_heads, MAX_SPLITS, head_dim]
    decode_ml_part: Tensor  # [B, max_q_heads, MAX_SPLITS, 2]
    decode_out: Tensor  # [B, max_q_heads, head_dim]


# Backward compatibility: union type for existing code
KVCache = Union[PrefillKVCache, DecodeKVCache]
