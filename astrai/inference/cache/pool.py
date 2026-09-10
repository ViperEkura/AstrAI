"""KV cache orchestration: PagePool + TaskCacheManager.

PagePool owns the physical buffers (``KVStorage`` + ``ReqToTokenPool``)
and wires them to an allocation strategy.  It assembles the ``KVCache``
dataclass passed to the model forward.

TaskCacheManager owns the ``task_id`` → ``TaskCacheState`` mapping and
delegates physical slot allocation to the strategy, and KV bind to the pool.

See ``cache_buffer.py`` for the raw buffer primitives and ``cache_strategy.py``
for the allocation policies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from astrai.inference.cache.buffer import (
    DecodeKVCache,
    KVCache,
    KVStorage,
    PrefillKVCache,
    ReqToTokenPool,
)
from astrai.inference.cache.strategy import (
    AllocationStrategy,
    Allocator,
    ContiguousStrategy,
    PagedStrategy,
    RadixCache,
    TaskCacheState,
)
from astrai.inference.workspace import Q_TILE_ROWS, InferenceWorkspace

# Re-export everything so existing ``from astrai.inference.cache import ...``
# continues to work unchanged after the file split.
__all__ = [
    "KVCache",
    "PrefillKVCache",
    "DecodeKVCache",
    "KVStorage",
    "ReqToTokenPool",
    "Allocator",
    "RadixCache",
    "AllocationStrategy",
    "ContiguousStrategy",
    "PagedStrategy",
    "PagePool",
    "TaskCacheManager",
    "TaskCacheState",
    "page_hash",
]

# ---- helpers ----


def page_hash(
    token_ids: List[int], page_idx: int, page_size: int, parent_hash: int = 0
) -> int:
    start = page_idx * page_size
    end = min(start + page_size, len(token_ids))
    h = parent_hash
    for i in range(start, end):
        h = (h * 31 + token_ids[i]) & 0xFFFFFFFFFFFFFFFF
    return h


def _is_steady_increment(
    prev_sig: Optional[tuple],
    prev_vals: Optional[List[int]],
    cur_sig: tuple,
    cur_vals: List[int],
) -> bool:
    return (
        prev_sig is not None
        and prev_vals is not None
        and prev_sig == cur_sig
        and len(prev_vals) == len(cur_vals)
        and all(c == p + 1 for c, p in zip(cur_vals, prev_vals))
    )


# ---- task-scoped bind state ----
@dataclass
class _BindState:
    """Cached bind metadata for steady-state decode increment detection."""

    sig: tuple
    seq_lens: List[int]


# ---- pool + manager ----


class PagePool:
    """Physical KV cache: buffers + req-to-token table + allocation strategy + bind.

    Does not know about tasks — task lifecycle is managed by
    :class:`TaskCacheManager`, which holds a reference to this pool.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        page_size: int = 1,
        n_tokens: Optional[int] = None,
    ):
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.contiguous = n_tokens is None
        self.n_tokens = max_batch_size * max_seq_len if self.contiguous else n_tokens
        if self.n_tokens > torch.iinfo(torch.int32).max:
            raise ValueError("KV cache token count exceeds the int32 slot index limit")

        self._storage = KVStorage(
            self.n_tokens, n_layers, n_kv_heads, head_dim, device, dtype
        )
        self._req_pool = ReqToTokenPool(max_batch_size, max_seq_len, device)

        if self.contiguous:
            for i in range(max_batch_size):
                self._req_pool.req_to_token[i] = torch.arange(
                    i * max_seq_len,
                    (i + 1) * max_seq_len,
                    dtype=torch.int32,
                    device=device,
                )
            self._strategy: AllocationStrategy = ContiguousStrategy()
        else:
            n_pages = self.n_tokens // page_size
            alloc = Allocator(n_pages)
            prefix = RadixCache(page_size) if page_size > 1 else None
            if prefix is not None:
                alloc.on_evict = prefix.evict
            self._strategy = PagedStrategy(
                alloc, prefix, page_size, self._req_pool, device
            )

    @property
    def strategy(self) -> AllocationStrategy:
        return self._strategy

    @property
    def req_pool(self) -> ReqToTokenPool:
        return self._req_pool

    def bind_tasks(
        self,
        req_indices: List[int],
        seq_lens: List[int],
        workspace: InferenceWorkspace,
        device: Optional[torch.device] = None,
        start_pos: Optional[int] = None,
        incremental: bool = False,
    ) -> KVCache:
        """Assemble the ``KVCache`` metadata for a batch of tasks.

        Args:
            req_indices: request slot indices (from ``ReqToTokenPool``).
            seq_lens:    current sequence length per task.
            workspace:   pre-allocated fixed-shape buffers (CUDA-graph safe).
            start_pos:   if set, produce **prefill** cache (full q_len range).
                         If ``None``, produce **decode** cache (last position).
            incremental: if ``True``, reuse workspace state from previous step
                         by incrementing counters in-place (decode hot path).

        Returns:
            ``KVCache`` dataclass with the correct output shapes for the
            attention backend (prefill: ``[B, q_len]``, decode: ``[B, 1]``).
        """
        if device is None:
            device = workspace.device
        b = len(req_indices)

        rpi_buf = workspace.req_pool_indices
        sl_buf = workspace.seq_lens
        kvp_buf = workspace.kv_indptr
        inc_buf = workspace.inc
        ocl_buf = workspace.out_cache_loc

        if incremental:
            sl_buf[:b] += 1
            kvp_buf[: b + 1] += inc_buf[: b + 1]
        else:
            rpi_buf[:b].copy_(
                torch.tensor(req_indices, dtype=torch.int32, device=device)
            )
            sl_buf[:b].copy_(torch.tensor(seq_lens, dtype=torch.long, device=device))
            kvp_buf[: b + 1].zero_()
            kvp_buf[1 : b + 1] = sl_buf[:b].cumsum(0).to(torch.int32)

        req_pool_indices = rpi_buf[:b]
        seq_lens_t = sl_buf[:b]
        kv_indptr = kvp_buf[: b + 1]

        if start_pos is not None:
            # Packed prefill concatenates each request's query tokens.
            q_lens = [seq_len - start_pos for seq_len in seq_lens]
            if any(q_len <= 0 for q_len in q_lens):
                raise ValueError("prefill sequence lengths must exceed start_pos")
            out_cache_loc = torch.cat(
                [
                    self._req_pool.req_to_token[
                        req_pool_indices[i], start_pos : seq_lens[i]
                    ]
                    for i in range(b)
                ]
            )
            workspace.qo_indptr[: b + 1].zero_()
            workspace.qo_indptr[1 : b + 1].copy_(
                torch.tensor(q_lens, dtype=torch.int32, device=device).cumsum(0)
            )
            qo_indptr = workspace.qo_indptr[: b + 1]
            tile_batches = []
            tile_indices = []
            for batch, q_len in enumerate(q_lens):
                n_tiles = (q_len + Q_TILE_ROWS - 1) // Q_TILE_ROWS
                tile_batches.extend([batch] * n_tiles)
                tile_indices.extend(range(n_tiles))
            n_tiles = len(tile_batches)
            workspace.q_tile_to_batch[:n_tiles].copy_(
                torch.tensor(tile_batches, dtype=torch.int32, device=device)
            )
            workspace.q_tile_to_index[:n_tiles].copy_(
                torch.tensor(tile_indices, dtype=torch.int32, device=device)
            )
            q_tile_to_batch = workspace.q_tile_to_batch[:n_tiles]
            q_tile_to_index = workspace.q_tile_to_index[:n_tiles]

            return PrefillKVCache(
                k_buffer=self._storage.k_buffer,
                v_buffer=self._storage.v_buffer,
                req_to_token=self._req_pool.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_t,
                max_len=max(seq_lens),
                kv_indptr=kv_indptr,
                out_cache_loc=out_cache_loc,
                qo_indptr=qo_indptr,
                q_tile_to_batch=q_tile_to_batch,
                q_tile_to_index=q_tile_to_index,
            )
        else:
            # ---- decode: out_cache_loc is a single column (last position) ----
            write_pos = seq_lens_t - 1
            loc = self._req_pool.req_to_token[req_pool_indices, write_pos].unsqueeze(-1)
            ocl_buf[:b].copy_(loc)
            out_cache_loc = ocl_buf[:b].reshape(-1)
            workspace.qo_indptr[: b + 1].copy_(inc_buf[: b + 1])
            qo_indptr = workspace.qo_indptr[: b + 1]
            decode_o_part = getattr(workspace, "decode_o_part", None)
            decode_ml_part = getattr(workspace, "decode_ml_part", None)
            decode_out = getattr(workspace, "decode_out", None)

            return DecodeKVCache(
                k_buffer=self._storage.k_buffer,
                v_buffer=self._storage.v_buffer,
                req_to_token=self._req_pool.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_t,
                max_len=max(seq_lens),
                kv_indptr=kv_indptr,
                out_cache_loc=out_cache_loc,
                qo_indptr=qo_indptr,
                decode_o_part=decode_o_part,
                decode_ml_part=decode_ml_part,
                decode_out=decode_out,
            )


class TaskCacheManager:
    """Task ↔ KV slot lifecycle manager.

    Sole owner of ``task_id → TaskCacheState``.  Delegates physical slot
    allocation to the strategy (via ``pool.strategy``) and KV bind to
    ``pool.bind_tasks()``.

    Usage::

        pool = PagePool(...)
        mgr = TaskCacheManager(pool)
        mgr.task_alloc("req_1", [101, 202, 303])
        ...
        kv = mgr.bind(["req_1"], workspace)
    """

    def __init__(self, pool: PagePool):
        self._pool = pool
        self._strategy = pool.strategy
        self._req_pool = pool.req_pool
        self._max_seq_len = pool.max_seq_len
        self._states: Dict[str, TaskCacheState] = {}
        self._bind_state: Optional[_BindState] = None
        self._bind_was_steady = False

    # -- public task lifecycle --

    def task_alloc(self, task_id: str, prompt_ids: List[int]) -> bool:
        self._bind_state = None
        req_slots = self._req_pool.alloc(1)
        if req_slots is None:
            return False
        state = TaskCacheState(req_idx=req_slots[0])
        self._states[task_id] = state
        if not self._strategy.alloc(state, prompt_ids):
            self._rollback(state, task_id)
            return False
        self._strategy.write_indices(state, prompt_ids)
        state.length = len(prompt_ids)
        return True

    def task_free(self, task_id: str):
        self._bind_state = None
        state = self._states.pop(task_id, None)
        if state is None:
            return
        self._strategy.free(state)
        self._req_pool.free([state.req_idx])

    def task_extend(self, task_id: str, pos: int) -> bool:
        state = self._states.get(task_id)
        if state is None or pos >= self._max_seq_len:
            return False
        if not self._strategy.extend(state, pos):
            return False
        state.length = pos + 1
        return True

    @property
    def task_count(self) -> int:
        """Number of tasks currently holding KV request state."""
        return len(self._states)

    def task_cached(self, task_id: str) -> int:
        state = self._states.get(task_id)
        return state.cached if state is not None else 0

    def task_record_hashes(
        self, task_id: str, prompt_ids: List[int], start_logical_page: int = 0
    ):
        state = self._states.get(task_id)
        if state is not None:
            self._strategy.record_hashes(state, prompt_ids, start_logical_page)

    def invalidate_cache(self) -> int:
        """Drop reusable KV entries once all task-owned entries are released."""
        if self._states:
            raise RuntimeError("Cannot invalidate KV cache while tasks are active")
        self._bind_state = None
        self._bind_was_steady = False
        return self._strategy.invalidate_cache()

    @staticmethod
    def task_cacheable_ids(task_id: str, prompt_ids: List[int], output_ids: List[int]):
        return list(prompt_ids) + list(output_ids[:-1])

    # -- bind (assemble KVCache for the model forward) --

    def bind(
        self,
        task_ids: List[str],
        workspace: InferenceWorkspace,
        device: Optional[torch.device] = None,
        start_pos: Optional[int] = None,
    ) -> KVCache:
        """Build ``KVCache`` for an ordered list of task IDs."""
        states = [self._states[tid] for tid in task_ids]
        req_indices = [s.req_idx for s in states]
        seq_lens = [s.length for s in states]
        sig = tuple(req_indices)

        prev = self._bind_state
        incremental = (
            start_pos is None
            and prev is not None
            and _is_steady_increment(prev.sig, prev.seq_lens, sig, seq_lens)
        )
        self._bind_state = _BindState(sig, list(seq_lens))
        self._bind_was_steady = incremental

        return self._pool.bind_tasks(
            req_indices,
            seq_lens,
            workspace,
            device=device,
            start_pos=start_pos,
            incremental=incremental,
        )

    @property
    def bind_was_steady(self) -> bool:
        """True if the last bind was a steady-state increment (same tasks, +1 seq_lens)."""
        return self._bind_was_steady

    # -- internals --

    def _rollback(self, state: TaskCacheState, task_id: str):
        self._strategy.free(state)
        self._req_pool.free([state.req_idx])
        self._states.pop(task_id, None)
