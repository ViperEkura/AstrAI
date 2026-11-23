"""KV cache orchestration: BlockPool + KVCacheManager.

BlockPool owns the physical buffers (``KVStorage`` + ``RequestPageTable``)
and wires them to an allocation strategy.  It assembles the ``KVCache``
dataclass passed to the model forward.

KVCacheManager owns the ``request_id`` → ``RequestCacheState`` mapping and
delegates physical slot allocation to the strategy, and KV bind to the pool.

See ``cache_buffer.py`` for the raw buffer primitives and ``cache_strategy.py``
for the allocation policies.
"""

from dataclasses import dataclass
from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import torch

from astrai.inference.core.cache.strategy import (
    AllocationStrategy,
    Allocator,
    ContiguousStrategy,
    PagedStrategy,
    RadixCache,
    RequestCacheState,
)
from astrai.inference.worker.workspace import InferenceWorkspace
from astrai.model.kv_cache import (
    DecodeKVCache,
    KVCache,
    KVStorage,
    PrefillKVCache,
    RequestPageTable,
)

# Re-export everything so existing ``from astrai.inference.core.cache import ...``
# continues to work unchanged after the file split.
__all__ = [
    "AllocationStrategy",
    "Allocator",
    "BlockPool",
    "ContiguousStrategy",
    "DecodeKVCache",
    "KVCache",
    "KVCacheManager",
    "KVStorage",
    "PagedStrategy",
    "PrefillKVCache",
    "RadixCache",
    "RequestPageTable",
    "RequestCacheState",
]

# ---- helpers ----


def _is_steady_increment(
    prev_sig: Optional[Tuple],
    prev_vals: Optional[List[int]],
    cur_sig: Tuple,
    cur_vals: List[int],
) -> bool:
    return (
        prev_sig is not None
        and prev_vals is not None
        and prev_sig == cur_sig
        and len(prev_vals) == len(cur_vals)
        and all(c == p + 1 for c, p in zip(cur_vals, prev_vals))
    )


# ---- request-scoped bind state ----
@dataclass
class _BindState:
    """Cached bind metadata for steady-state decode increment detection."""

    sig: Tuple
    seq_lens: List[int]
    workspace: InferenceWorkspace


# ---- pool + manager ----


class BlockPool:
    """Physical KV cache: buffers + request page table + allocation strategy + bind.

    Does not know about requests — request lifecycle is managed by
    :class:`KVCacheManager`, which holds a reference to this pool.
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
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.contiguous = n_tokens is None
        pages_per_req = (max_seq_len + page_size - 1) // page_size
        self.n_tokens = (
            max_batch_size * pages_per_req * page_size if self.contiguous else n_tokens
        )
        if self.n_tokens > torch.iinfo(torch.int32).max:
            raise ValueError("KV cache token count exceeds the int32 slot index limit")

        self._storage = KVStorage(
            self.n_tokens, n_layers, page_size, n_kv_heads, head_dim, device, dtype
        )
        self._req_pool = RequestPageTable(max_batch_size, pages_per_req, device)

        if self.contiguous:
            for i in range(max_batch_size):
                self._req_pool.page_table[i] = torch.arange(
                    i * pages_per_req,
                    (i + 1) * pages_per_req,
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
    def req_pool(self) -> RequestPageTable:
        return self._req_pool

    def bind_tasks(
        self,
        req_indices: List[int],
        seq_lens: List[int],
        workspace: InferenceWorkspace,
        device: Optional[torch.device] = None,
        start_pos: Optional[int] = None,
        incremental: bool = False,
        refresh_pages: bool = True,
    ) -> KVCache:
        """Assemble the ``KVCache`` metadata for a batch of requests.

        Args:
            req_indices: request slot indices (from ``RequestPageTable``).
            seq_lens:    current sequence length per request.
            workspace:   pre-allocated fixed-shape buffers (CUDA-graph safe).
            start_pos:   if set, produce **prefill** cache (full q_len range).
                         If ``None``, produce **decode** cache (last position).
            incremental: if ``True``, reuse workspace state from previous step
                         by incrementing counters in-place (decode hot path).
            refresh_pages: whether the allocation strategy changed a page mapping.

        Returns:
            ``KVCache`` dataclass with the correct output shapes for the
            attention backend (prefill: ``[B, q_len]``, decode: ``[B, 1]``).
        """
        if device is None:
            device = workspace.device
        b = len(req_indices)

        rows = workspace.req_pool_indices[:b]
        past = workspace.cache_seqlens[:b]
        table = self._req_pool.page_table
        shape = (workspace.req_pool_indices.numel(), table.size(1))
        if workspace.page_table is None or workspace.page_table.shape != shape:
            with torch.inference_mode(False):
                workspace.page_table = torch.empty(
                    shape, dtype=torch.int32, device=device
                )
            incremental = False
        if not incremental:
            rows.copy_(torch.tensor(req_indices, dtype=torch.int32, device=device))
        pages = workspace.page_table[:b]
        if not incremental or refresh_pages:
            torch.index_select(table, 0, rows, out=pages)
        if start_pos is None:
            if incremental:
                past.add_(1)
            else:
                past.copy_(
                    torch.tensor(
                        [n - 1 for n in seq_lens], dtype=torch.int32, device=device
                    )
                )
            return DecodeKVCache(
                self._storage.k_buffer,
                self._storage.v_buffer,
                pages,
                past,
                workspace.decode_o_part,
                workspace.decode_ml_part,
                workspace.decode_out,
                layer_views=self._storage.layer_views,
            )

        lengths = [n - start_pos for n in seq_lens]
        if min(lengths) <= 0:
            raise ValueError("prefill sequence lengths must exceed start_pos")
        offsets = workspace.cu_seqlens_q[: b + 1]
        offsets.copy_(
            torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)
        )
        past.fill_(start_pos)
        return PrefillKVCache(
            self._storage.k_buffer,
            self._storage.v_buffer,
            pages,
            past,
            offsets,
            layer_views=self._storage.layer_views,
        )


class KVCacheManager:
    """Request ↔ KV slot lifecycle manager.

    Sole owner of ``request_id → RequestCacheState``.  Delegates physical slot
    allocation to the strategy (via ``pool.strategy``) and KV bind to
    ``pool.bind_tasks()``.

    Usage::

        pool = BlockPool(...)
        mgr = KVCacheManager(pool)
        mgr.alloc_slots("req_1", [101, 202, 303])
        ...
        kv = mgr.bind(["req_1"], workspace)
    """

    def __init__(self, pool: BlockPool):
        self._pool = pool
        self._strategy = pool.strategy
        self._req_pool = pool.req_pool
        self._max_seq_len = pool.max_seq_len
        self._states: Dict[str, RequestCacheState] = {}
        self._bind_state: Optional[_BindState] = None
        self._bind_was_steady = False

    # -- public request lifecycle --

    def alloc_slots(self, request_id: str, prompt_ids: List[int]) -> bool:
        self._bind_state = None
        req_slots = self._req_pool.alloc(1)
        if req_slots is None:
            return False
        state = RequestCacheState(req_idx=req_slots[0])
        self._states[request_id] = state
        if not self._strategy.alloc(state, prompt_ids):
            self._rollback(state, request_id)
            return False
        self._strategy.write_indices(state, prompt_ids)
        state.length = len(prompt_ids)
        return True

    def free_slots(self, request_id: str):
        self._bind_state = None
        state = self._states.pop(request_id, None)
        if state is None:
            return
        self._strategy.free(state)
        self._req_pool.free([state.req_idx])

    def extend_slots(self, request_id: str, pos: int) -> bool:
        state = self._states.get(request_id)
        if state is None or pos >= self._max_seq_len:
            return False
        if not self._strategy.extend(state, pos):
            return False
        state.length = pos + 1
        return True

    def extend_slots_batch(
        self, request_ids: List[str], positions: List[int]
    ) -> List[bool]:
        """``extend_slots`` for a whole decode step, one strategy call.

        The decode-steady hot path extends every active request by exactly
        one position; a single batched strategy call replaces one
        allocation round-trip per request (see
        ``PagedStrategy.extend_batch``).  Per-request semantics are
        unchanged: a missing state or an out-of-cap position fails that
        request alone.
        """
        results = [False] * len(request_ids)
        live_idx: List[int] = []
        live_states: List[RequestCacheState] = []
        live_pos: List[int] = []
        for i, tid in enumerate(request_ids):
            state = self._states.get(tid)
            if state is not None and positions[i] < self._max_seq_len:
                live_idx.append(i)
                live_states.append(state)
                live_pos.append(positions[i])
        if not live_idx:
            return results
        extended = self._strategy.extend_batch(live_states, live_pos)
        for i, state, pos, ok in zip(live_idx, live_states, live_pos, extended):
            if ok:
                state.length = pos + 1
                results[i] = True
        return results

    @property
    def request_count(self) -> int:
        """Number of requests currently holding KV request state."""
        return len(self._states)

    def cached_tokens(self, request_id: str) -> int:
        state = self._states.get(request_id)
        return state.cached if state is not None else 0

    def record_block_hashes(
        self,
        request_id: str,
        prompt_ids: List[int],
        start_logical_page: int = 0,
        *,
        materialized_end: int,
    ):
        """Publish full pages whose KV is materialized below an exclusive end.

        ``materialized_end`` is an absolute token position, not the number
        of tokens in this chunk. It is mandatory because allocation reserves
        the whole prompt before any KV is written. A completed full prefill
        explicitly passes ``len(prompt_ids)``; a chunk passes its actual end.
        """
        state = self._states.get(request_id)
        if state is not None:
            self._strategy.record_hashes(
                state,
                prompt_ids,
                start_logical_page,
                materialized_end=materialized_end,
            )

    def invalidate_cache(self) -> int:
        """Drop reusable KV entries once all request-owned entries are released."""
        if self._states:
            raise RuntimeError("Cannot invalidate KV cache while requests are active")
        self._bind_state = None
        self._bind_was_steady = False
        return self._strategy.invalidate_cache()

    def can_ever_fit(self, n_tokens: int) -> bool:
        """Whether a request of ``n_tokens`` could ever be admitted.

        The admission-time livelock guard: requests that can never fit are
        rejected outright by the scheduler instead of retrying forever.
        """
        return self._strategy.can_ever_fit(n_tokens)

    # -- bind (assemble KVCache for the model forward) --

    def bind(
        self,
        request_ids: List[str],
        workspace: InferenceWorkspace,
        device: Optional[torch.device] = None,
        start_pos: Optional[int] = None,
        seq_ends: Optional[List[int]] = None,
    ) -> KVCache:
        """Build ``KVCache`` for an ordered list of request IDs.

        ``seq_ends`` (chunked prefill) overrides the per-request sequence
        length with the chunk window's end: attention reads the prefix
        ``[0, start_pos)`` plus this window only. Writes begin at ``start_pos``.  Without it the
        states' full lengths are used, as before.
        """
        states = [self._states[tid] for tid in request_ids]
        req_indices = [s.req_idx for s in states]
        seq_lens = [s.length for s in states]
        if seq_ends is not None:
            if len(seq_ends) != len(states):
                raise ValueError("seq_ends must match request_ids in length")
            if (
                start_pos is None
                or start_pos < 0
                or any(
                    not start_pos < end <= state.length
                    for end, state in zip(seq_ends, states)
                )
            ):
                raise ValueError("chunk must fit the reserved request interval")
            seq_lens = list(seq_ends)
        sig = tuple(req_indices)

        prev = self._bind_state
        incremental = (
            start_pos is None
            and prev is not None
            and prev.workspace is workspace
            and _is_steady_increment(prev.sig, prev.seq_lens, sig, seq_lens)
        )
        # Publish newly allocated pages before gathering this batch’s rows.
        pages_changed = self._strategy.flush_pages(states, device or workspace.device)

        bound = self._pool.bind_tasks(
            req_indices,
            seq_lens,
            workspace,
            device=device,
            start_pos=start_pos,
            incremental=incremental,
            refresh_pages=pages_changed,
        )

        self._bind_state = _BindState(
            sig if start_pos is None else None, list(seq_lens), workspace
        )
        self._bind_was_steady = incremental
        return bound

    @property
    def bind_was_steady(self) -> bool:
        """True if the last bind was a steady-state increment (same requests, +1 seq_lens)."""
        return self._bind_was_steady

    # -- internals --

    def _rollback(self, state: RequestCacheState, request_id: str):
        self._strategy.free(state)
        self._req_pool.free([state.req_idx])
        self._states.pop(request_id, None)
