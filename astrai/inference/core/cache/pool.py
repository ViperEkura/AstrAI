"""Physical KV storage and request allocation lifecycle."""

from typing import Dict, List, Optional

import torch

from astrai.inference.core.cache.strategy import (
    AllocationStrategy,
    Allocator,
    ContiguousStrategy,
    PagedStrategy,
    RadixCache,
    RequestCacheState,
    RequestPageTable,
)
from astrai.inference.worker.workspace import InferenceWorkspace
from astrai.model.kv_cache import KVCache, KVStorage


class BlockPool:
    """Physical KV cache: buffers + request page table + allocation strategy.

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
            self._req_pool.page_table.copy_(
                torch.arange(
                    max_batch_size * pages_per_req, dtype=torch.int32, device=device
                ).view(max_batch_size, pages_per_req)
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


class KVCacheManager:
    """Request ↔ KV slot lifecycle manager.

    Sole owner of ``request_id → RequestCacheState``.  Delegates physical slot
    allocation to the strategy (via ``pool.strategy``) and binds each batch to its execution workspace.

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

    # -- public request lifecycle --

    def alloc_slots(self, request_id: str, prompt_ids: List[int]) -> bool:
        req_slots = self._req_pool.alloc(1)
        if req_slots is None:
            return False
        state = RequestCacheState(req_idx=req_slots[0])
        self._states[request_id] = state
        if not self._strategy.alloc(state, prompt_ids):
            self._rollback(state, request_id)
            return False
        self._strategy.flush_pages([state])
        state.length = len(prompt_ids)
        return True

    def free_slots(self, request_id: str):
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
        *,
        start_pos: Optional[int] = None,
        seq_ends: Optional[List[int]] = None,
        positions: Optional[List[int]] = None,
    ) -> KVCache:
        """Build ``KVCache`` for an ordered list of request IDs.

        ``seq_ends`` (chunked prefill) overrides the per-request sequence
        length with the chunk window's end: attention reads the prefix
        ``[0, start_pos)`` plus this window only. Writes begin at ``start_pos``.  Without it the
        states' full lengths are used, as before.
        """
        states = [self._states[tid] for tid in request_ids]
        if start_pos is None:
            if seq_ends is not None:
                raise ValueError("chunk must fit the reserved request interval")
            if positions is None:
                positions = [state.length - 1 for state in states]
            elif len(positions) != len(states) or any(
                pos < 0 or pos >= state.length for pos, state in zip(positions, states)
            ):
                raise ValueError(
                    "decode position must fit the reserved request interval"
                )
        else:
            if positions is not None:
                raise ValueError("positions apply only to decode")
            if seq_ends is None:
                seq_ends = [state.length for state in states]
            if len(seq_ends) != len(states):
                raise ValueError("seq_ends must match request_ids in length")
            if start_pos < 0 or any(
                not start_pos < end <= s.length for end, s in zip(seq_ends, states)
            ):
                raise ValueError("chunk must fit the reserved request interval")
        self._strategy.flush_pages(states)
        return workspace.bind(
            self._pool._storage,
            self._req_pool.page_table,
            self._req_pool.version,
            tuple(s.req_idx for s in states),
            positions,
            start_pos,
            seq_ends,
        )

    # -- internals --

    def _rollback(self, state: RequestCacheState, request_id: str):
        self._strategy.free(state)
        self._req_pool.free([state.req_idx])
        self._states.pop(request_id, None)
