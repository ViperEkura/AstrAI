"""Physical KV storage and request allocation lifecycle."""

from typing import Dict, List, Optional

import torch

from astrai.inference.core.cache.pages import (
    Allocator,
    RadixCache,
    RequestCacheState,
    RequestPageTable,
)
from astrai.inference.worker.workspace import InferenceWorkspace
from astrai.model.kv_cache import KVCache, KVStorage


class BlockPool:
    """KV buffers, request page table and optional dynamic page allocator.

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
        self.req_pool = RequestPageTable(max_batch_size, pages_per_req, device)

        self.allocator = None
        self.prefix = None
        if self.contiguous:
            self.req_pool.page_table.copy_(
                torch.arange(
                    max_batch_size * pages_per_req, dtype=torch.int32, device=device
                ).view(max_batch_size, pages_per_req)
            )
        else:
            self.allocator = Allocator(self.n_tokens // page_size)
            if page_size > 1:
                self.prefix = RadixCache(page_size)
                self.allocator.on_evict = self.prefix.evict


class KVCacheManager:
    """Own request slots, page reservations and prefix publication."""

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self._alloc = pool.allocator
        self._prefix = pool.prefix
        self._page_size = pool.page_size
        self._req_pool = pool.req_pool
        self._max_seq_len = pool.max_seq_len
        self._states: Dict[str, RequestCacheState] = {}

    # -- public request lifecycle --

    def admit(self, request_id: str, prompt_ids: List[int]) -> bool:
        """Reserve this prompt and acquire any reusable prefix pages."""
        if request_id in self._states:
            raise ValueError(f"duplicate cache request: {request_id}")
        if not self.can_ever_fit(len(prompt_ids)):
            return False
        req_slots = self._req_pool.alloc(1)
        if req_slots is None:
            return False
        state = RequestCacheState(req_idx=req_slots[0])
        if self._alloc is not None:
            if self._prefix is not None:
                state.pages = self._prefix.lookup(prompt_ids)
                state.cached = len(state.pages) * self._page_size
                for page in state.pages:
                    self._alloc.inc_ref(page)
            total = (len(prompt_ids) + self._page_size - 1) // self._page_size
            pages = self._alloc.alloc_many(total - len(state.pages))
            if pages is None:
                self._alloc.free_many(
                    state.pages, self._prefix.has_page if self._prefix else None
                )
                self._req_pool.free(req_slots)
                return False
            state.pages.extend(pages)
            self._req_pool.flush_pages([state])
        self._states[request_id] = state
        state.length = len(prompt_ids)
        return True

    def release(self, request_id: str):
        state = self._states.pop(request_id, None)
        if state is None:
            return
        if self._alloc is not None:
            self._alloc.free_many(
                state.pages, self._prefix.has_page if self._prefix else None
            )
        self._req_pool.free([state.req_idx])

    def reserve_decode(
        self, request_ids: List[str], positions: List[int]
    ) -> List[bool]:
        """Reserve through each token position. Results follow input order."""
        if len(request_ids) != len(positions) or len(set(request_ids)) != len(
            request_ids
        ):
            raise ValueError("decode positions must match distinct request IDs")
        ok = []
        pending = []
        for i, (key, pos) in enumerate(zip(request_ids, positions)):
            state = self._states.get(key)
            valid = state is not None and 0 <= pos < self._max_seq_len
            ok.append(valid)
            if valid:
                count = (
                    pos // self._page_size + 1 - len(state.pages)
                    if self._alloc is not None
                    else 0
                )
                if count > 0:
                    pending.append((i, state, count))
                elif pos >= state.length:
                    state.length = pos + 1
        if pending:
            pages = self._alloc.alloc_many(sum(count for _, _, count in pending))
            offset = 0
            for i, state, count in pending:
                if pages is None:
                    # Under pressure earlier requests get the available pages first.
                    new_pages = self._alloc.alloc_many(count)
                    if new_pages is None:
                        ok[i] = False
                        continue
                else:
                    new_pages = pages[offset : offset + count]
                    offset += count
                state.pages.extend(new_pages)
                state.length = positions[i] + 1
        return ok

    @property
    def request_count(self) -> int:
        """Number of requests currently holding KV request state."""
        return len(self._states)

    def cached_tokens(self, request_id: str) -> int:
        state = self._states.get(request_id)
        return state.cached if state is not None else 0

    def publish_prefix(
        self,
        request_id: str,
        prompt_ids: List[int],
        *,
        start: int = 0,
        materialized_end: int,
    ):
        """Publish full pages whose KV is materialized below an exclusive end.

        ``materialized_end`` is an absolute token position, not the number
        of tokens in this chunk. It is mandatory because allocation reserves
        the whole prompt before any KV is written. A completed full prefill
        explicitly passes ``len(prompt_ids)``; a chunk passes its actual end.
        """
        state = self._states.get(request_id)
        if state is None or self._alloc is None:
            return
        if materialized_end < 0 or start < 0:
            raise ValueError("materialized_end and start must be nonnegative")
        if self._prefix is not None:
            full = min(materialized_end, len(prompt_ids)) // self._page_size
            for i in range(start // self._page_size, min(full, len(state.pages))):
                revoked = self._prefix.record(state.pages[i], prompt_ids, i)
                self._alloc.discard_cached(revoked)

    def clear_prefix_cache(self) -> int:
        """Drop reusable KV entries once all request-owned entries are released."""
        if self._states:
            raise RuntimeError("Cannot invalidate KV cache while requests are active")
        return self._alloc.clear_cached() if self._prefix is not None else 0

    def can_ever_fit(self, n_tokens: int) -> bool:
        """Whether a request of ``n_tokens`` could ever be admitted.

        The admission-time livelock guard: requests that can never fit are
        rejected outright by the scheduler instead of retrying forever.
        """
        return n_tokens <= self._max_seq_len and (
            self._alloc is None
            or self._alloc.can_ever_satisfy(
                (n_tokens + self._page_size - 1) // self._page_size
            )
        )

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
        if self._alloc is not None:
            self._req_pool.flush_pages(states)
        return workspace.bind(
            self.pool._storage,
            self._req_pool.page_table,
            self._req_pool.version,
            tuple(s.req_idx for s in states),
            positions,
            start_pos,
            seq_ends,
        )
