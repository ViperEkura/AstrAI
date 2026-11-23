"""Physical KV storage and request allocation lifecycle."""

from typing import Dict, List, Optional, Sequence

import torch

from astrai.inference.core.cache.pages import (
    Allocator,
    RadixCache,
    RequestCacheState,
    RequestPageTable,
)
from astrai.inference.core.request import Request
from astrai.model.kv_cache import KVStorage


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

    def alloc(
        self, requests: Sequence[Request], ends: Optional[Sequence[int]] = None
    ) -> List[bool]:
        """Reserve up to each exclusive token end, in input order. Default to prompt length."""
        if ends is not None and len(ends) != len(requests):
            raise ValueError("one allocation end is required per request")
        results = []
        dirty = []
        for i, request in enumerate(requests):
            key = request.request_id
            end = len(request.prompt_ids) if ends is None else ends[i]
            state = self._states.get(key)
            if state is not None and state.owner != request.identity:
                results.append(False)
                continue
            if end < 0 or end > self._max_seq_len:
                results.append(False)
                continue
            if state is not None and end <= state.length:
                results.append(True)
                continue
            new = state is None
            if new:
                slots = self._req_pool.alloc(1)
                if slots is None:
                    results.append(False)
                    continue
                state = RequestCacheState(req_idx=slots[0], owner=request.identity)
                if self._prefix is not None:
                    tokens = request.prompt_ids
                    if end < len(tokens):
                        tokens = tokens[:end]
                    state.pages = self._prefix.lookup(tokens)
                    for page in state.pages:
                        self._alloc.inc_ref(page)
                    state.published = len(state.pages)
            if self._alloc is not None:
                needed = (end + self._page_size - 1) // self._page_size - len(
                    state.pages
                )
                if needed > 0:
                    pages = self._alloc.alloc_many(needed)
                    if pages is None:
                        if new:
                            self._alloc.free_many(
                                state.pages,
                                self._prefix.has_page if self._prefix else None,
                            )
                            self._req_pool.free([state.req_idx])
                        results.append(False)
                        continue
                    state.pages.extend(pages)
                if state._flushed < len(state.pages):
                    dirty.append(state)
            if new:
                self._states[key] = state
                request.cache_slot = state.req_idx
                request.cached_tokens = state.published * self._page_size
            state.length = end
            results.append(True)
        if dirty:
            self._req_pool.flush_pages(dirty)
        return results

    def commit(self, requests: Sequence[Request], ends: Sequence[int]) -> None:
        """Index newly completed full pages up to each exclusive execution end."""
        if len(requests) != len(ends) or any(end < 0 for end in ends):
            raise ValueError("one nonnegative commit end is required per request")
        if self._prefix is None:
            return
        for request, end in zip(requests, ends):
            state = self._states.get(request.request_id)
            if state is None or state.owner != request.identity:
                continue
            end = min(
                end, state.length, len(request.prompt_ids) + len(request.output_ids)
            )
            full = end // self._page_size
            if full <= state.published:
                continue
            tokens = request.prompt_ids
            if full * self._page_size > len(tokens):
                tokens = tokens + request.output_ids
            for i in range(state.published, full):
                revoked = self._prefix.record(state.pages[i], tokens, i)
                self._alloc.discard_cached(revoked)
            state.published = full

    def free(self, requests: Sequence[Request]) -> None:
        """Release retired request slots and their physical page references."""
        slots = []
        pages = []
        for request in requests:
            state = self._states.get(request.request_id)
            if state is None or state.owner != request.identity:
                continue
            del self._states[request.request_id]
            slots.append(state.req_idx)
            pages.extend(state.pages)
            request.cache_slot = -1
            request.cached_tokens = 0
        if pages:
            self._alloc.free_many(
                pages, self._prefix.has_page if self._prefix else None
            )
        if slots:
            self._req_pool.free(slots)

    @property
    def request_count(self) -> int:
        """Number of requests currently holding KV request state."""
        return len(self._states)

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
