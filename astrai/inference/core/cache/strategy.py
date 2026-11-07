"""KV cache allocation layer.

Encapsulates the physical slot allocation policy, isolated from GPU buffers
and request lifecycle management.

- ``RequestCacheState``: data contract between strategy and manager (per-request slot state)
- ``Allocator``:       bitmask-based page allocator with LRU eviction
- ``RadixCache``:      page-granular prefix index (exact token match)
- ``AllocationStrategy``: ABC for physical slot allocation
- ``ContiguousStrategy``: statically partitioned, no dynamic allocation
- ``PagedStrategy``:    dynamic paged allocation from a shared pool
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple

import torch

from astrai.model.kv_cache import ReqToTokenPool

# ---- data contract: per-request slot state ----


@dataclass
class RequestCacheState:
    """Per-request cache allocation state.

    Co-locates all request-owned cache metadata so the alloc/free/extend
    lifecycle is atomic.  Owned by ``KVCacheManager``, consumed by
    every ``AllocationStrategy`` method.

    ``slots`` mirrors the page-size-1 tail of ``req_to_token[req_idx]`` on
    the host: ``extend`` appends to it instead of issuing one tiny H2D copy
    per decoded token, and ``write_indices`` flushes the prefix it did not
    already cover in one bulk write.
    """

    req_idx: int
    length: int = 0
    cached: int = 0
    pages: List[int] = field(default_factory=list)
    _slots: List[int] = field(default_factory=list)
    _flushed: int = 0


# ---- allocation primitives ----


class Allocator:
    """Bitmask-based page allocator with ref-counting and LRU eviction.

    The free set is a Python big-int bitmask (kept as the source of truth:
    ``_free_mask`` is read by tests and debug tooling). Scanning it
    directly costs a full limb-chain walk per extracted page on
    serving-scale pools, so the fast paths go through a shadow index of
    per-64-page words (``_words``) plus a cursor to the lowest non-empty
    word. Allocation only touches the handful of words it consumes;
    freeing sets one bit and pulls the cursor back. The two structures
    are updated together under the same lock, and ``_words`` can be
    rebuilt from the mask at any time (see ``_sync_words``).
    """

    _WORD_BITS = 64

    def __init__(self, n_pages: int):
        self._free_mask = (1 << n_pages) - 1
        self._n_pages = n_pages
        self._refs: List[int] = [0] * n_pages
        self._lru: OrderedDict[int, None] = OrderedDict()
        # Eviction may revoke an entire prefix subtree, not just its root.
        self.on_evict: Optional[Callable[[int], Optional[List[int]]]] = None
        self._lock = threading.Lock()
        self._sync_words()

    def _sync_words(self):
        """Rebuild the word index from ``_free_mask`` (caller holds or
        owns exclusive access, e.g. at init or after mask surgery)."""
        n_words = (self._n_pages + self._WORD_BITS - 1) // self._WORD_BITS
        self._words: List[int] = [0] * n_words
        m = self._free_mask
        w = 0
        while m:
            self._words[w] = m & 0xFFFFFFFFFFFFFFFF
            m >>= self._WORD_BITS
            w += 1
        self._first_nonempty = 0
        while self._first_nonempty < n_words and self._words[self._first_nonempty] == 0:
            self._first_nonempty += 1

    def _mask_alloc_bits(self, pages: List[int]):
        # One aggregated mask subtraction for the whole batch: per-page
        # big-int clears cost a limb walk each, which is what the word
        # index exists to avoid.
        clear = 0
        for idx in pages:
            clear |= 1 << idx
            w = idx >> 6
            self._words[w] &= ~(1 << (idx & 63))
        self._free_mask &= ~clear
        # Advance the cursor past every word the batch emptied, in order.
        for w in range(self._first_nonempty, len(self._words)):
            if self._words[w] == 0:
                self._first_nonempty = w + 1
            else:
                break

    def _mask_free_bit(self, idx: int):
        bit = 1 << idx
        self._free_mask |= bit
        w = idx >> 6
        self._words[w] |= 1 << (idx & 63)
        if w < self._first_nonempty:
            self._first_nonempty = w

    def _discard_cached(self, idxs: List[int]) -> int:
        """Release revoked cache identities; caller holds ``_lock``.

        Referenced pages remain owned by their requests. Only unreferenced
        LRU pages return to the free mask, without recursively evicting.
        """
        released = 0
        for idx in idxs:
            if idx not in self._lru:
                continue
            if self._refs[idx] != 0:
                raise RuntimeError("Cannot invalidate a referenced cache page")
            del self._lru[idx]
            self._mask_free_bit(idx)
            released += 1
        return released

    def discard_cached(self, idxs: List[int]) -> int:
        """Forget identities revoked by prefix replacement or relocation."""
        if not idxs:
            return 0
        with self._lock:
            return self._discard_cached(idxs)

    def _evict_cached(self, idx: int) -> int:
        """Evict one LRU victim and reclaim its revoked descendants."""
        revoked = self.on_evict(idx) if self.on_evict is not None else None
        released = self._discard_cached(revoked or [])
        # Also support callbacks that only notify and return None.
        return released + self._discard_cached([idx])

    def alloc(self) -> int:
        with self._lock:
            if self._free_mask:
                lsb = self._free_mask & -self._free_mask
                idx = lsb.bit_length() - 1
                self._mask_alloc_bits([idx])
                self._refs[idx] = 1
                return idx
            if self._lru:
                idx = next(iter(self._lru))
                self._evict_cached(idx)
                self._mask_alloc_bits([idx])
                self._refs[idx] = 1
                return idx
            return -1

    def alloc_many(self, n: int) -> Optional[List[int]]:
        """Allocate exactly ``n`` pages, or ``None`` if fewer are free.

        One lock acquisition per prompt. Free pages come from the low
        words of the mask in bulk; only when the free set cannot satisfy
        the request does it promote LRU pages (page-by-page, evicting as
        it goes).
        """
        if n <= 0:
            return []
        with self._lock:
            free = self._free_mask.bit_count()
            if free + len(self._lru) < n:
                return None
            while free < n:
                # One ancestor eviction can release several LRU pages.
                free += self._evict_cached(next(iter(self._lru)))
            # Word-index harvest: only the words actually consumed are
            # touched, so a fragmented pool costs the skipped empty words
            # (cursor advance) instead of a big-int limb walk per page.
            out: List[int] = []
            w = self._first_nonempty
            words = self._words
            while len(out) < n and w < len(words):
                word = words[w]
                if word == 0:
                    w += 1
                    continue
                base = w * self._WORD_BITS
                while word and len(out) < n:
                    lsb = word & -word
                    out.append(base + lsb.bit_length() - 1)
                    word ^= lsb
                words[w] = word
                w += 1
            self._mask_alloc_bits(out)
            for idx in out:
                self._refs[idx] = 1
            return out

    def free(self, idx: int, keep_cached: bool = False):
        with self._lock:
            self._refs[idx] -= 1
            if self._refs[idx] == 0:
                if keep_cached:
                    self._lru[idx] = None
                else:
                    self._mask_free_bit(idx)

    def free_many(self, idxs: List[int], keep_cached_for=None):
        """Release many pages under one lock acquisition.

        ``keep_cached_for`` is a predicate over the page index: pages it
        accepts stay in the LRU (prefix-cache hits), the rest return to the
        free mask.
        """
        if not idxs:
            return
        keep = keep_cached_for or (lambda idx: False)
        with self._lock:
            # Set-latch pass first (refs may drop to zero multiple times in
            # one batch; the freed bits are aggregated into ONE big-int
            # insertion and the word index updated per page — the index is
            # what the fast paths read, the mask follows in bulk).
            freed_bits = 0
            lowest_word = len(self._words)
            for idx in idxs:
                self._refs[idx] -= 1
                if self._refs[idx] == 0:
                    if keep(idx):
                        self._lru[idx] = None
                    else:
                        freed_bits |= 1 << idx
                        w = idx >> 6
                        self._words[w] |= 1 << (idx & 63)
                        if w < lowest_word:
                            lowest_word = w
            if freed_bits:
                self._free_mask |= freed_bits
                if lowest_word < self._first_nonempty:
                    self._first_nonempty = lowest_word

    def can_ever_satisfy(self, n_pages: int) -> bool:
        """Whether ``n_pages`` could ever be allocated, evicting freely.

        A request larger than the whole pool must be rejected at admission
        instead of looping: every retry would promote (and evict) the same
        pages forever — the busy-spin livelock this check exists to close.
        Must be called without the lock held (read-only probes).
        """
        with self._lock:
            return n_pages <= self._n_pages

    def reclaimable_pages(self) -> int:
        """Pages currently cached-but-unreferenced (LRU entries)."""
        with self._lock:
            return len(self._lru)

    def inc_ref(self, idx: int):
        with self._lock:
            self._refs[idx] += 1
            self._lru.pop(idx, None)

    def ref_count(self, idx: int) -> int:
        with self._lock:
            return self._refs[idx]

    def clear_cached(self) -> int:
        """Release every unreferenced LRU page back to the free pool."""
        with self._lock:
            count = len(self._lru)
            while self._lru:
                self._evict_cached(next(iter(self._lru)))
            return count


class RadixNode:
    """A page-aligned edge in the CPU-side prefix radix trie."""

    __slots__ = ("parent", "children", "page_idx", "tokens")

    def __init__(self, parent=None, tokens=(), page_idx=None):
        self.parent = parent
        self.children: Dict[Tuple, "RadixNode"] = {}
        self.page_idx = page_idx
        self.tokens = tuple(tokens)


class RadixCache:
    """Page-granular radix prefix index with exact token matching."""

    def __init__(self, page_size: int):
        self._page_size = page_size
        self._root = RadixNode()
        self._page_to_node: Dict[int, RadixNode] = {}
        self._lock = threading.Lock()

    def _remove_subtree(self, node: RadixNode) -> List[int]:
        """Remove every dependent identity; caller holds ``_lock``."""
        if node.parent is not None:
            node.parent.children.pop(node.tokens, None)
        revoked: List[int] = []
        pending = [node]
        while pending:
            child = pending.pop()
            pending.extend(child.children.values())
            if child.page_idx is not None:
                revoked.append(child.page_idx)
                self._page_to_node.pop(child.page_idx, None)
            child.page_idx = None
            child.parent = None
            child.children.clear()
        return revoked

    def evict(self, idx: int) -> List[int]:
        """Revoke a page and its subtree; return identities for the allocator."""
        with self._lock:
            node = self._page_to_node.get(idx)
            return self._remove_subtree(node) if node is not None else []

    def has_page(self, idx: int) -> bool:
        with self._lock:
            return idx in self._page_to_node

    def lookup(self, token_ids: List[int]) -> List[int]:
        with self._lock:
            full_pages = len(token_ids) // self._page_size
            hits: List[int] = []
            node = self._root
            for i in range(full_pages):
                start = i * self._page_size
                page_tokens = tuple(token_ids[start : start + self._page_size])
                child = node.children.get(page_tokens)
                if child is None or child.page_idx is None:
                    break
                hits.append(child.page_idx)
                node = child
            return hits

    def record(
        self, page_idx: int, token_ids: List[int], logical_page_idx: int
    ) -> List[int]:
        """Index a materialized page and return revoked physical identities.

        Re-recording the same page/prefix is a no-op. Replacing its physical
        page at the same prefix preserves descendants (their token context
        is unchanged); moving a physical page to another prefix revokes its
        old subtree. Only pages with already-indexed ancestors are reachable
        and may be recorded.
        """
        with self._lock:
            full_pages = len(token_ids) // self._page_size
            if logical_page_idx < 0 or logical_page_idx >= full_pages:
                return []

            node = self._root
            for i in range(logical_page_idx + 1):
                start = i * self._page_size
                page_tokens = tuple(token_ids[start : start + self._page_size])
                child = node.children.get(page_tokens)
                if child is None:
                    if i < logical_page_idx:
                        node = None
                        break
                    child = RadixNode(node, page_tokens)
                    node.children[page_tokens] = child
                node = child

            old = self._page_to_node.get(page_idx)
            if old is node and node is not None:
                return []
            revoked = self._remove_subtree(old) if old is not None else []
            # A missing ancestor, or relocating an ancestor into its own
            # subtree, cannot leave a reachable entry at the destination.
            if node is None or node.parent is None:
                return revoked
            if node.page_idx is not None:
                revoked.append(node.page_idx)
                self._page_to_node.pop(node.page_idx, None)
            node.page_idx = page_idx
            self._page_to_node[page_idx] = node
            if old is not None:
                revoked.remove(page_idx)
            return revoked


class AllocationStrategy(ABC):
    """Physical slot allocation policy.

    Subclasses implement the actual allocation semantics.  This ABC declares
    the contract; there are no default implementations.
    """

    @abstractmethod
    def alloc(self, state: RequestCacheState, prompt_ids: List[int]) -> bool: ...

    @abstractmethod
    def free(self, state: RequestCacheState) -> None: ...

    @abstractmethod
    def extend(self, state: RequestCacheState, pos: int) -> bool: ...

    def extend_batch(
        self, states: List[RequestCacheState], positions: List[int]
    ) -> List[bool]:
        """Extend many requests by one position each; per-item success flags.

        Default: loop ``extend``.  Strategies with a per-extension
        allocation cost override this with a batched harvest.
        """
        return [self.extend(s, p) for s, p in zip(states, positions)]

    @abstractmethod
    def write_indices(
        self, state: RequestCacheState, prompt_ids: List[int]
    ) -> None: ...

    @abstractmethod
    def record_hashes(
        self,
        state: RequestCacheState,
        prompt_ids: List[int],
        start: int,
        *,
        materialized_end: int,
    ) -> None: ...

    def flush_slots(self, states: List[RequestCacheState], device) -> None:
        """Push host-staged slot maps to the device row.

        Only the paged strategy stages slots on the host; the default is a
        no-op for strategies whose req_to_token rows are already current
        (contiguous rows are pre-filled and never rewritten).
        """

    def invalidate_cache(self) -> int:
        """Drop reusable KV entries after an inference weight update."""
        return 0

    def can_ever_fit(self, n_tokens: int) -> bool:
        """Whether a request of ``n_tokens`` could ever be admitted."""
        return True


class ContiguousStrategy(AllocationStrategy):
    """Static contiguous allocation: slots are pre-assigned at pool init.

    No dynamic allocation or prefix caching.  All operations are no-ops
    because ``ReqToTokenPool`` is pre-filled with contiguous ranges.
    """

    def alloc(self, state: RequestCacheState, prompt_ids: List[int]) -> bool:
        return True

    def free(self, state: RequestCacheState) -> None:
        pass

    def extend(self, state: RequestCacheState, pos: int) -> bool:
        return True

    def write_indices(self, state: RequestCacheState, prompt_ids: List[int]) -> None:
        pass

    def record_hashes(
        self,
        state: RequestCacheState,
        prompt_ids: List[int],
        start: int,
        *,
        materialized_end: int,
    ) -> None:
        pass


class PagedStrategy(AllocationStrategy):
    """Dynamic paged allocation from a shared bitmask pool.

    ``page_size`` is a parameter, not a separate strategy: at ``page_size=1``
    each allocated page *is* one token slot (``page * 1 + 0``), and prefix
    caching is simply disabled (``prefix=None``).  The unified page formula
    ``pages[page_idx] * page_size + offset`` holds for both.
    """

    def __init__(
        self,
        alloc: Allocator,
        prefix: Optional[RadixCache],
        page_size: int,
        req_pool: ReqToTokenPool,
        device,
    ):
        self._alloc = alloc
        self._prefix = prefix
        self._page_size = page_size
        self._req_pool = req_pool
        self._device = device
        # Pinned staging for the per-step flush of host-staged slot tails:
        # synchronous ``torch.tensor(..., device=)`` construction costs a
        # blocking H2D per tensor per decode step; staging into pre-pinned
        # buffers keeps those transfers asynchronous.  The ring depth of 2
        # lets step t+1 write its staging while step t's transfer may still
        # be in flight (same stream, so ordering is preserved either way).
        # Pinned allocation needs a CUDA driver — CPU-only environments
        # (CI) stage through pageable memory instead (the flush then just
        # keeps its old blocking-copy behaviour).
        max_batch = req_pool.req_to_token.shape[0]
        pin = torch.cuda.is_available() and torch.device(device).type == "cuda"
        self._flush_pin = [
            (
                torch.empty(max_batch, dtype=torch.int64, pin_memory=pin),
                torch.empty(max_batch, dtype=torch.int64, pin_memory=pin),
                torch.empty(max_batch, dtype=torch.int32, pin_memory=pin),
            )
            for _ in range(2)
        ]
        self._flush_ring = 0

    def alloc(self, state: RequestCacheState, prompt_ids: List[int]) -> bool:
        if self._prefix is not None:
            hits = self._prefix.lookup(prompt_ids)
            state.cached = len(hits) * self._page_size
            for p in hits:
                self._alloc.inc_ref(p)
            state.pages = list(hits)

        remaining = len(prompt_ids) - state.cached
        if remaining <= 0:
            return True
        n_new = (remaining + self._page_size - 1) // self._page_size
        new_pages = self._alloc.alloc_many(n_new)
        if new_pages is None:
            return False
        state.pages.extend(new_pages)
        return True

    def free(self, state: RequestCacheState) -> None:
        if self._prefix is not None:
            self._alloc.free_many(state.pages, keep_cached_for=self._prefix.has_page)
        else:
            self._alloc.free_many(state.pages)

    def extend(self, state: RequestCacheState, pos: int) -> bool:
        page_idx = pos // self._page_size
        if page_idx >= len(state.pages):
            p = self._alloc.alloc()
            if p < 0:
                return False
            state.pages.append(p)
        slot = state.pages[page_idx] * self._page_size + pos % self._page_size
        if self._page_size == 1 and pos == len(state._slots):
            # page_size=1 fast path: positions arrive in order, so the slot
            # list extends by one element instead of each call issuing a
            # single-element H2D copy to req_to_token (128 such copies per
            # decode step at serving batch sizes). _slots is flushed to the
            # device in bulk by write_indices/bind.
            state._slots.append(slot)
            return True
        self._req_pool.req_to_token[state.req_idx, pos] = slot
        return True

    def extend_batch(
        self, states: List[RequestCacheState], positions: List[int]
    ) -> List[bool]:
        """Extend a decode step's requests with ONE word-indexed harvest.

        ``extend`` allocates at most one page per call through the
        big-int-masked ``Allocator.alloc`` (~13us on serving-scale pools;
        128 per decode step = ~1.7ms of host time).  In the steady decode
        loop every state needs exactly one new page, so the batch asks
        the allocator once and hands the pages out lowest-first — the
        same order the per-request lsb scans would have produced, making
        the harvest unobservable in slot numbering.  When the batch
        harvest cannot satisfy every state (a mixed batch or pool
        pressure), the call falls back to per-request ``extend`` so the
        success/failure ORDER matches the historical per-request path.
        """
        page_size = self._page_size
        need: List[int] = []
        for i, (state, pos) in enumerate(zip(states, positions)):
            if pos // page_size >= len(state.pages):
                need.append(i)
        if need:
            pages = self._alloc.alloc_many(len(need))
            if pages is None or len(pages) < len(need):
                return [self.extend(s, p) for s, p in zip(states, positions)]
            for i, p in zip(need, pages):
                states[i].pages.append(p)
        # Existing pages still need their new decode position mapped. Page
        # reservation alone does not initialize the whole req_to_token row.
        results = [True] * len(states)
        for state, pos in zip(states, positions):
            page_idx = pos // page_size
            slot = state.pages[page_idx] * page_size + pos % page_size
            if page_size == 1 and pos == len(state._slots):
                state._slots.append(slot)
            else:
                self._req_pool.req_to_token[state.req_idx, pos] = slot
        return results

    def write_indices(self, state: RequestCacheState, prompt_ids: List[int]) -> None:
        total = min(len(prompt_ids), len(state.pages) * self._page_size)
        if total <= 0:
            return
        # One H2D write per row: every per-position assignment is a separate
        # tiny copy (~15us of driver overhead each), so a 512-token prompt
        # spends ~8ms in the loop while the same row written once as a tensor
        # costs ~60us. The position -> slot map is an affine function of the
        # page ids, built on the host in bulk either way.
        page_size = self._page_size
        pages = state.pages
        if page_size == 1:
            vals = pages[:total]
        else:
            vals = []
            for page_idx, p in enumerate(pages[: (total + page_size - 1) // page_size]):
                base = p * page_size
                n = min(page_size, total - page_idx * page_size)
                vals.extend(range(base, base + n))
        self._req_pool.req_to_token[state.req_idx, :total] = torch.tensor(
            vals, dtype=torch.int32, device=self._device
        )
        state._slots = list(vals)
        state._flushed = total

    def flush_slots(self, states: List[RequestCacheState], device) -> None:
        # req_to_token rows are gathered on-device by every decode bind
        # (out_cache_loc), so staged tails must land there before the
        # gather runs — steady incremental steps included. The staged tails
        # of the whole batch are concatenated into ONE tensor write (one
        # H2D per step instead of one per request; 128 per-request copies cost
        # ~5ms/step at serving batch sizes) using a built-once scatter
        # index (request row, in-row offset) pairs.
        flat_vals: List[int] = []
        row_idx: List[int] = []
        col_idx: List[int] = []
        for state in states:
            slots = state._slots
            n = len(slots)
            if n <= state._flushed:
                continue
            start = state._flushed
            state._flushed = n
            flat_vals.extend(slots[start:])
            req = state.req_idx
            row_idx.extend([req] * (n - start))
            col_idx.extend(range(start, n))
        if not flat_vals:
            return
        # Depth-2 pinned staging: writes go into the free ring slot, then
        # a single non_blocking H2D per tensor replaces three blocking
        # ``torch.tensor(..., device=)`` constructions.  Depth 2 (vs a
        # single buffer) keeps the host writable while the previous
        # transfer may still be queued — the caller (bind) runs strictly
        # step-serial, but the same stream executes the copies in order
        # regardless.
        pin_rows, pin_cols, pin_vals = self._flush_pin[self._flush_ring]
        self._flush_ring ^= 1
        n = len(flat_vals)
        pin_rows[:n] = torch.as_tensor(row_idx, dtype=torch.int64)
        pin_cols[:n] = torch.as_tensor(col_idx, dtype=torch.int64)
        pin_vals[:n] = torch.as_tensor(flat_vals, dtype=torch.int32)
        rows = pin_rows[:n].to(self._device, non_blocking=True)
        cols = pin_cols[:n].to(self._device, non_blocking=True)
        vals = pin_vals[:n].to(self._device, non_blocking=True)
        self._req_pool.req_to_token[rows, cols] = vals

    def record_hashes(
        self,
        state: RequestCacheState,
        prompt_ids: List[int],
        start: int,
        *,
        materialized_end: int,
    ) -> None:
        """Publish complete pages below the exclusive, materialized KV end.

        Reserved pages and the full prompt length are not evidence that a
        chunk has run. The caller must explicitly report its computed end.
        """
        if materialized_end < 0 or start < 0:
            raise ValueError("materialized_end and start must be nonnegative")
        if self._prefix is None:
            return
        full = min(materialized_end, len(prompt_ids)) // self._page_size
        for i in range(start, min(full, len(state.pages))):
            revoked = self._prefix.record(state.pages[i], prompt_ids, i)
            # Radix mutations finish before taking the allocator lock:
            # allocator-driven eviction takes these locks in the other order.
            self._alloc.discard_cached(revoked)

    def invalidate_cache(self) -> int:
        if self._prefix is None:
            return 0
        return self._alloc.clear_cached()

    def can_ever_fit(self, n_tokens: int) -> bool:
        # Page-align the request the same way ``alloc`` does; prefix hits
        # can only shrink the need, so the unaligned upper bound is a
        # safe admission test (a borderline request simply waits for a
        # hit it might not get, never spins forever).
        n_pages = (n_tokens + self._page_size - 1) // self._page_size
        return self._alloc.can_ever_satisfy(n_pages)
