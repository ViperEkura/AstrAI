"""Physical page allocation, request mappings and reusable prefix indexing."""

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple

import torch


class RequestPageTable:
    """Physical page IDs indexed by request slot and logical page."""

    def __init__(self, size: int, max_pages: int, device: torch.device):
        self.version = 0
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

    def flush_pages(self, states: List["RequestCacheState"]) -> None:
        dirty = [s for s in states if s._flushed < len(s.pages)]
        if not dirty:
            return
        if len(dirty) == 1:
            state = dirty[0]
            values = torch.tensor(
                state.pages[state._flushed :],
                dtype=torch.int32,
                device=self.page_table.device,
            )
            indices = (state.req_idx, slice(state._flushed, len(state.pages)))
        else:
            updates = [
                (s.req_idx, col, s.pages[col])
                for s in dirty
                for col in range(s._flushed, len(s.pages))
            ]
            rows, columns, values = torch.tensor(
                updates, device=self.page_table.device, dtype=torch.int32
            ).unbind(1)
            indices = (rows, columns)
        self.page_table[indices] = values
        self.version += 1
        for state in dirty:
            state._flushed = len(state.pages)


@dataclass
class RequestCacheState:
    """Per-request cache allocation state.

    Co-locates all request-owned cache metadata so the alloc/free/extend
    lifecycle is atomic.  Owned by ``KVCacheManager``.

    ``pages`` holds logical-to-physical page IDs; extensions flush only new IDs.
    """

    req_idx: int
    length: int = 0
    cached: int = 0
    pages: List[int] = field(default_factory=list)
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
