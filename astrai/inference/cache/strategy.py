"""KV cache allocation layer.

Encapsulates the physical slot allocation policy, isolated from GPU buffers
and task lifecycle management.

- ``TaskCacheState``: data contract between strategy and manager (per-task slot state)
- ``Allocator``:       bitmask-based page allocator with LRU eviction
- ``RadixCache``:      page-granular prefix index (exact token match)
- ``AllocationStrategy``: ABC for physical slot allocation
- ``ContiguousStrategy``: statically partitioned, no dynamic allocation
- ``PagedStrategy``:    dynamic paged allocation from a shared pool
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, OrderedDict

from astrai.model.kv_cache import ReqToTokenPool

# ---- data contract: per-task slot state ----


@dataclass
class TaskCacheState:
    """Per-task cache allocation state.

    Co-locates all task-owned cache metadata so the alloc/free/extend
    lifecycle is atomic.  Owned by ``TaskCacheManager``, consumed by
    every ``AllocationStrategy`` method.
    """

    req_idx: int
    length: int = 0
    cached: int = 0
    pages: List[int] = field(default_factory=list)


# ---- allocation primitives ----


class Allocator:
    """Bitmask-based page allocator with ref-counting and LRU eviction."""

    def __init__(self, n_pages: int):
        self._free_mask = (1 << n_pages) - 1
        self._refs: List[int] = [0] * n_pages
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.on_evict: Optional[Callable[[int], None]] = None
        self._lock = threading.Lock()

    def alloc(self) -> int:
        with self._lock:
            if self._free_mask:
                lsb = self._free_mask & -self._free_mask
                idx = lsb.bit_length() - 1
                self._free_mask ^= lsb
                self._refs[idx] = 1
                return idx
            if self._lru:
                idx, _ = self._lru.popitem(last=False)
                if self.on_evict:
                    self.on_evict(idx)
                self._refs[idx] = 1
                self._free_mask &= ~(1 << idx)
                return idx
            return -1

    def free(self, idx: int, keep_cached: bool = False):
        with self._lock:
            self._refs[idx] -= 1
            if self._refs[idx] == 0:
                if keep_cached:
                    self._lru[idx] = None
                else:
                    self._free_mask |= 1 << idx

    def inc_ref(self, idx: int):
        with self._lock:
            self._refs[idx] += 1
            self._lru.pop(idx, None)

    def ref_count(self, idx: int) -> int:
        with self._lock:
            return self._refs[idx]

    def touch(self, idx: int):
        with self._lock:
            if idx in self._lru:
                self._lru.move_to_end(idx)

    def clear_cached(self) -> int:
        """Release every unreferenced LRU page back to the free pool."""
        with self._lock:
            cached = list(self._lru)
            self._lru.clear()
            for idx in cached:
                if self._refs[idx] != 0:
                    raise RuntimeError("Cannot invalidate a referenced cache page")
                if self.on_evict:
                    self.on_evict(idx)
                self._free_mask |= 1 << idx
            return len(cached)


class RadixNode:
    """A page-aligned edge in the CPU-side prefix radix trie."""

    __slots__ = ("parent", "children", "page_idx", "tokens", "lock_ref")

    def __init__(self, parent=None, tokens=(), page_idx=None):
        self.parent = parent
        self.children: Dict[tuple, "RadixNode"] = {}
        self.page_idx = page_idx
        self.tokens = tuple(tokens)
        self.lock_ref = 0


class RadixCache:
    """Page-granular radix prefix index with exact token matching."""

    def __init__(self, page_size: int):
        self._page_size = page_size
        self._root = RadixNode()
        self._page_to_node: Dict[int, RadixNode] = {}
        self._lock = threading.Lock()

    def evict(self, idx: int):
        with self._lock:
            node = self._page_to_node.pop(idx, None)
            if node is None:
                return
            node.page_idx = None
            parent = node.parent
            if parent is not None:
                parent.children.pop(node.tokens, None)

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

    def record(self, page_idx: int, token_ids: List[int], logical_page_idx: int):
        with self._lock:
            full_pages = len(token_ids) // self._page_size
            if logical_page_idx >= full_pages:
                return
            old = self._page_to_node.pop(page_idx, None)
            if old is not None and old.parent is not None:
                old.parent.children.pop(old.tokens, None)

            node = self._root
            for i in range(logical_page_idx + 1):
                start = i * self._page_size
                page_tokens = tuple(token_ids[start : start + self._page_size])
                child = node.children.get(page_tokens)
                if child is None:
                    child = RadixNode(node, page_tokens)
                    node.children[page_tokens] = child
                node = child
            if node.page_idx is not None and node.page_idx != page_idx:
                replaced = node.page_idx
                self._page_to_node.pop(replaced, None)
            node.page_idx = page_idx
            self._page_to_node[page_idx] = node

    def release(self, pages: List[int]) -> None:
        with self._lock:
            for page_idx in pages:
                node = self._page_to_node.get(page_idx)
                if node is not None and node.lock_ref:
                    node.lock_ref -= 1


class AllocationStrategy(ABC):
    """Physical slot allocation policy.

    Subclasses implement the actual allocation semantics.  This ABC declares
    the contract; there are no default implementations.
    """

    @abstractmethod
    def alloc(self, state: TaskCacheState, prompt_ids: List[int]) -> bool: ...

    @abstractmethod
    def free(self, state: TaskCacheState) -> None: ...

    @abstractmethod
    def extend(self, state: TaskCacheState, pos: int) -> bool: ...

    @abstractmethod
    def write_indices(self, state: TaskCacheState, prompt_ids: List[int]) -> None: ...

    @abstractmethod
    def record_hashes(
        self,
        state: TaskCacheState,
        prompt_ids: List[int],
        start: int,
    ) -> None: ...

    def invalidate_cache(self) -> int:
        """Drop reusable KV entries after an inference weight update."""
        return 0


class ContiguousStrategy(AllocationStrategy):
    """Static contiguous allocation: slots are pre-assigned at pool init.

    No dynamic allocation or prefix caching.  All operations are no-ops
    because ``ReqToTokenPool`` is pre-filled with contiguous ranges.
    """

    def alloc(self, state: TaskCacheState, prompt_ids: List[int]) -> bool:
        return True

    def free(self, state: TaskCacheState) -> None:
        pass

    def extend(self, state: TaskCacheState, pos: int) -> bool:
        return True

    def write_indices(self, state: TaskCacheState, prompt_ids: List[int]) -> None:
        pass

    def record_hashes(
        self,
        state: TaskCacheState,
        prompt_ids: List[int],
        start: int,
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

    def alloc(self, state: TaskCacheState, prompt_ids: List[int]) -> bool:
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
        for _ in range(n_new):
            p = self._alloc.alloc()
            if p < 0:
                return False
            state.pages.append(p)
        return True

    def free(self, state: TaskCacheState) -> None:
        if self._prefix is not None:
            for p in state.pages:
                keep = self._prefix.has_page(p)
                self._alloc.free(p, keep_cached=keep)
                if not keep:
                    self._prefix.evict(p)
        else:
            for p in state.pages:
                self._alloc.free(p)

    def extend(self, state: TaskCacheState, pos: int) -> bool:
        page_idx = pos // self._page_size
        if page_idx >= len(state.pages):
            p = self._alloc.alloc()
            if p < 0:
                return False
            state.pages.append(p)
        offset = pos % self._page_size
        self._req_pool.req_to_token[state.req_idx, pos] = (
            state.pages[page_idx] * self._page_size + offset
        )
        return True

    def write_indices(self, state: TaskCacheState, prompt_ids: List[int]) -> None:
        total = len(prompt_ids)
        for pos in range(total):
            page_idx = pos // self._page_size
            offset = pos % self._page_size
            if page_idx < len(state.pages):
                self._req_pool.req_to_token[state.req_idx, pos] = (
                    state.pages[page_idx] * self._page_size + offset
                )

    def record_hashes(
        self,
        state: TaskCacheState,
        prompt_ids: List[int],
        start: int,
    ) -> None:
        if self._prefix is None:
            return
        full = len(prompt_ids) // self._page_size
        for i in range(start, min(full, len(state.pages))):
            self._prefix.record(state.pages[i], prompt_ids, i)

    def invalidate_cache(self) -> int:
        if self._prefix is None:
            return 0
        return self._alloc.clear_cached()
