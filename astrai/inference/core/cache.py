import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor


def page_hash(token_ids: List[int], page_idx: int, page_size: int) -> int:
    start = page_idx * page_size
    end = min(start + page_size, len(token_ids))
    h = 0
    for i in range(start, end):
        h = (h * 31 + token_ids[i]) & 0xFFFFFFFFFFFFFFFF
    return h


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


class PrefixCache:
    """Hash-based prefix matching: maps page hashes to physical page indices."""

    def __init__(self, page_size: int):
        self._page_size = page_size
        self._page_to_hash: Dict[int, int] = {}
        self._hash_to_page: Dict[int, int] = {}
        self._lock = threading.Lock()

    def evict(self, idx: int):
        with self._lock:
            h = self._page_to_hash.pop(idx, None)
            if h is not None:
                self._hash_to_page.pop(h, None)

    def has_page(self, idx: int) -> bool:
        with self._lock:
            return idx in self._page_to_hash

    def lookup(self, token_ids: List[int]) -> List[int]:
        with self._lock:
            full_pages = len(token_ids) // self._page_size
            hits: List[int] = []
            for i in range(full_pages):
                h = page_hash(token_ids, i, self._page_size)
                p = self._hash_to_page.get(h)
                if p is None:
                    break
                hits.append(p)
            return hits

    def record(self, page_idx: int, token_ids: List[int], logical_page_idx: int):
        with self._lock:
            h = page_hash(token_ids, logical_page_idx, self._page_size)
            old_h = self._page_to_hash.pop(page_idx, None)
            if old_h is not None:
                self._hash_to_page.pop(old_h, None)
            self._page_to_hash[page_idx] = h
            self._hash_to_page[h] = page_idx


class PagePool:
    """Orchestrates allocator (page management) and PrefixCache (content addressing)."""

    def __init__(self, allocator: Allocator, prefix: PrefixCache):
        self._alloc = allocator
        self._prefix = prefix
        self._alloc.on_evict = prefix.evict

    @property
    def allocator(self) -> Allocator:
        return self._alloc

    @property
    def prefix(self) -> PrefixCache:
        return self._prefix

    def alloc(self) -> int:
        return self._alloc.alloc()

    def free(self, idx: int):
        keep = self._prefix.has_page(idx)
        self._alloc.free(idx, keep_cached=keep)
        if not keep:
            self._prefix.evict(idx)

    def inc_ref(self, idx: int):
        self._alloc.inc_ref(idx)

    def lookup(self, token_ids: List[int]) -> List[int]:
        hits = self._prefix.lookup(token_ids)
        for p in hits:
            self._alloc.touch(p)
        return hits

    def record(self, page_idx: int, token_ids: List[int], logical_page_idx: int):
        self._prefix.record(page_idx, token_ids, logical_page_idx)


class TaskTable:
    """Maps task_ids to page tables and cached token counts."""

    def __init__(self, page_size: int):
        self._page_size = page_size
        self._pages: Dict[str, List[int]] = {}
        self._cached: Dict[str, int] = {}
        self._lock = threading.Lock()

    def set(self, task_id: str, page_table: List[int], cached: int):
        with self._lock:
            self._pages[task_id] = page_table
            self._cached[task_id] = cached

    def get(self, task_id: str) -> List[int]:
        with self._lock:
            return self._pages.get(task_id, [])

    def get_cached(self, task_id: str) -> int:
        with self._lock:
            return self._cached.get(task_id, 0)

    def pop(self, task_id: str) -> Tuple[List[int], int]:
        with self._lock:
            pages = self._pages.pop(task_id, [])
            cached = self._cached.pop(task_id, 0)
            return pages, cached

    def get_ref(self, task_id: str) -> List[int]:
        with self._lock:
            return self._pages.setdefault(task_id, [])

    def table_tensor(self, task_ids: List[str], device: torch.device) -> Tensor:
        with self._lock:
            states = [self._pages.get(tid, []) for tid in task_ids]
            max_pages = max((len(s) for s in states), default=0)
            rows = [s + [-1] * (max_pages - len(s)) for s in states]
            return torch.tensor(rows, dtype=torch.long, device=device)


class Storage:
    """KV-cache tensor storage with paged write/gather."""

    def __init__(
        self,
        n_layers: int,
        n_pages: int,
        page_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.page_size = page_size
        self.k_cache = torch.empty(
            (n_layers, n_pages, page_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_cache = torch.empty(
            (n_layers, n_pages, page_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

    def write(
        self,
        layer_id: int,
        page_table: Tensor,
        start_pos: int,
        k: Tensor,
        v: Tensor,
    ):
        seq_len = k.size(1)
        if seq_len == 0:
            return
        page_size = self.page_size
        written = 0
        first_page = start_pos // page_size
        last_page = (start_pos + seq_len - 1) // page_size
        for pi in range(first_page, last_page + 1):
            phys_pages = page_table[:, pi]
            page_start = pi * page_size
            write_start = max(page_start, start_pos)
            write_end = min(page_start + page_size, start_pos + seq_len)
            offset = write_start - page_start
            chunk = write_end - write_start
            valid = phys_pages >= 0
            if not valid.all():
                if valid.any():
                    valid_pages = phys_pages[valid]
                    self.k_cache[layer_id, valid_pages, offset : offset + chunk] = k[
                        valid, written : written + chunk
                    ]
                    self.v_cache[layer_id, valid_pages, offset : offset + chunk] = v[
                        valid, written : written + chunk
                    ]
                written += chunk
                continue
            self.k_cache[layer_id, phys_pages, offset : offset + chunk] = k[
                :, written : written + chunk
            ]
            self.v_cache[layer_id, phys_pages, offset : offset + chunk] = v[
                :, written : written + chunk
            ]
            written += chunk

    def gather(
        self, layer_id: int, page_table: Tensor, total_len: int
    ) -> Tuple[Tensor, Tensor]:
        safe = page_table.clamp(min=0)
        k = self.k_cache[layer_id, safe]
        v = self.v_cache[layer_id, safe]
        k = k.flatten(1, 2)
        v = v.flatten(1, 2)
        if (page_table < 0).any():
            invalid = (
                (page_table < 0)
                .unsqueeze(-1)
                .expand(-1, -1, self.page_size)
                .flatten(1, 2)
            )
            invalid = invalid[:, :, None, None].expand_as(k)
            k = k.masked_fill(invalid, 0.0)
            v = v.masked_fill(invalid, 0.0)
        k = k[:, :total_len]
        v = v[:, :total_len]
        return k, v


class CacheView(ABC):
    """Abstract view passed to attention layers for KV-cache I/O."""

    @abstractmethod
    def write(self, layer_id: int, k: Tensor, v: Tensor): ...

    @abstractmethod
    def gather(self, layer_id: int) -> Tuple[Tensor, Tensor]: ...


class KVCache(ABC):
    """Abstract KV-cache facade for scheduler/executor."""

    @abstractmethod
    def task_alloc(self, task_id: str, prompt_ids: List[int]) -> bool: ...

    @abstractmethod
    def task_free(self, task_id: str): ...

    @abstractmethod
    def task_extend(self, task_id: str, pos: int) -> bool: ...

    @abstractmethod
    def bind_tasks(
        self,
        task_ids: List[str],
        total_len: int,
        device: torch.device,
        write_positions: Optional[Tensor] = None,
    ) -> CacheView: ...

    def task_cached(self, task_id: str) -> int:
        return 0

    def task_record_hashes(
        self, task_id: str, prompt_ids: List[int], start_logical_page: int = 0
    ): ...


class PageCacheView(CacheView):
    """Bundles Storage + page_table + total_len for attention layers."""

    def __init__(self, storage: Storage, page_table: Tensor, total_len: int = 0):
        self._storage = storage
        self._page_table = page_table
        self._total_len = total_len

    def write(self, layer_id: int, k: Tensor, v: Tensor):
        start_pos = self._total_len - k.size(1)
        self._storage.write(layer_id, self._page_table, start_pos, k, v)

    def gather(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        return self._storage.gather(layer_id, self._page_table, self._total_len)


class PageCache(KVCache):
    """Paged KV-cache with prefix sharing."""

    def __init__(
        self,
        n_layers: int,
        n_pages: int,
        page_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.page_size = page_size
        self._pool = PagePool(Allocator(n_pages), PrefixCache(page_size))
        self._table = TaskTable(page_size)
        self._storage = Storage(
            n_layers, n_pages, page_size, n_kv_heads, head_dim, device, dtype
        )

    def task_alloc(self, task_id: str, prompt_ids: List[int]) -> bool:
        hits = self._pool.lookup(prompt_ids)
        cached = len(hits) * self.page_size
        for p in hits:
            self._pool.inc_ref(p)

        remaining = len(prompt_ids) - cached
        n_new = (
            (remaining + self.page_size - 1) // self.page_size if remaining > 0 else 0
        )
        new_pages: List[int] = []
        if n_new > 0:
            for _ in range(n_new):
                p = self._pool.alloc()
                if p < 0:
                    for hp in hits:
                        self._pool.free(hp)
                    for np in new_pages:
                        self._pool.free(np)
                    return False
                new_pages.append(p)

        self._table.set(task_id, hits + new_pages, cached)
        return True

    def task_free(self, task_id: str):
        page_table, _ = self._table.pop(task_id)
        for idx in page_table:
            self._pool.free(idx)

    def task_extend(self, task_id: str, pos: int) -> bool:
        page_table = self._table.get(task_id)
        needed = (pos + 1 + self.page_size - 1) // self.page_size
        while len(page_table) < needed:
            p = self._pool.alloc()
            if p < 0:
                return False
            page_table.append(p)
        return True

    def task_cached(self, task_id: str) -> int:
        return self._table.get_cached(task_id)

    def task_record_hashes(
        self, task_id: str, prompt_ids: List[int], start_logical_page: int = 0
    ):
        page_table = self._table.get(task_id)
        full_pages = len(prompt_ids) // self.page_size
        for i in range(start_logical_page, full_pages):
            self._pool.record(page_table[i], prompt_ids, i)

    def bind_tasks(
        self,
        task_ids: List[str],
        total_len: int,
        device: torch.device,
        write_positions: Optional[Tensor] = None,
    ) -> PageCacheView:
        page_table = self._table.table_tensor(task_ids, device)
        return PageCacheView(self._storage, page_table, total_len)


class ContiguousCacheView(CacheView):
    """Contiguous KV-cache view for attention layers."""

    def __init__(
        self,
        cache: "ContiguousCache",
        batch_indices: Tensor,
        total_len: int = 0,
        write_positions: Optional[Tensor] = None,
    ):
        self._cache = cache
        self._batch_indices = batch_indices
        self._total_len = total_len
        self._write_positions = write_positions

    def write(self, layer_id: int, k: Tensor, v: Tensor):
        seq_len = k.size(1)
        indices = self._batch_indices
        if self._write_positions is not None and seq_len == 1:
            pos = self._write_positions
            self._cache.k[layer_id, indices, pos] = k.squeeze(1)
            self._cache.v[layer_id, indices, pos] = v.squeeze(1)
            for s, p in zip(indices.tolist(), pos.tolist()):
                cur = self._cache._slot_len.get(s, 0)
                if p + 1 > cur:
                    self._cache._slot_len[s] = p + 1
        else:
            start_pos = self._total_len - seq_len
            self._cache.k[layer_id, indices, start_pos : start_pos + seq_len] = k
            self._cache.v[layer_id, indices, start_pos : start_pos + seq_len] = v
            new_len = start_pos + seq_len
            for s in indices.tolist():
                cur = self._cache._slot_len.get(s, 0)
                if new_len > cur:
                    self._cache._slot_len[s] = new_len

    def gather(self, layer_id: int) -> Tuple[Tensor, Tensor]:
        max_len = max(
            self._cache._slot_len.get(int(s), 0) for s in self._batch_indices.tolist()
        )
        indices = self._batch_indices
        k = self._cache.k[layer_id, indices, :max_len]
        v = self._cache.v[layer_id, indices, :max_len]
        return k, v


class ContiguousCache(KVCache):
    """Contiguous per-slot KV cache (default implementation)."""

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.max_seq_len = max_seq_len
        self.k = torch.zeros(
            n_layers,
            max_batch_size,
            max_seq_len,
            n_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.v = torch.zeros(
            n_layers,
            max_batch_size,
            max_seq_len,
            n_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self._slot_len: Dict[int, int] = {}
        self._task_slot: Dict[str, int] = {}
        self._free_slots = list(range(max_batch_size))
        self._device = device

    def task_alloc(self, task_id: str, prompt_ids: List[int]) -> bool:
        if not self._free_slots:
            return False
        slot = self._free_slots.pop(0)
        self._task_slot[task_id] = slot
        self._slot_len[slot] = 0
        return True

    def task_free(self, task_id: str):
        slot = self._task_slot.pop(task_id, None)
        if slot is not None:
            self._slot_len.pop(slot, None)
            self._free_slots.append(slot)

    def task_extend(self, task_id: str, pos: int) -> bool:
        return pos < self.max_seq_len

    def task_cached(self, task_id: str) -> int:
        slot = self._task_slot.get(task_id)
        if slot is None:
            return 0
        return self._slot_len.get(slot, 0)

    def bind_tasks(
        self,
        task_ids: List[str],
        total_len: int,
        device: torch.device,
        write_positions: Optional[Tensor] = None,
    ) -> ContiguousCacheView:
        slots = [self._task_slot[tid] for tid in task_ids]
        batch_indices = torch.tensor(slots, dtype=torch.long, device=device)
        return ContiguousCacheView(
            self, batch_indices, total_len, write_positions=write_positions
        )
