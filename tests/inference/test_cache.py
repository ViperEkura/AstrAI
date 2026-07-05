"""Unit tests for inference cache components."""

import torch

from astrai.inference import (
    Allocator,
    PageCache,
    PagePool,
    PrefixCache,
    Storage,
    TaskTable,
    page_hash,
)


def make_pool(n_pages: int, page_size: int) -> PagePool:
    return PagePool(Allocator(n_pages), PrefixCache(page_size))


def test_page_hash_full_page():
    token_ids = list(range(256))
    h = page_hash(token_ids, 0, 64)
    assert isinstance(h, int)
    assert h >= 0


def test_page_hash_different_page_differs():
    token_ids = list(range(256))
    assert page_hash(token_ids, 0, 64) != page_hash(token_ids, 1, 64)


def test_page_pool_alloc_free_cycle():
    pool = make_pool(4, 64)
    a = pool.alloc()
    b = pool.alloc()
    assert a != b
    pool.free(a)
    pool.free(b)
    c = pool.alloc()
    assert c in (a, b)


def test_page_pool_alloc_when_full():
    pool = make_pool(2, 64)
    pool.alloc()
    pool.alloc()
    assert pool.alloc() == -1


def test_page_pool_lru_eviction():
    pool = make_pool(2, 64)
    p0 = pool.alloc()
    p1 = pool.alloc()
    pool.record(p0, list(range(64)), 0)
    pool.record(p1, list(range(64, 128)), 0)
    pool.free(p0)
    pool.free(p1)
    pool.alloc()
    assert p0 in pool._alloc._lru or p1 in pool._alloc._lru


def test_page_pool_inc_ref_and_free():
    pool = make_pool(2, 64)
    p = pool.alloc()
    pool.inc_ref(p)
    assert pool._alloc._refs[p] == 2
    pool.free(p)
    assert pool._alloc._refs[p] == 1
    pool.free(p)
    assert pool._alloc._refs[p] == 0


def test_page_pool_keep_cached_realloc():
    """Free mask has priority over LRU; cached page returned only when no free pages."""
    pool = make_pool(3, 64)
    p0 = pool.alloc()
    p1 = pool.alloc()
    p2 = pool.alloc()
    for p in (p0, p1, p2):
        pool.record(p, [p] * 64, 0)
    pool.free(p0)
    pool.free(p1)
    pool.free(p2)
    assert pool.alloc() == p0


def test_prefix_cache_lookup_returns_hits():
    token_ids = list(range(256))
    pool = make_pool(16, 64)
    pages = [pool.alloc() for _ in range(4)]
    for i, p in enumerate(pages):
        pool.record(p, token_ids, i)
        pool.free(p)
    hits = pool.lookup(token_ids)
    assert hits == pages


def test_prefix_cache_lookup_stops_at_first_miss():
    token_ids = list(range(256))
    pool = make_pool(16, 64)
    p0 = pool.alloc()
    pool.record(p0, token_ids, 0)
    pool.free(p0)
    p1 = pool.alloc()
    pool.record(p1, [99] * 64, 1)
    pool.free(p1)
    hits = pool.lookup(token_ids)
    assert len(hits) == 1
    assert hits[0] == p0


def test_prefix_cache_ignores_partial_last_page():
    token_ids = list(range(100))
    pool = make_pool(16, 64)
    p = pool.alloc()
    pool.record(p, token_ids, 0)
    pool.free(p)
    hits = pool.lookup(token_ids)
    assert len(hits) == 1


def test_prefix_cache_on_evict_clears_mappings():
    pool = make_pool(4, 64)
    p = pool.alloc()
    pool.record(p, list(range(64)), 0)
    pool.free(p)
    assert p in pool._prefix._page_to_hash
    pool._prefix.evict(p)
    assert p not in pool._prefix._page_to_hash


def test_prefix_cache_has_page():
    pool = make_pool(4, 64)
    p = pool.alloc()
    assert p not in pool._prefix._page_to_hash
    pool.record(p, list(range(64)), 0)
    pool.free(p)
    assert p in pool._prefix._page_to_hash


def test_task_table_set_get():
    table = TaskTable(page_size=64)
    table.set("task1", [0, 1, 2], 128)
    assert table.get("task1") == [0, 1, 2]
    assert table.get_cached("task1") == 128


def test_task_table_get_missing():
    table = TaskTable(page_size=64)
    assert table.get("nonexistent") == []
    assert table.get_cached("nonexistent") == 0


def test_task_table_pop():
    table = TaskTable(page_size=64)
    table.set("task1", [0, 1], 64)
    pages, cached = table.pop("task1")
    assert pages == [0, 1]
    assert cached == 64
    assert table.get("task1") == []


def test_kv_cache_task_extend_allocates():
    cache = PageCache(
        n_layers=1,
        n_pages=8,
        page_size=64,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache._table.set("task1", [], 0)
    ok = cache.task_extend("task1", 200)
    assert ok
    assert len(cache._table.get("task1")) == 4


def test_kv_cache_task_extend_fails_when_pool_full():
    cache = PageCache(
        n_layers=1,
        n_pages=2,
        page_size=64,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache._table.set("task1", [0, 1], 0)
    ok = cache.task_extend("task1", 300)
    assert not ok


def test_task_table_table_tensor():
    table = TaskTable(page_size=64)
    table.set("a", [0, 1], 0)
    table.set("b", [2, 3, 4], 0)
    t = table.table_tensor(["a", "b"], torch.device("cpu"))
    assert t.shape == (2, 3)
    assert t[0].tolist() == [0, 1, -1]
    assert t[1].tolist() == [2, 3, 4]


def test_task_table_table_tensor_empty_input():
    table = TaskTable(page_size=64)
    t = table.table_tensor([], torch.device("cpu"))
    assert t.numel() == 0


def test_storage_write_gather_single_page():
    storage = Storage(
        n_layers=2,
        n_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    page_table = torch.tensor([[0]], dtype=torch.long)
    k = torch.randn(1, 2, 2, 8)
    v = torch.randn(1, 2, 2, 8)

    storage.write(0, page_table, 0, k, v)
    gk, gv = storage.gather(0, page_table, 2)
    assert torch.allclose(gk, k)


def test_storage_write_cross_page():
    storage = Storage(
        n_layers=1,
        n_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    page_table = torch.tensor([[0, 1]], dtype=torch.long)
    k = torch.randn(1, 8, 2, 8)
    v = torch.randn(1, 8, 2, 8)

    storage.write(0, page_table, 0, k, v)
    gk, gv = storage.gather(0, page_table, 8)
    assert torch.allclose(gk, k)


def test_storage_gather_truncates_to_total_len():
    storage = Storage(
        n_layers=1,
        n_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    page_table = torch.tensor([[0, 1]], dtype=torch.long)
    k = torch.randn(1, 6, 2, 8)
    v = torch.randn(1, 6, 2, 8)
    storage.write(0, page_table, 0, k, v)

    gk, gv = storage.gather(0, page_table, 5)
    assert gk.shape == (1, 5, 2, 8)


def test_storage_gather_clamps_negative_padding():
    storage = Storage(
        n_layers=1,
        n_pages=8,
        page_size=4,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    page_table = torch.tensor([[0, -1]], dtype=torch.long)
    gk, gv = storage.gather(0, page_table, 4)
    assert gk.shape == (1, 4, 2, 8)
