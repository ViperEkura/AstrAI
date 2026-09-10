"""Unit tests for inference cache components."""

import pytest
import torch

from astrai.inference.cache import (
    Allocator,
    KVStorage,
    PagePool,
    RadixCache,
    ReqToTokenPool,
    TaskCacheManager,
    page_hash,
)
from astrai.inference.workspace import InferenceWorkspace


def _ws(pool: PagePool) -> InferenceWorkspace:
    """Workspace sized to the pool (bind_tasks requires it)."""
    return InferenceWorkspace(
        pool.max_batch_size,
        pool.max_seq_len,
        max_q_heads=2,
        head_dim=4,
        device=pool.device,
        dtype=pool.dtype,
    )


def _make_task_cache(pool: PagePool) -> TaskCacheManager:
    return TaskCacheManager(pool)


# ---- page_hash ----


def test_page_hash_different_page_differs():
    token_ids = list(range(256))
    assert page_hash(token_ids, 0, 64) != page_hash(token_ids, 1, 64)


# ---- Allocator ----


def test_allocator_alloc_free_cycle():
    alloc = Allocator(4)
    a = alloc.alloc()
    b = alloc.alloc()
    assert a != b
    alloc.free(a)
    alloc.free(b)
    c = alloc.alloc()
    assert c in (a, b)


def test_allocator_alloc_when_full():
    alloc = Allocator(2)
    alloc.alloc()
    alloc.alloc()
    assert alloc.alloc() == -1


def test_allocator_lru_eviction():
    alloc = Allocator(2)
    p0 = alloc.alloc()
    p1 = alloc.alloc()
    alloc.free(p0, keep_cached=True)
    alloc.free(p1, keep_cached=True)
    alloc.alloc()
    assert p0 in alloc._lru or p1 in alloc._lru


def test_allocator_inc_ref_and_free():
    alloc = Allocator(2)
    p = alloc.alloc()
    alloc.inc_ref(p)
    assert alloc._refs[p] == 2
    alloc.free(p)
    assert alloc._refs[p] == 1
    alloc.free(p)
    assert alloc._refs[p] == 0


# ---- RadixCache ----


def test_prefix_cache_lookup_returns_hits():
    token_ids = list(range(256))
    prefix = RadixCache(64)
    pages = [0, 1, 2, 3]
    for i, p in enumerate(pages):
        prefix.record(p, token_ids, i)
    hits = prefix.lookup(token_ids)
    assert hits == pages


def test_prefix_cache_lookup_stops_at_first_miss():
    token_ids = list(range(256))
    prefix = RadixCache(64)
    prefix.record(0, token_ids, 0)
    prefix.record(1, [99] * 64, 1)
    hits = prefix.lookup(token_ids)
    assert len(hits) == 1
    assert hits[0] == 0


def test_prefix_cache_ignores_partial_last_page():
    token_ids = list(range(100))
    prefix = RadixCache(64)
    prefix.record(0, token_ids, 0)
    hits = prefix.lookup(token_ids)
    assert len(hits) == 1


def test_prefix_cache_on_evict_clears_mappings():
    prefix = RadixCache(64)
    assert not prefix.has_page(0)
    prefix.record(0, list(range(64)), 0)
    assert prefix.has_page(0)
    prefix.evict(0)
    assert not prefix.has_page(0)


def test_prefix_cache_does_not_reuse_page_without_parent_prefix():
    prefix = RadixCache(2)
    prefix.record(0, [1, 2, 3, 4], 0)
    prefix.record(1, [1, 2, 3, 4, 5, 6], 1)
    prefix.record(2, [9, 10, 5, 6], 0)
    prefix.record(3, [9, 10, 5, 6, 7, 8], 1)
    assert prefix.lookup([1, 2, 3, 4, 5, 6]) == [0, 1]
    assert prefix.lookup([9, 10, 5, 6, 7, 8]) == [2, 3]


def test_prefix_cache_shares_branch_prefix():
    prefix = RadixCache(2)
    prefix.record(0, [1, 2, 3, 4], 0)
    prefix.record(1, [1, 2, 3, 4], 1)
    prefix.record(2, [1, 2, 7, 8], 1)
    assert prefix.lookup([1, 2, 3, 4]) == [0, 1]
    assert prefix.lookup([1, 2, 7, 8]) == [0, 2]
    prefix.evict(1)
    assert prefix.lookup([1, 2, 3, 4]) == [0]
    assert prefix.lookup([1, 2, 7, 8]) == [0, 2]


def test_prefix_cache_does_not_record_partial_page():
    prefix = RadixCache(4)
    prefix.record(0, [1, 2, 3, 4, 5, 6], 0)
    prefix.record(1, [1, 2, 3, 4, 5, 6], 1)
    assert prefix.lookup([1, 2, 3, 4, 5, 6]) == [0]

    prefix.record(1, [1, 2, 3, 4, 5, 6, 7, 8], 1)
    assert prefix.lookup([1, 2, 3, 4, 5, 6, 7, 8]) == [0, 1]


def test_page_pool_task_cacheable_ids_excludes_unmaterialized_tail():
    pool = _make_paged_pool_ps64()
    task_cache = _make_task_cache(pool)
    assert task_cache.task_cacheable_ids("missing", [1, 2], [3, 4]) == [1, 2, 3]


# ---- ReqToTokenPool ----


def test_req_to_token_pool_alloc_free():
    pool = ReqToTokenPool(4, 128, torch.device("cpu"))
    assert pool.req_to_token.dtype == torch.int32
    slots = pool.alloc(2)
    assert len(slots) == 2
    assert len(pool.free_slots) == 2
    pool.free(slots)
    assert len(pool.free_slots) == 4


def test_req_to_token_pool_alloc_when_full():
    pool = ReqToTokenPool(2, 128, torch.device("cpu"))
    pool.alloc(2)
    assert pool.alloc(1) is None


def test_req_to_token_pool_write():
    pool = ReqToTokenPool(4, 128, torch.device("cpu"))
    slots = pool.alloc(1)
    pool.write((slots[0], slice(0, 3)), torch.tensor([10, 20, 30]))
    assert pool.req_to_token[slots[0], 0].item() == 10
    assert pool.req_to_token[slots[0], 2].item() == 30


# ---- KVStorage ----


def test_kv_storage_buffer_shape():
    storage = KVStorage(
        size=32,
        n_layers=3,
        n_kv_heads=8,
        head_dim=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert storage.k_buffer.shape == (3, 32, 8, 16)
    assert storage.v_buffer.shape == (3, 32, 8, 16)


# ---- PagePool (contiguous mode) ----


def _make_contiguous_pool(**kwargs):
    defaults = dict(
        n_layers=2,
        n_kv_heads=4,
        head_dim=8,
        max_batch_size=4,
        max_seq_len=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    defaults.update(kwargs)
    return PagePool(**defaults)


def test_page_pool_contiguous_task_alloc_free():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    assert task_cache.task_alloc("t1", [1, 2, 3])
    assert "t1" in task_cache._states
    task_cache.task_free("t1")
    assert "t1" not in task_cache._states


def test_page_pool_contiguous_task_extend():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", [1, 2, 3])
    assert task_cache.task_extend("t1", 3)
    assert task_cache.task_extend("t1", 63)
    assert not task_cache.task_extend("t1", 64)


def test_page_pool_contiguous_task_cached():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", [1, 2, 3])
    assert task_cache.task_cached("t1") == 0


def test_page_pool_contiguous_bind_tasks_prefill():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", list(range(10)))
    task_cache.task_alloc("t2", list(range(10)))
    kv = task_cache.bind(["t1", "t2"], _ws(pool), start_pos=0)
    assert kv.out_cache_loc.shape == (20,)
    assert kv.out_cache_loc.dtype == torch.int32
    assert kv.seq_lens.tolist() == [10, 10]
    assert kv.req_pool_indices.shape == (2,)
    assert kv.req_pool_indices.dtype == torch.int32


def test_page_pool_bind_tasks_builds_compact_q_tile_mapping():
    pool = _make_contiguous_pool(max_batch_size=3, max_seq_len=256)

    kv = pool.bind_tasks([0, 1, 2], [70, 10, 130], _ws(pool), start_pos=0)

    assert kv.qo_indptr.tolist() == [0, 70, 80, 210]
    assert kv.q_tile_to_batch.tolist() == [0, 0, 1, 2, 2, 2]
    assert kv.q_tile_to_index.tolist() == [0, 1, 0, 0, 1, 2]


def test_page_pool_contiguous_bind_tasks_decode():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", list(range(10)))
    task_cache.task_alloc("t2", list(range(8)))
    # Simulate one decode extension so seq_lens advance to 11 and 9.
    assert task_cache.task_extend("t1", 10)
    assert task_cache.task_extend("t2", 8)
    kv = task_cache.bind(["t1", "t2"], _ws(pool))
    assert kv.out_cache_loc.shape == (2,)
    assert kv.seq_lens.tolist() == [11, 9]


def test_page_pool_contiguous_bind_roundtrip():
    """Write KV via bind_tasks, then gather via req_to_token indexing."""
    pool = _make_contiguous_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", list(range(4)))

    kv = task_cache.bind(["t1"], _ws(pool), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0, kv.out_cache_loc] = k
    kv.v_buffer[0, kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :4]
    gathered_k = kv.k_buffer[0, indices]
    gathered_v = kv.v_buffer[0, indices]
    assert torch.allclose(gathered_k, k)
    assert torch.allclose(gathered_v, v)


# ---- PagePool (paged mode, page_size=1) ----


def _make_paged_pool(**kwargs):
    defaults = dict(
        n_layers=1,
        n_kv_heads=2,
        head_dim=4,
        max_batch_size=4,
        max_seq_len=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
        page_size=1,
        n_tokens=128,
    )
    defaults.update(kwargs)
    return PagePool(**defaults)


def test_page_pool_paged_task_alloc():
    pool = _make_paged_pool()
    task_cache = _make_task_cache(pool)
    assert task_cache.task_alloc("t1", list(range(10)))
    state = task_cache._states["t1"]
    assert len(state.pages) == 10
    assert pool.req_pool.req_to_token[state.req_idx, 0].item() == state.pages[0]


def test_page_pool_paged_task_extend():
    pool = _make_paged_pool()
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", list(range(4)))
    assert task_cache.task_extend("t1", 4)
    req_idx = task_cache._states["t1"].req_idx
    slot = pool.req_pool.req_to_token[req_idx, 4].item()
    assert slot >= 0


def test_page_pool_paged_task_free_releases_slots():
    pool = _make_paged_pool(n_tokens=16)
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", list(range(8)))
    task_cache.task_free("t1")
    assert "t1" not in task_cache._states
    assert len(pool.req_pool.free_slots) == 4


def test_page_pool_paged_bind_roundtrip():
    pool = _make_paged_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", list(range(4)))

    kv = task_cache.bind(["t1"], _ws(pool), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0, kv.out_cache_loc] = k
    kv.v_buffer[0, kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :4]
    gathered_k = kv.k_buffer[0, indices]
    assert torch.allclose(gathered_k, k)


# ---- PagePool (paged mode, page_size>1) ----


def _make_paged_pool_ps64(**kwargs):
    defaults = dict(
        n_layers=1,
        n_kv_heads=2,
        head_dim=4,
        max_batch_size=4,
        max_seq_len=256,
        device=torch.device("cpu"),
        dtype=torch.float32,
        page_size=64,
        n_tokens=512,
    )
    defaults.update(kwargs)
    return PagePool(**defaults)


def test_page_pool_paged_ps64_task_alloc():
    pool = _make_paged_pool_ps64()
    task_cache = _make_task_cache(pool)
    prompt = list(range(200))
    assert task_cache.task_alloc("t1", prompt)
    assert task_cache.task_cached("t1") == 0
    n_pages = (200 + 63) // 64
    assert len(task_cache._states["t1"].pages) == n_pages


def test_page_pool_paged_ps64_task_extend_crosses_page():
    pool = _make_paged_pool_ps64()
    task_cache = _make_task_cache(pool)
    task_cache.task_alloc("t1", list(range(64)))
    assert task_cache.task_extend("t1", 64)
    assert len(task_cache._states["t1"].pages) >= 2


def test_page_pool_prefix_hit_populates_request_mapping():
    pool = _make_paged_pool_ps64(page_size=2, max_seq_len=8, n_tokens=16)
    task_cache = _make_task_cache(pool)
    prompt = [11, 12, 13, 14]

    assert task_cache.task_alloc("first", prompt)
    task_cache.task_record_hashes("first", prompt)
    task_cache.task_free("first")

    assert task_cache.task_alloc("second", prompt)
    second_state = task_cache._states["second"]
    expected = [
        page * pool.page_size + offset
        for page in second_state.pages
        for offset in range(pool.page_size)
    ]

    assert second_state.cached == len(prompt)
    assert (
        pool.req_pool.req_to_token[second_state.req_idx, : len(prompt)].tolist()
        == expected
    )


def test_task_cache_invalidation_drops_cross_version_prefix_hits():
    pool = _make_paged_pool_ps64(page_size=2, max_seq_len=8, n_tokens=16)
    task_cache = _make_task_cache(pool)
    prompt = [11, 12, 13, 14]

    assert task_cache.task_alloc("first", prompt)
    task_cache.task_record_hashes("first", prompt)
    task_cache.task_free("first")
    assert task_cache.task_alloc("cached", prompt)
    assert task_cache.task_cached("cached") == len(prompt)

    with pytest.raises(RuntimeError, match="while tasks are active"):
        task_cache.invalidate_cache()

    task_cache.task_free("cached")
    assert task_cache.invalidate_cache() == 2
    assert task_cache.task_alloc("after_update", prompt)
    assert task_cache.task_cached("after_update") == 0


def test_page_pool_paged_ps64_bind_roundtrip():
    pool = _make_paged_pool_ps64(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    prompt = list(range(128))
    task_cache.task_alloc("t1", prompt)

    kv = task_cache.bind(["t1"], _ws(pool), start_pos=0)
    k = torch.randn(1, 128, 2, 4)
    v = torch.randn(1, 128, 2, 4)
    kv.k_buffer[0, kv.out_cache_loc] = k
    kv.v_buffer[0, kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :128]
    gathered_k = kv.k_buffer[0, indices]
    assert torch.allclose(gathered_k, k)
