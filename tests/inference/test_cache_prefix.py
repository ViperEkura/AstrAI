import pytest
import torch

from astrai.inference.core.cache import KVCacheManager
from astrai.inference.core.request import Request
from tests.support.cache import (
    appended_slots,
    assert_allocator_cache_consistent,
    bind_cache,
    make_cache_workspace,
    make_paged_pool,
    make_paged_pool_ps64,
    physical_slots,
)
from tests.support.cache import kv_device as kv_device


def test_page_pool_prefix_hit_populates_request_mapping():
    pool = make_paged_pool_ps64(page_size=2, max_seq_len=8, n_tokens=16)
    task_cache = KVCacheManager(pool)
    prompt = [11, 12, 13, 14]

    req_first = Request("first", prompt)
    assert task_cache.alloc([req_first])[0]
    task_cache.commit([req_first], [len(prompt)])
    task_cache.free([req_first])

    req_second = Request("second", prompt)
    assert task_cache.alloc([req_second])[0]
    second_state = task_cache._states["second"]
    expected = [
        page * pool.page_size + offset
        for page in second_state.pages
        for offset in range(pool.page_size)
    ]

    assert req_second.cached_tokens == len(prompt)
    assert (
        physical_slots(
            pool.req_pool.page_table[second_state.req_idx],
            pool.page_size,
            0,
            len(prompt),
        ).tolist()
        == expected
    )


def test_task_cache_invalidation_drops_cross_version_prefix_hits():
    pool = make_paged_pool_ps64(page_size=2, max_seq_len=8, n_tokens=16)
    task_cache = KVCacheManager(pool)
    prompt = [11, 12, 13, 14]

    req_first = Request("first", prompt)
    assert task_cache.alloc([req_first])[0]
    task_cache.commit([req_first], [len(prompt)])
    task_cache.free([req_first])
    req_cached = Request("cached", prompt)
    assert task_cache.alloc([req_cached])[0]
    assert req_cached.cached_tokens == len(prompt)

    with pytest.raises(RuntimeError, match="while requests are active"):
        task_cache.clear_prefix_cache()

    task_cache.free([req_cached])
    assert task_cache.clear_prefix_cache() == 2
    req_after_update = Request("after_update", prompt)
    assert task_cache.alloc([req_after_update])[0]
    assert req_after_update.cached_tokens == 0


def test_publish_prefix_requires_explicit_materialized_end():
    pool = make_paged_pool(page_size=4)
    task_cache = KVCacheManager(pool)
    prompt = list(range(8))
    req_writer = Request("writer", prompt)
    assert task_cache.alloc([req_writer])[0]

    with pytest.raises(TypeError, match="ends"):
        task_cache.commit([req_writer])
    assert pool.prefix.lookup(prompt) == []


@pytest.mark.parametrize("allocated, ids_len, end", [(8, 6, 8), (4, 8, 8)])
def test_publish_prefix_is_bounded_by_token_ids_and_allocated_pages(
    allocated, ids_len, end
):
    pool = make_paged_pool(page_size=4)
    task_cache = KVCacheManager(pool)
    req_writer = Request("writer", list(range(allocated)))
    assert task_cache.alloc([req_writer])[0]
    req_writer.prompt_ids = list(range(ids_len))
    task_cache.commit([req_writer], [end])
    assert pool.prefix.lookup(list(range(8))) == [task_cache._states["writer"].pages[0]]


def test_commit_rejects_negative_end():
    pool = make_paged_pool(page_size=4)
    task_cache = KVCacheManager(pool)
    prompt = list(range(8))
    req_writer = Request("writer", prompt)
    assert task_cache.alloc([req_writer])[0]
    with pytest.raises(ValueError, match="nonnegative"):
        task_cache.commit([req_writer], [-1])
    assert pool.prefix.lookup(prompt) == []


def test_chunked_prefix_only_reuses_fully_written_kv(kv_device):
    pool = make_paged_pool(page_size=4, max_seq_len=16, n_tokens=64, device=kv_device)
    task_cache = KVCacheManager(pool)
    ws = make_cache_workspace(pool)
    prompt = list(range(8))
    req_writer = Request("writer", prompt)
    assert task_cache.alloc([req_writer])[0]
    writer = task_cache._states["writer"]
    pool._storage.k_buffer.fill_(float("nan"))
    pool._storage.v_buffer.fill_(float("nan"))
    expected_k = torch.arange(64, dtype=pool.dtype, device=kv_device).reshape(8, 2, 4)
    expected_v = expected_k + 1000

    for start, end in [(0, 2), (2, 4), (4, 6), (6, 8)]:
        kv = bind_cache(task_cache, ws, [req_writer], start_pos=start, seq_ends=[end])
        assert appended_slots(kv).numel() == end - start
        kv.k_buffer[0].flatten(0, 1)[appended_slots(kv)] = expected_k[start:end]
        kv.v_buffer[0].flatten(0, 1)[appended_slots(kv)] = expected_v[start:end]
        task_cache.commit([req_writer], [end])

        req_reader = Request("reader", prompt)
        assert task_cache.alloc([req_reader])[0]
        reader = task_cache._states["reader"]
        n_cached = end // pool.page_size * pool.page_size
        n_pages = n_cached // pool.page_size
        assert req_reader.cached_tokens == n_cached
        assert reader.pages[:n_pages] == writer.pages[:n_pages]
        assert set(reader.pages[n_pages:]).isdisjoint(writer.pages[n_pages:])
        cached_slots = physical_slots(
            pool.req_pool.page_table[reader.req_idx], pool.page_size, 0, n_cached
        )
        torch.testing.assert_close(
            kv.k_buffer[0].flatten(0, 1)[cached_slots], expected_k[:n_cached]
        )
        torch.testing.assert_close(
            kv.v_buffer[0].flatten(0, 1)[cached_slots], expected_v[:n_cached]
        )
        unwritten = physical_slots(
            pool.req_pool.page_table[writer.req_idx], pool.page_size, end, len(prompt)
        )
        assert torch.isnan(kv.k_buffer[0].flatten(0, 1)[unwritten]).all().item()
        assert torch.isnan(kv.v_buffer[0].flatten(0, 1)[unwritten]).all().item()
        task_cache.free([req_reader])
        assert_allocator_cache_consistent(pool.allocator, pool.prefix)

    task_cache.free([req_writer])
    assert task_cache.clear_prefix_cache() == 2
    assert_allocator_cache_consistent(pool.allocator, pool.prefix)


@pytest.mark.parametrize("release_first", [False, True])
def test_prefix_replacement_keeps_allocator_identity_consistent(release_first):
    pool = make_paged_pool(page_size=4, max_seq_len=16, n_tokens=32)
    task_cache = KVCacheManager(pool)
    prompt = list(range(8))
    # Both prefills reserve private pages before either publishes its KV.
    req_first = Request("first", prompt)
    assert task_cache.alloc([req_first])[0]
    req_second = Request("second", prompt)
    assert task_cache.alloc([req_second])[0]
    first_pages = list(task_cache._states["first"].pages)
    second_pages = list(task_cache._states["second"].pages)
    task_cache.commit([req_first], [8])
    if release_first:
        task_cache.free([req_first])

    # Replacing the first page leaves the already-materialized descendant
    # reusable, but drops the old first page's cached allocator identity.
    task_cache.commit([req_second], [4])
    prefix = pool.prefix
    alloc = pool.allocator
    assert prefix.lookup(prompt) == [second_pages[0], first_pages[1]]
    assert not prefix.has_page(first_pages[0])
    assert alloc.ref_count(first_pages[0]) == (0 if release_first else 1)
    assert_allocator_cache_consistent(alloc, prefix)

    task_cache.commit([req_second], [8])
    assert prefix.lookup(prompt) == second_pages
    assert all(not prefix.has_page(page) for page in first_pages)
    task_cache.free([req_first])
    assert all(alloc._free_mask & (1 << page) for page in first_pages)
    task_cache.free([req_second])
    assert_allocator_cache_consistent(alloc, prefix)
    req_reader = Request("reader", prompt)
    assert task_cache.alloc([req_reader])[0]
    assert req_reader.cached_tokens == len(prompt)
    assert task_cache._states["reader"].pages == second_pages
    assert_allocator_cache_consistent(alloc, prefix)


def test_batched_commit_and_free_preserve_prefix_and_slot_ownership():
    pool = make_paged_pool(page_size=2, max_seq_len=12, n_tokens=16)
    cache = KVCacheManager(pool)
    a = Request("a", [1, 2, 3, 4])
    b = Request("b", [5, 6, 7, 8])
    assert cache.alloc([a, b]) == [True, True]
    cache.commit([a, b], [3, 4])
    cache.commit([a, b], [2, 4])
    assert len(pool.prefix.lookup(a.prompt_ids)) == 1
    assert len(pool.prefix.lookup(b.prompt_ids)) == 2
    cache.free([a, b])
    replacement = Request("a", a.prompt_ids)
    assert cache.alloc([replacement]) == [True]
    assert replacement.cached_tokens == 2
    cache.commit([a], [4])
    cache.free([a, b])
    assert cache.request_count == 1
    assert len(pool.prefix.lookup(a.prompt_ids)) == 1
    cache.free([replacement])
    assert cache.request_count == 0
