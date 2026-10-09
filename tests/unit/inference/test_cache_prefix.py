"""Unit tests for inference cache components."""

import pytest
import torch

from tests.support.cache import (
    _assert_allocator_cache_consistent,
    _make_paged_pool,
    _make_paged_pool_ps64,
    _make_task_cache,
    _ws,
)
from tests.support.cache import kv_device as kv_device


def test_page_pool_prefix_hit_populates_request_mapping():
    pool = _make_paged_pool_ps64(page_size=2, max_seq_len=8, n_tokens=16)
    task_cache = _make_task_cache(pool)
    prompt = [11, 12, 13, 14]

    assert task_cache.alloc_slots("first", prompt)
    task_cache.record_block_hashes("first", prompt, materialized_end=len(prompt))
    task_cache.free_slots("first")

    assert task_cache.alloc_slots("second", prompt)
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

    assert task_cache.alloc_slots("first", prompt)
    task_cache.record_block_hashes("first", prompt, materialized_end=len(prompt))
    task_cache.free_slots("first")
    assert task_cache.alloc_slots("cached", prompt)
    assert task_cache.cached_tokens("cached") == len(prompt)

    with pytest.raises(RuntimeError, match="while requests are active"):
        task_cache.invalidate_cache()

    task_cache.free_slots("cached")
    assert task_cache.invalidate_cache() == 2
    assert task_cache.alloc_slots("after_update", prompt)
    assert task_cache.cached_tokens("after_update") == 0


def test_record_block_hashes_requires_explicit_materialized_end():
    pool = _make_paged_pool(page_size=4)
    task_cache = _make_task_cache(pool)
    prompt = list(range(8))
    assert task_cache.alloc_slots("writer", prompt)

    with pytest.raises(TypeError, match="materialized_end"):
        task_cache.record_block_hashes("writer", prompt)
    assert pool.strategy._prefix.lookup(prompt) == []


@pytest.mark.parametrize("allocated, ids_len, end", [(8, 6, 8), (4, 8, 8)])
def test_record_block_hashes_is_bounded_by_token_ids_and_allocated_pages(
    allocated, ids_len, end
):
    pool = _make_paged_pool(page_size=4)
    task_cache = _make_task_cache(pool)
    assert task_cache.alloc_slots("writer", list(range(allocated)))
    task_cache.record_block_hashes("writer", list(range(ids_len)), materialized_end=end)
    assert pool.strategy._prefix.lookup(list(range(8))) == [
        task_cache._states["writer"].pages[0]
    ]


@pytest.mark.parametrize("start, end", [(-1, 4), (0, -1)])
def test_record_block_hashes_rejects_negative_watermark_or_page(start, end):
    pool = _make_paged_pool(page_size=4)
    task_cache = _make_task_cache(pool)
    prompt = list(range(8))
    assert task_cache.alloc_slots("writer", prompt)
    with pytest.raises(ValueError, match="nonnegative"):
        task_cache.record_block_hashes("writer", prompt, start, materialized_end=end)
    assert pool.strategy._prefix.lookup(prompt) == []


def test_chunked_prefix_only_reuses_fully_written_kv(kv_device):
    pool = _make_paged_pool(page_size=4, max_seq_len=16, n_tokens=64, device=kv_device)
    task_cache = _make_task_cache(pool)
    ws = _ws(pool)
    prompt = list(range(8))
    assert task_cache.alloc_slots("writer", prompt)
    writer = task_cache._states["writer"]
    pool._storage.k_buffer.fill_(float("nan"))
    pool._storage.v_buffer.fill_(float("nan"))
    expected_k = torch.arange(64, dtype=pool.dtype, device=kv_device).reshape(8, 2, 4)
    expected_v = expected_k + 1000

    for start, end in [(0, 2), (2, 4), (4, 6), (6, 8)]:
        kv = task_cache.bind(["writer"], ws, start_pos=start, seq_ends=[end])
        assert kv.out_cache_loc.numel() == end - start
        kv.k_buffer[0, kv.out_cache_loc] = expected_k[start:end]
        kv.v_buffer[0, kv.out_cache_loc] = expected_v[start:end]
        task_cache.record_block_hashes(
            "writer", prompt, start // pool.page_size, materialized_end=end
        )

        assert task_cache.alloc_slots("reader", prompt)
        reader = task_cache._states["reader"]
        n_cached = end // pool.page_size * pool.page_size
        n_pages = n_cached // pool.page_size
        assert reader.cached == n_cached
        assert reader.pages[:n_pages] == writer.pages[:n_pages]
        assert set(reader.pages[n_pages:]).isdisjoint(writer.pages[n_pages:])
        cached_slots = pool.req_pool.req_to_token[reader.req_idx, :n_cached]
        torch.testing.assert_close(kv.k_buffer[0, cached_slots], expected_k[:n_cached])
        torch.testing.assert_close(kv.v_buffer[0, cached_slots], expected_v[:n_cached])
        unwritten = pool.req_pool.req_to_token[writer.req_idx, end : len(prompt)]
        assert torch.isnan(kv.k_buffer[0, unwritten]).all().item()
        assert torch.isnan(kv.v_buffer[0, unwritten]).all().item()
        task_cache.free_slots("reader")
        _assert_allocator_cache_consistent(pool.strategy._alloc, pool.strategy._prefix)

    task_cache.free_slots("writer")
    assert task_cache.invalidate_cache() == 2
    _assert_allocator_cache_consistent(pool.strategy._alloc, pool.strategy._prefix)


@pytest.mark.parametrize("release_first", [False, True])
def test_prefix_replacement_keeps_allocator_identity_consistent(release_first):
    pool = _make_paged_pool(page_size=4, max_seq_len=16, n_tokens=32)
    task_cache = _make_task_cache(pool)
    prompt = list(range(8))
    # Both prefills reserve private pages before either publishes its KV.
    assert task_cache.alloc_slots("first", prompt)
    assert task_cache.alloc_slots("second", prompt)
    first_pages = list(task_cache._states["first"].pages)
    second_pages = list(task_cache._states["second"].pages)
    task_cache.record_block_hashes("first", prompt, materialized_end=8)
    if release_first:
        task_cache.free_slots("first")

    # Replacing the first page leaves the already-materialized descendant
    # reusable, but drops the old first page's cached allocator identity.
    task_cache.record_block_hashes("second", prompt, materialized_end=4)
    prefix = pool.strategy._prefix
    alloc = pool.strategy._alloc
    assert prefix.lookup(prompt) == [second_pages[0], first_pages[1]]
    assert not prefix.has_page(first_pages[0])
    assert alloc.ref_count(first_pages[0]) == (0 if release_first else 1)
    _assert_allocator_cache_consistent(alloc, prefix)

    task_cache.record_block_hashes("second", prompt, materialized_end=8)
    assert prefix.lookup(prompt) == second_pages
    assert all(not prefix.has_page(page) for page in first_pages)
    task_cache.free_slots("first")
    assert all(alloc._free_mask & (1 << page) for page in first_pages)
    task_cache.free_slots("second")
    _assert_allocator_cache_consistent(alloc, prefix)
    assert task_cache.alloc_slots("reader", prompt)
    assert task_cache.cached_tokens("reader") == len(prompt)
    assert task_cache._states["reader"].pages == second_pages
    _assert_allocator_cache_consistent(alloc, prefix)
