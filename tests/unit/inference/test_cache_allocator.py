"""Unit tests for inference cache components."""

import random

import pytest

from astrai.inference.core.cache import (
    Allocator,
    RadixCache,
)
from tests.support.cache import (
    _assert_allocator_cache_consistent,
    _assert_prefix_reachable,
)


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


def test_prefix_cache_repeated_record_preserves_descendants_and_branches():
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)
    branch = [1, 2, 7, 8]
    prefix.record(3, branch, 1)
    original_nodes = dict(prefix._page_to_node)

    for _ in range(3):
        for i in range(3):
            assert prefix.record(i, prompt, i) == []
        assert prefix.lookup(prompt) == [0, 1, 2]
        assert prefix.lookup(branch) == [0, 3]
        assert prefix._page_to_node == original_nodes
        _assert_prefix_reachable(prefix)


@pytest.mark.parametrize("victim, revoked", [(0, {0, 1, 2, 3}), (1, {1, 2})])
def test_prefix_cache_ancestor_eviction_revokes_only_its_subtree(victim, revoked):
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)
    prefix.record(3, [1, 2, 7, 8], 1)
    prefix.record(4, [9, 10], 0)

    assert set(prefix.evict(victim)) == revoked
    assert prefix.evict(victim) == []
    assert prefix.lookup(prompt) == ([] if victim == 0 else [0])
    assert prefix.lookup([1, 2, 7, 8]) == ([] if victim == 0 else [0, 3])
    assert prefix.lookup([9, 10]) == [4]
    for page in range(5):
        assert prefix.has_page(page) == (page not in revoked)
    _assert_prefix_reachable(prefix)


def test_prefix_cache_relocating_physical_page_revokes_old_descendants():
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)

    assert set(prefix.record(0, [9, 10], 0)) == {1, 2}
    assert prefix.lookup(prompt) == []
    assert prefix.lookup([9, 10]) == [0]
    assert not prefix.has_page(1)
    assert not prefix.has_page(2)
    _assert_prefix_reachable(prefix)


def test_prefix_cache_replacing_same_prefix_preserves_descendants():
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)

    assert prefix.record(3, prompt, 0) == [0]
    assert prefix.lookup(prompt) == [3, 1, 2]
    assert not prefix.has_page(0)
    _assert_prefix_reachable(prefix)
    assert set(prefix.evict(3)) == {1, 2, 3}
    assert not prefix._page_to_node


def test_prefix_cache_does_not_index_unreachable_descendant():
    prefix = RadixCache(2)
    prompt = [1, 2, 3, 4]
    assert prefix.record(1, prompt, 1) == []
    assert not prefix.has_page(1)
    prefix.record(0, prompt, 0)
    assert prefix.lookup(prompt) == [0]
    prefix.record(1, prompt, 1)
    assert prefix.lookup(prompt) == [0, 1]
    _assert_prefix_reachable(prefix)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("hold_descendant", [False, True])
def test_allocator_ancestor_eviction_reclaims_only_unreferenced_pages(
    batched, hold_descendant
):
    alloc = Allocator(4)
    prefix = RadixCache(2)
    alloc.on_evict = prefix.evict
    assert alloc.alloc_many(4) == [0, 1, 2, 3]
    prompt = [1, 2, 3, 4, 5, 6]
    for i in range(3):
        prefix.record(i, prompt, i)
    prefix.record(3, [9, 10], 0)
    released = [0, 2, 3] if hold_descendant else [0, 1, 2, 3]
    alloc.free_many(released, keep_cached_for=prefix.has_page)
    _assert_allocator_cache_consistent(alloc, prefix)

    taken = alloc.alloc_many(2) if batched else [alloc.alloc()]
    expected = [0, 2 if hold_descendant else 1] if batched else [0]
    assert taken == expected
    assert prefix.lookup(prompt) == []
    assert prefix.lookup([9, 10]) == [3]
    assert list(alloc._lru) == [3]
    if hold_descendant:
        assert alloc.ref_count(1) == 1
        assert not alloc._free_mask & (1 << 1)
        alloc.free(1, keep_cached=prefix.has_page(1))
    _assert_allocator_cache_consistent(alloc, prefix)
    assert alloc.clear_cached() == 1
    alloc.free_many(taken)
    assert alloc._free_mask == (1 << 4) - 1
    _assert_allocator_cache_consistent(alloc, prefix)


def test_allocator_failed_bulk_allocation_does_not_revoke_prefixes():
    alloc = Allocator(2)
    prefix = RadixCache(2)
    alloc.on_evict = prefix.evict
    assert alloc.alloc_many(2) == [0, 1]
    prefix.record(0, [1, 2], 0)
    alloc.free(0, keep_cached=True)

    assert alloc.alloc_many(2) is None
    assert prefix.lookup([1, 2]) == [0]
    _assert_allocator_cache_consistent(alloc, prefix)


def test_allocator_alloc_many_matches_free_set_exactly():
    """Bulk allocation must yield exactly the pages the free set held.

    The word-window harvest once re-issued already-harvested pages (stale
    mask read across windows) and once left cleared pages marked free
    (mixed absolute/shifted bit coordinates); this randomized
    reference-model check pins the mask == free-set invariant.
    """
    rng = random.Random(7)
    alloc = Allocator(300)
    free = set(range(300))
    held = []
    for _ in range(400):
        if free and rng.random() < 0.55:
            n = rng.randint(1, 25)
            got = alloc.alloc_many(n)
            if got is None:
                assert len(free) < n
                continue
            assert len(got) == n
            assert len(set(got)) == n
            for p in got:
                assert p in free
                free.remove(p)
            held.append(got)
        elif held:
            g = held.pop(rng.randrange(len(held)))
            alloc.free_many(g)
            free.update(g)
        mask = alloc._free_mask
        for p in range(300):
            assert bool(mask >> p & 1) == (p in free)


def test_allocator_alloc_many_fragmentation_roundtrip():
    alloc = Allocator(10000)
    first = alloc.alloc_many(100)
    alloc.free_many(first[0::2])
    assert alloc.alloc_many(50) == sorted(first[0::2])
