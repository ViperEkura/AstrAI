"""Unit tests for inference cache components."""

import pytest
import torch

from astrai.inference.core.cache import (
    BlockPool,
    KVCacheManager,
)
from astrai.inference.worker.workspace import InferenceWorkspace


def _ws(pool: BlockPool) -> InferenceWorkspace:
    """Workspace sized to the pool (bind_tasks requires it)."""
    return InferenceWorkspace(
        pool.max_batch_size,
        pool.max_seq_len,
        max_q_heads=2,
        head_dim=4,
        device=pool.device,
        dtype=pool.dtype,
    )


def _make_task_cache(pool: BlockPool) -> KVCacheManager:
    return KVCacheManager(pool)


@pytest.fixture(params=["cuda"])
def kv_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def _assert_prefix_reachable(prefix):
    reachable = {}
    pending = list(prefix._root.children.values())
    while pending:
        node = pending.pop()
        assert node.page_idx is not None
        assert node.parent.children[node.tokens] is node
        assert node.page_idx not in reachable
        reachable[node.page_idx] = node
        pending.extend(node.children.values())
    assert prefix._page_to_node == reachable


def _assert_allocator_cache_consistent(alloc, prefix):
    _assert_prefix_reachable(prefix)
    for page in range(alloc._n_pages):
        cached = prefix.has_page(page)
        free = bool(alloc._free_mask & (1 << page))
        assert free == (alloc._refs[page] == 0 and not cached)
        assert (page in alloc._lru) == (alloc._refs[page] == 0 and cached)
    assert alloc._free_mask == sum(
        word << (64 * i) for i, word in enumerate(alloc._words)
    )


# ---- Allocator ----


# ---- RadixCache ----


# ---- ReqToTokenPool ----


# ---- KVStorage ----


# ---- BlockPool (contiguous mode) ----


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
    return BlockPool(**defaults)


# ---- BlockPool (paged mode, page_size=1) ----


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
    return BlockPool(**defaults)


# ---- BlockPool (paged mode, page_size>1) ----


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
    return BlockPool(**defaults)


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


@pytest.mark.parametrize("prompt_lens", [(5,), (5, 1, 2, 6), (4, 5, 8, 3)])
def test_paged_extend_batch_preserves_real_kv_across_mixed_positions(
    kv_device, prompt_lens
):
    pool = _make_paged_pool(page_size=4, max_seq_len=16, n_tokens=64, device=kv_device)
    task_cache = _make_task_cache(pool)
    ws = _ws(pool)
    ids = [f"request{i}" for i in range(len(prompt_lens))]
    for request_id, length in zip(ids, prompt_lens):
        assert task_cache.alloc_slots(request_id, list(range(length)))
        state = task_cache._states[request_id]
        # Unmapped positions must never be mistaken for physical slot zero.
        pool.req_pool.req_to_token[state.req_idx, length:].fill_(-1)

    pool._storage.k_buffer.fill_(float("nan"))
    pool._storage.v_buffer.fill_(float("nan"))
    initial = torch.arange(
        sum(prompt_lens) * 8, dtype=pool.dtype, device=kv_device
    ).reshape(-1, 2, 4)
    kv = task_cache.bind(ids, ws, start_pos=0)
    kv.k_buffer[0, kv.out_cache_loc] = initial
    kv.v_buffer[0, kv.out_cache_loc] = -initial - 1
    expected = list(initial.split(prompt_lens))

    # Includes all-in-page steps, page crossings and mixed in-page/crossing
    # requests. Single prompt=5 must map position 5 to slot 5, not slot 0.
    for step in range(5):
        positions = [length + step for length in prompt_lens]
        assert task_cache.extend_slots_batch(ids, positions) == [True] * len(ids)
        kv = task_cache.bind(ids, ws)
        expected_slots = [
            task_cache._states[request_id].pages[pos // pool.page_size] * pool.page_size
            + pos % pool.page_size
            for request_id, pos in zip(ids, positions)
        ]
        assert kv.out_cache_loc.tolist() == expected_slots
        assert kv.seq_lens.tolist() == [pos + 1 for pos in positions]
        assert len(set(expected_slots)) == len(ids)
        new_k = torch.arange(len(ids) * 8, dtype=pool.dtype, device=kv_device).reshape(
            -1, 2, 4
        ) + 1000 * (step + 1)
        kv.k_buffer[0, kv.out_cache_loc] = new_k
        kv.v_buffer[0, kv.out_cache_loc] = -new_k - 1
        for i, (request_id, pos) in enumerate(zip(ids, positions)):
            expected[i] = torch.cat([expected[i], new_k[i : i + 1]])
            row = task_cache._states[request_id].req_idx
            slots = pool.req_pool.req_to_token[row, : pos + 1]
            torch.testing.assert_close(kv.k_buffer[0, slots], expected[i])
            torch.testing.assert_close(kv.v_buffer[0, slots], -expected[i] - 1)
