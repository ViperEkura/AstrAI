"""Unit tests for inference cache components."""

import pytest
import torch

from astrai.inference.core.cache import (
    BlockPool,
    KVStorage,
    ReqToTokenPool,
)
from tests.support.cache import (
    _make_contiguous_pool,
    _make_paged_pool,
    _make_paged_pool_ps64,
    _make_task_cache,
    _ws,
)
from tests.support.cache import kv_device as kv_device


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


def test_page_pool_contiguous_task_alloc_free():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    assert task_cache.alloc_slots("t1", [1, 2, 3])
    assert "t1" in task_cache._states
    task_cache.free_slots("t1")
    assert "t1" not in task_cache._states


def test_page_pool_contiguous_task_extend():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", [1, 2, 3])
    assert task_cache.extend_slots("t1", 3)
    assert task_cache.extend_slots("t1", 63)
    assert not task_cache.extend_slots("t1", 64)


def test_page_pool_contiguous_task_cached():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", [1, 2, 3])
    assert task_cache.cached_tokens("t1") == 0


def test_page_pool_contiguous_bind_tasks_prefill():
    pool = _make_contiguous_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(10)))
    task_cache.alloc_slots("t2", list(range(10)))
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
    task_cache.alloc_slots("t1", list(range(10)))
    task_cache.alloc_slots("t2", list(range(8)))
    # Simulate one decode extension so seq_lens advance to 11 and 9.
    assert task_cache.extend_slots("t1", 10)
    assert task_cache.extend_slots("t2", 8)
    kv = task_cache.bind(["t1", "t2"], _ws(pool))
    assert kv.out_cache_loc.shape == (2,)
    assert kv.seq_lens.tolist() == [11, 9]


def test_page_pool_contiguous_bind_roundtrip():
    """Write KV via bind_tasks, then gather via req_to_token indexing."""
    pool = _make_contiguous_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(4)))

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


def test_page_pool_paged_task_alloc():
    pool = _make_paged_pool()
    task_cache = _make_task_cache(pool)
    assert task_cache.alloc_slots("t1", list(range(10)))
    state = task_cache._states["t1"]
    assert len(state.pages) == 10
    assert pool.req_pool.req_to_token[state.req_idx, 0].item() == state.pages[0]


def test_page_pool_paged_task_extend():
    pool = _make_paged_pool()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(4)))
    assert task_cache.extend_slots("t1", 4)
    req_idx = task_cache._states["t1"].req_idx
    slot = pool.req_pool.req_to_token[req_idx, 4].item()
    assert slot >= 0


def test_page_pool_paged_task_free_releases_slots():
    pool = _make_paged_pool(n_tokens=16)
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(8)))
    task_cache.free_slots("t1")
    assert "t1" not in task_cache._states
    assert len(pool.req_pool.free_slots) == 4


def test_page_pool_paged_bind_roundtrip():
    pool = _make_paged_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(4)))

    kv = task_cache.bind(["t1"], _ws(pool), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0, kv.out_cache_loc] = k
    kv.v_buffer[0, kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :4]
    gathered_k = kv.k_buffer[0, indices]
    assert torch.allclose(gathered_k, k)


def test_page_pool_paged_ps64_task_alloc():
    pool = _make_paged_pool_ps64()
    task_cache = _make_task_cache(pool)
    prompt = list(range(200))
    assert task_cache.alloc_slots("t1", prompt)
    assert task_cache.cached_tokens("t1") == 0
    n_pages = (200 + 63) // 64
    assert len(task_cache._states["t1"].pages) == n_pages


def test_page_pool_paged_ps64_task_extend_crosses_page():
    pool = _make_paged_pool_ps64()
    task_cache = _make_task_cache(pool)
    task_cache.alloc_slots("t1", list(range(64)))
    assert task_cache.extend_slots("t1", 64)
    assert len(task_cache._states["t1"].pages) >= 2


def test_page_pool_paged_ps64_bind_roundtrip():
    pool = _make_paged_pool_ps64(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = _make_task_cache(pool)
    prompt = list(range(128))
    task_cache.alloc_slots("t1", prompt)

    kv = task_cache.bind(["t1"], _ws(pool), start_pos=0)
    k = torch.randn(1, 128, 2, 4)
    v = torch.randn(1, 128, 2, 4)
    kv.k_buffer[0, kv.out_cache_loc] = k
    kv.v_buffer[0, kv.out_cache_loc] = v

    indices = kv.req_to_token[kv.req_pool_indices, :128]
    gathered_k = kv.k_buffer[0, indices]
    assert torch.allclose(gathered_k, k)


def test_page_pool_paged_steady_decode_slots_reach_device():
    """The extend fast path stages slot ids on the host; every bind (the
    steady incremental decode path included) gathers req_to_token rows
    on-device, so the staged tails must land there before the gather."""
    pool = _make_paged_pool(n_tokens=64, max_seq_len=16)
    task_cache = _make_task_cache(pool)
    ws = _ws(pool)
    prompt = list(range(4))
    assert task_cache.alloc_slots("t1", prompt)

    # Prefill bind flushes the whole staged prefix.
    task_cache.bind(["t1"], ws, start_pos=0)
    state = task_cache._states["t1"]

    # Decode steps: extend stages one slot per token, bind (incremental or
    # not) must push it to the device row.
    for pos in range(4, 10):
        assert task_cache.extend_slots("t1", pos)
        task_cache.bind(["t1"], ws)
        device_row = pool.req_pool.req_to_token[state.req_idx, : pos + 1].tolist()
        expected = [p * pool.page_size for p in state.pages[: pos + 1]]
        assert device_row == expected


def test_extend_batch_matches_per_task_extend():
    """Batched decode-step extension is observably identical to per-request.

    Steady decode extends every request by exactly one position; the batch
    must produce the same pages, the same slot staging and the same
    length bookkeeping as the historical per-request loop, including the
    page ORDER (both harvest lowest-first).
    """

    def make_pool():
        return BlockPool(
            n_layers=2,
            n_kv_heads=1,
            head_dim=4,
            max_batch_size=8,
            max_seq_len=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
            page_size=1,
            n_tokens=8 * 64,
        )

    pool_a, pool_b = make_pool(), make_pool()
    mgr_a = _make_task_cache(pool_a)
    mgr_b = _make_task_cache(pool_b)
    ws = _ws(pool_a)

    ids = [f"t{i}" for i in range(4)]
    for tid in ids:
        assert mgr_a.alloc_slots(tid, [10, 20, 30])
        assert mgr_b.alloc_slots(tid, [10, 20, 30])

    # Per-request reference: four extends at position 3.
    for tid in ids:
        assert mgr_a.extend_slots(tid, 3)
    # Batched: same positions through one strategy call.
    assert mgr_b.extend_slots_batch(list(ids), [3] * 4) == [True] * 4

    for tid in ids:
        sa = mgr_a._states[tid]
        sb = mgr_b._states[tid]
        assert sa.pages == sb.pages
        assert sa._slots == sb._slots
        assert sa.length == sb.length == 4

    # Next positions stay in lockstep, and bind flushes both equally.
    assert mgr_b.extend_slots_batch(list(ids), [4] * 4) == [True] * 4
    for tid in ids:
        assert mgr_a.extend_slots(tid, 4)
    mgr_a.bind(ids, ws)
    mgr_b.bind(ids, _ws(pool_b))
    for tid in ids:
        sa = mgr_a._states[tid]
        sb = mgr_b._states[tid]
        assert sa._slots == sb._slots
        assert sa.length == sb.length == 5

    # A missing request fails alone; the rest still extend.
    assert mgr_b.extend_slots_batch(["ghost"] + ids[1:], [5] * 4) == [
        False,
        True,
        True,
        True,
    ]


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


def test_paged_extend_batch_fallback_still_maps_in_page_positions():
    pool = _make_paged_pool(page_size=4, n_tokens=28, max_seq_len=16)
    task_cache = _make_task_cache(pool)
    lengths = [5, 4, 4, 4]
    ids = [f"request{i}" for i in range(len(lengths))]
    for request_id, length in zip(ids, lengths):
        assert task_cache.alloc_slots(request_id, list(range(length)))
        state = task_cache._states[request_id]
        pool.req_pool.req_to_token[state.req_idx, length:].fill_(-1)

    # Three page crossings but only two pages remain; the first request
    # needs no new page and must still have its intra-page write mapped.
    assert task_cache.extend_slots_batch(ids, lengths) == [True, True, True, False]
    kv = task_cache.bind(ids[:3], _ws(pool))
    assert kv.out_cache_loc.tolist() == [5, 20, 24]
    assert [task_cache._states[rid].length for rid in ids] == [6, 5, 5, 4]
    last = task_cache._states[ids[-1]]
    assert pool.req_pool.req_to_token[last.req_idx, 4].item() == -1


def test_extend_batch_falls_back_when_pool_runs_dry():
    """Pool exhaustion keeps per-request success ORDER via the fallback.

    The batch harvest cannot satisfy every state when the pool is
    nearly empty; the strategy then re-runs per-request extend so failure
    lands on the same requests the historical loop would have failed.
    """
    # 6 pages total: 4 consumed by prompts, 2 free.
    pool = BlockPool(
        n_layers=1,
        n_kv_heads=1,
        head_dim=4,
        max_batch_size=4,
        max_seq_len=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        page_size=1,
        n_tokens=6,
    )
    mgr = _make_task_cache(pool)
    ids = [f"dry{i}" for i in range(4)]
    for tid in ids:
        # One page each leaves 2 free for decode extension.
        assert mgr.alloc_slots(tid, [7])
    results = mgr.extend_slots_batch(list(ids), [1] * 4)
    assert results == [True, True, False, False]
    for tid, ok in zip(ids, results):
        assert (mgr._states[tid].length == 2) == ok
