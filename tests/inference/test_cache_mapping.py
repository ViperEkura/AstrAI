import pytest
import torch

from astrai.inference.core.cache import (
    BlockPool,
    KVCacheManager,
    RequestCacheState,
    RequestPageTable,
)
from tests.support.cache import (
    appended_slots,
    make_cache_workspace,
    make_contiguous_pool,
    make_paged_pool,
    make_paged_pool_ps64,
    physical_slots,
)
from tests.support.cache import kv_device as kv_device


def test_request_page_table_alloc_free():
    pool = RequestPageTable(4, 128, torch.device("cpu"))
    assert pool.page_table.dtype == torch.int32
    slots = pool.alloc(2)
    assert len(slots) == 2
    assert len(pool.free_slots) == 2
    pool.free(slots)
    assert len(pool.free_slots) == 4


def test_request_page_table_alloc_when_full():
    pool = RequestPageTable(2, 128, torch.device("cpu"))
    pool.alloc(2)
    assert pool.alloc(1) is None


def test_request_page_table_publishes_dirty_pages_once():
    pool = RequestPageTable(4, 128, torch.device("cpu"))
    slots = pool.alloc(1)
    state = RequestCacheState(req_idx=slots[0], pages=[10, 20, 30])
    pool.flush_pages([state])
    pool.flush_pages([state])
    assert pool.version == 1
    assert pool.page_table[slots[0], 0].item() == 10
    assert pool.page_table[slots[0], 2].item() == 30


def test_page_pool_contiguous_task_alloc_free():
    pool = make_contiguous_pool()
    task_cache = KVCacheManager(pool)
    assert task_cache.admit("t1", [1, 2, 3])
    assert "t1" in task_cache._states
    task_cache.release("t1")
    assert "t1" not in task_cache._states


def test_page_pool_contiguous_task_extend():
    pool = make_contiguous_pool()
    task_cache = KVCacheManager(pool)
    task_cache.admit("t1", [1, 2, 3])
    assert task_cache.reserve_decode(["t1"], [3])[0]
    assert task_cache.reserve_decode(["t1"], [63])[0]
    assert not task_cache.reserve_decode(["t1"], [64])[0]


def test_page_pool_contiguous_bind_prefill():
    pool = make_contiguous_pool()
    task_cache = KVCacheManager(pool)
    task_cache.admit("t1", list(range(10)))
    task_cache.admit("t2", list(range(10)))
    kv = task_cache.bind(["t1", "t2"], make_cache_workspace(pool), start_pos=0)
    assert appended_slots(kv).shape == (20,)
    assert appended_slots(kv).dtype == torch.int32
    assert kv.cache_seqlens.tolist() == [0, 0]
    assert kv.page_table.size(0) == 2
    assert kv.page_table.dtype == torch.int32


def test_page_pool_bind_binds_packed_query_offsets():
    pool = make_contiguous_pool(max_batch_size=3, max_seq_len=256)

    manager = KVCacheManager(pool)
    ids = ["a", "b", "c"]
    for request_id, length in zip(ids, [70, 10, 130]):
        assert manager.admit(request_id, list(range(length)))
    kv = manager.bind(ids, make_cache_workspace(pool), start_pos=0)

    assert kv.cu_seqlens_q.tolist() == [0, 70, 80, 210]


def test_page_pool_contiguous_bind_decode():
    pool = make_contiguous_pool()
    task_cache = KVCacheManager(pool)
    task_cache.admit("t1", list(range(10)))
    task_cache.admit("t2", list(range(8)))
    # Simulate one decode extension so seq_lens advance to 11 and 9.
    assert task_cache.reserve_decode(["t1"], [10])[0]
    assert task_cache.reserve_decode(["t2"], [8])[0]
    kv = task_cache.bind(["t1", "t2"], make_cache_workspace(pool))
    assert appended_slots(kv).shape == (2,)
    assert kv.cache_seqlens.tolist() == [10, 8]


def test_page_pool_contiguous_bind_roundtrip():
    """Write and gather KV through the request page table."""
    pool = make_contiguous_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = KVCacheManager(pool)
    task_cache.admit("t1", list(range(4)))

    kv = task_cache.bind(["t1"], make_cache_workspace(pool), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0].flatten(0, 1)[appended_slots(kv)] = k
    kv.v_buffer[0].flatten(0, 1)[appended_slots(kv)] = v

    indices = physical_slots(kv.page_table[0], pool.page_size, 0, 4)[None]
    gathered_k = kv.k_buffer[0].flatten(0, 1)[indices]
    gathered_v = kv.v_buffer[0].flatten(0, 1)[indices]
    assert torch.allclose(gathered_k, k)
    assert torch.allclose(gathered_v, v)


def test_page_pool_paged_task_alloc():
    pool = make_paged_pool()
    task_cache = KVCacheManager(pool)
    assert task_cache.admit("t1", list(range(10)))
    state = task_cache._states["t1"]
    assert len(state.pages) == 10
    assert pool.req_pool.page_table[state.req_idx, 0].item() == state.pages[0]


def test_page_pool_paged_task_extend():
    pool = make_paged_pool()
    task_cache = KVCacheManager(pool)
    task_cache.admit("t1", list(range(4)))
    assert task_cache.reserve_decode(["t1"], [4])[0]
    req_idx = task_cache._states["t1"].req_idx
    task_cache.bind(["t1"], make_cache_workspace(pool))
    assert (
        pool.req_pool.page_table[req_idx, 4].item() == task_cache._states["t1"].pages[4]
    )


def test_page_pool_paged_task_free_releases_slots():
    pool = make_paged_pool(n_tokens=16)
    task_cache = KVCacheManager(pool)
    task_cache.admit("t1", list(range(8)))
    task_cache.release("t1")
    assert "t1" not in task_cache._states
    assert len(pool.req_pool.free_slots) == 4


def test_page_pool_paged_bind_roundtrip():
    pool = make_paged_pool(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = KVCacheManager(pool)
    task_cache.admit("t1", list(range(4)))

    kv = task_cache.bind(["t1"], make_cache_workspace(pool), start_pos=0)
    k = torch.randn(1, 4, 2, 4)
    v = torch.randn(1, 4, 2, 4)
    kv.k_buffer[0].flatten(0, 1)[appended_slots(kv)] = k
    kv.v_buffer[0].flatten(0, 1)[appended_slots(kv)] = v

    indices = physical_slots(kv.page_table[0], pool.page_size, 0, 4)[None]
    gathered_k = kv.k_buffer[0].flatten(0, 1)[indices]
    assert torch.allclose(gathered_k, k)


def test_page_pool_paged_ps64_task_alloc():
    pool = make_paged_pool_ps64()
    task_cache = KVCacheManager(pool)
    prompt = list(range(200))
    assert task_cache.admit("t1", prompt)
    assert task_cache.cached_tokens("t1") == 0
    n_pages = (200 + 63) // 64
    assert len(task_cache._states["t1"].pages) == n_pages


def test_page_pool_paged_ps64_task_extend_crosses_page():
    pool = make_paged_pool_ps64()
    task_cache = KVCacheManager(pool)
    task_cache.admit("t1", list(range(64)))
    assert task_cache.reserve_decode(["t1"], [64])[0]
    assert len(task_cache._states["t1"].pages) >= 2


def test_page_pool_paged_ps64_bind_roundtrip():
    pool = make_paged_pool_ps64(n_layers=1, n_kv_heads=2, head_dim=4)
    task_cache = KVCacheManager(pool)
    prompt = list(range(128))
    task_cache.admit("t1", prompt)

    kv = task_cache.bind(["t1"], make_cache_workspace(pool), start_pos=0)
    k = torch.randn(1, 128, 2, 4)
    v = torch.randn(1, 128, 2, 4)
    kv.k_buffer[0].flatten(0, 1)[appended_slots(kv)] = k
    kv.v_buffer[0].flatten(0, 1)[appended_slots(kv)] = v

    indices = physical_slots(kv.page_table[0], pool.page_size, 0, 128)[None]
    gathered_k = kv.k_buffer[0].flatten(0, 1)[indices]
    assert torch.allclose(gathered_k, k)


def test_page_pool_paged_steady_decode_slots_reach_device():
    """The extend fast path stages slot ids on the host; every bind (the
    steady incremental decode path included) gathers req_to_token rows
    on-device, so the staged tails must land there before the gather."""
    pool = make_paged_pool(n_tokens=64, max_seq_len=16)
    task_cache = KVCacheManager(pool)
    ws = make_cache_workspace(pool)
    prompt = list(range(4))
    assert task_cache.admit("t1", prompt)

    # Prefill bind flushes the whole staged prefix.
    task_cache.bind(["t1"], ws, start_pos=0)
    state = task_cache._states["t1"]

    # Decode steps: extend stages one slot per token, bind (incremental or
    # not) must push it to the device row.
    for pos in range(4, 10):
        assert task_cache.reserve_decode(["t1"], [pos])[0]
        task_cache.bind(["t1"], ws)
        device_row = pool.req_pool.page_table[state.req_idx, : pos + 1].tolist()
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
    mgr_a = KVCacheManager(pool_a)
    mgr_b = KVCacheManager(pool_b)
    ws = make_cache_workspace(pool_a)

    ids = [f"t{i}" for i in range(4)]
    for tid in ids:
        assert mgr_a.admit(tid, [10, 20, 30])
        assert mgr_b.admit(tid, [10, 20, 30])

    # Per-request reference: four extends at position 3.
    for tid in ids:
        assert mgr_a.reserve_decode([tid], [3])[0]
    # Batched: same positions through one strategy call.
    assert mgr_b.reserve_decode(list(ids), [3] * 4) == [True] * 4

    for tid in ids:
        sa = mgr_a._states[tid]
        sb = mgr_b._states[tid]
        assert sa.pages == sb.pages
        assert sa.length == sb.length == 4

    # Next positions stay in lockstep, and bind flushes both equally.
    assert mgr_b.reserve_decode(list(ids), [4] * 4) == [True] * 4
    for tid in ids:
        assert mgr_a.reserve_decode([tid], [4])[0]
    mgr_a.bind(ids, ws)
    mgr_b.bind(ids, make_cache_workspace(pool_b))
    for tid in ids:
        sa = mgr_a._states[tid]
        sb = mgr_b._states[tid]
        assert sa.length == sb.length == 5

    # A missing request fails alone; the rest still extend.
    assert mgr_b.reserve_decode(["ghost"] + ids[1:], [5] * 4) == [
        False,
        True,
        True,
        True,
    ]


@pytest.mark.parametrize("prompt_lens", [(5,), (5, 1, 2, 6), (4, 5, 8, 3)])
def test_paged_extend_batch_preserves_real_kv_across_mixed_positions(
    kv_device, prompt_lens
):
    pool = make_paged_pool(page_size=4, max_seq_len=16, n_tokens=64, device=kv_device)
    task_cache = KVCacheManager(pool)
    ws = make_cache_workspace(pool)
    ids = [f"request{i}" for i in range(len(prompt_lens))]
    for request_id, length in zip(ids, prompt_lens):
        assert task_cache.admit(request_id, list(range(length)))
        state = task_cache._states[request_id]
        # Unmapped positions must never be mistaken for physical slot zero.
        pool.req_pool.page_table[state.req_idx, len(state.pages) :].fill_(-1)

    pool._storage.k_buffer.fill_(float("nan"))
    pool._storage.v_buffer.fill_(float("nan"))
    initial = torch.arange(
        sum(prompt_lens) * 8, dtype=pool.dtype, device=kv_device
    ).reshape(-1, 2, 4)
    kv = task_cache.bind(ids, ws, start_pos=0)
    kv.k_buffer[0].flatten(0, 1)[appended_slots(kv)] = initial
    kv.v_buffer[0].flatten(0, 1)[appended_slots(kv)] = -initial - 1
    expected = list(initial.split(prompt_lens))

    # Includes all-in-page steps, page crossings and mixed in-page/crossing
    # requests. Single prompt=5 must map position 5 to slot 5, not slot 0.
    for step in range(5):
        positions = [length + step for length in prompt_lens]
        assert task_cache.reserve_decode(ids, positions) == [True] * len(ids)
        kv = task_cache.bind(ids, ws)
        expected_slots = [
            task_cache._states[request_id].pages[pos // pool.page_size] * pool.page_size
            + pos % pool.page_size
            for request_id, pos in zip(ids, positions)
        ]
        assert appended_slots(kv).tolist() == expected_slots
        assert kv.cache_seqlens.tolist() == positions
        assert len(set(expected_slots)) == len(ids)
        new_k = torch.arange(len(ids) * 8, dtype=pool.dtype, device=kv_device).reshape(
            -1, 2, 4
        ) + 1000 * (step + 1)
        kv.k_buffer[0].flatten(0, 1)[appended_slots(kv)] = new_k
        kv.v_buffer[0].flatten(0, 1)[appended_slots(kv)] = -new_k - 1
        for i, (request_id, pos) in enumerate(zip(ids, positions)):
            expected[i] = torch.cat([expected[i], new_k[i : i + 1]])
            row = task_cache._states[request_id].req_idx
            slots = physical_slots(
                pool.req_pool.page_table[row], pool.page_size, 0, pos + 1
            )
            torch.testing.assert_close(kv.k_buffer[0].flatten(0, 1)[slots], expected[i])
            torch.testing.assert_close(
                kv.v_buffer[0].flatten(0, 1)[slots], -expected[i] - 1
            )


def test_paged_extend_batch_fallback_still_maps_in_page_positions():
    pool = make_paged_pool(page_size=4, n_tokens=28, max_seq_len=16)
    task_cache = KVCacheManager(pool)
    lengths = [5, 4, 4, 4]
    ids = [f"request{i}" for i in range(len(lengths))]
    for request_id, length in zip(ids, lengths):
        assert task_cache.admit(request_id, list(range(length)))
        state = task_cache._states[request_id]
        pool.req_pool.page_table[state.req_idx, len(state.pages) :].fill_(-1)

    # Three page crossings but only two pages remain; the first request
    # needs no new page and must still have its intra-page write mapped.
    assert task_cache.reserve_decode(ids, lengths) == [True, True, True, False]
    kv = task_cache.bind(ids[:3], make_cache_workspace(pool))
    assert appended_slots(kv).tolist() == [5, 20, 24]
    assert [task_cache._states[rid].length for rid in ids] == [6, 5, 5, 4]
    last = task_cache._states[ids[-1]]
    assert pool.req_pool.page_table[last.req_idx, 1].item() == -1


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
    mgr = KVCacheManager(pool)
    ids = [f"dry{i}" for i in range(4)]
    for tid in ids:
        # One page each leaves 2 free for decode extension.
        assert mgr.admit(tid, [7])
    results = mgr.reserve_decode(list(ids), [1] * 4)
    assert results == [True, True, False, False]
    for tid, ok in zip(ids, results):
        assert (mgr._states[tid].length == 2) == ok


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_decode_reuses_pages_and_layer_views_until_mapping_changes(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pool = BlockPool(
        2, 1, 32, 2, 16, torch.device(device), torch.float32, page_size=4, n_tokens=32
    )
    manager = KVCacheManager(pool)
    workspace = make_cache_workspace(pool)
    assert manager.admit("a", [1, 2])
    assert manager.admit("b", [8, 9])
    gathers = []
    index_select = torch.index_select

    def gather(*args, **kwargs):
        gathers.append(1)
        return index_select(*args, **kwargs)

    monkeypatch.setattr(torch, "index_select", gather)
    first = manager.bind(["a", "b"], workspace)
    address = first.page_table.data_ptr()
    for pos in (2, 3):
        assert all(manager.reserve_decode(["a", "b"], [pos, pos]))
        bound = manager.bind(["a", "b"], workspace)
        assert bound is first
        assert bound.cache_seqlens.tolist() == [pos, pos]
        assert bound.page_table.data_ptr() == address
    assert len(gathers) == 1
    with torch.no_grad():
        assert bound.layer(1)[0] is first.layer(1)[0]
        bound.layer(1)[0].fill_(3)
    torch.testing.assert_close(
        pool._storage.k_buffer[1], torch.full_like(pool._storage.k_buffer[1], 3)
    )
    # Crossing a physical page boundary refreshes mappings, preserving addresses.
    assert all(manager.reserve_decode(["a", "b"], [4, 4]))
    crossed = manager.bind(["a", "b"], workspace)
    assert len(gathers) == 2
    assert crossed.page_table.data_ptr() == address
    for i, key in enumerate(("a", "b")):
        assert crossed.page_table[i, :2].tolist() == manager._states[key].pages
    # Reordering requests must refresh even if all their pages already exist.
    reordered = manager.bind(["b", "a"], workspace)
    assert len(gathers) == 3
    assert reordered.page_table[0, :2].tolist() == manager._states["b"].pages


def test_decode_uses_planned_position_across_two_workspaces():
    pool = make_paged_pool()
    manager = KVCacheManager(pool)
    a, b = make_cache_workspace(pool), make_cache_workspace(pool)
    assert manager.admit("old", [1, 2, 3])
    original = manager.bind(["old"], a, positions=[1])
    assert original.cache_seqlens.tolist() == [1]
    assert manager.reserve_decode(["old"], [3])[0]
    manager.bind(["old"], b, positions=[3])
    assert (
        manager.bind(["old"], a, positions=[2]).page_table[0, 3]
        == manager._states["old"].pages[3]
    )


def test_reserve_decode_across_multiple_pages_is_atomic_under_pressure():
    pool = make_paged_pool(page_size=2, max_seq_len=16, n_tokens=8)
    manager = KVCacheManager(pool)
    assert manager.admit("a", [1]) and manager.admit("b", [2])
    assert manager.reserve_decode(["a", "b"], [7, 5]) == [False, True]
    assert len(manager._states["a"].pages) == 1
    assert len(manager._states["b"].pages) == 3
    with pytest.raises(ValueError, match="duplicate"):
        manager.admit("b", [3])
    assert manager.request_count == 2
