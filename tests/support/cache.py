import pytest
import torch

from astrai.inference.core.cache import (
    BlockPool,
)
from astrai.inference.worker.workspace import InferenceWorkspace


def make_cache_workspace(pool: BlockPool) -> InferenceWorkspace:
    """Workspace sized to the pool (bind_tasks requires it)."""
    return InferenceWorkspace(
        pool.max_batch_size,
        pool.max_seq_len,
        max_q_heads=2,
        head_dim=4,
        device=pool.device,
        dtype=pool.dtype,
    )


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def kv_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param)


def assert_prefix_reachable(prefix):
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


def assert_allocator_cache_consistent(alloc, prefix):
    assert_prefix_reachable(prefix)
    for page in range(alloc._n_pages):
        cached = prefix.has_page(page)
        free = bool(alloc._free_mask & (1 << page))
        assert free == (alloc._refs[page] == 0 and not cached)
        assert (page in alloc._lru) == (alloc._refs[page] == 0 and cached)
    assert alloc._free_mask == sum(
        word << (64 * i) for i, word in enumerate(alloc._words)
    )


def make_contiguous_pool(**kwargs):
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


def make_paged_pool(**kwargs):
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


def make_paged_pool_ps64(**kwargs):
    return make_paged_pool(
        **{"max_seq_len": 256, "page_size": 64, "n_tokens": 512, **kwargs}
    )


def physical_slots(row, page_size, start, end):
    positions = torch.arange(start, end, device=row.device, dtype=torch.int32)
    return row[positions // page_size] * page_size + positions % page_size


def appended_slots(cache):
    size = cache.k_buffer.size(2)
    lengths = (
        cache.cu_seqlens_q.diff().tolist()
        if hasattr(cache, "cu_seqlens_q")
        else [1] * cache.page_table.size(0)
    )
    return torch.cat(
        [
            physical_slots(row, size, past, past + length)
            for row, past, length in zip(
                cache.page_table, cache.cache_seqlens.tolist(), lengths
            )
        ]
    )
