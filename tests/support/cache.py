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


@pytest.fixture(params=["cpu"])
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
