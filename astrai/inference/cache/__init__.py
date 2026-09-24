"""KV cache subsystem: buffers, strategies, pool management.

The physical buffers (``KVCache`` / ``KVStorage`` / ``ReqToTokenPool``) live on
the model side in :mod:`astrai.model.kv_cache` — the attention layer consumes
them directly — and are re-exported here so existing imports keep working.
"""

from astrai.inference.cache.pool import PagePool, TaskCacheManager, page_hash
from astrai.inference.cache.strategy import (
    AllocationStrategy,
    Allocator,
    ContiguousStrategy,
    PagedStrategy,
    RadixCache,
    TaskCacheState,
)
from astrai.model.kv_cache import KVCache, KVStorage, ReqToTokenPool

__all__ = [
    "KVCache",
    "KVStorage",
    "ReqToTokenPool",
    "Allocator",
    "RadixCache",
    "TaskCacheState",
    "AllocationStrategy",
    "ContiguousStrategy",
    "PagedStrategy",
    "PagePool",
    "TaskCacheManager",
    "page_hash",
]
