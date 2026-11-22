"""KV cache subsystem: buffers, strategies, pool management.

The physical buffers (``KVCache`` / ``KVStorage`` / ``RequestPageTable``) live on
the model side in :mod:`astrai.model.kv_cache` — the attention layer consumes
them directly — and are re-exported here so existing imports keep working.
"""

from astrai.inference.core.cache.pool import BlockPool, KVCacheManager
from astrai.inference.core.cache.strategy import (
    AllocationStrategy,
    Allocator,
    ContiguousStrategy,
    PagedStrategy,
    RadixCache,
    RequestCacheState,
)
from astrai.model.kv_cache import KVCache, KVStorage, RequestPageTable

__all__ = [
    "AllocationStrategy",
    "Allocator",
    "BlockPool",
    "ContiguousStrategy",
    "KVCache",
    "KVCacheManager",
    "KVStorage",
    "PagedStrategy",
    "RadixCache",
    "RequestPageTable",
    "RequestCacheState",
]
