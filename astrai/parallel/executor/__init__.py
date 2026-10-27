"""Training executors: explicit gradient synchronization and parallel wrapping.

Importing this package registers every executor (``none``, ``ddp``, ``fsdp``)
with :class:`ExecutorFactory`.
"""

from astrai.parallel.executor.base import (
    BaseExecutor,
    ExecutorFactory,
    NoneExecutor,
    RolloutCapabilities,
    broadcast_state_dict,
    strip_compile_prefix,
)
from astrai.parallel.executor.ddp import DDPExecutor
from astrai.parallel.executor.fsdp import FSDPExecutor

__all__ = [
    "BaseExecutor",
    "DDPExecutor",
    "ExecutorFactory",
    "FSDPExecutor",
    "NoneExecutor",
    "RolloutCapabilities",
    "broadcast_state_dict",
    "strip_compile_prefix",
]
