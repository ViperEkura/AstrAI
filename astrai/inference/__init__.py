"""Inference module for continuous batching.

Subpackages:
  - cache/:     KV cache (buffers, strategies, pool)
  - runtime/:   Execution + sampling (executor, CUDA graph, sampling strategies)
  - task/:      Request lifecycle + performance metrics
  - network/:   HTTP protocol handling (server, protocol, OpenAI/Anthropic builders)

Modules:
  - scheduler.py:  Continuous batching loop
  - workspace.py:  Pre-allocated GPU buffers
  - engine.py:     Facade (InferenceEngine)
"""

from astrai.inference.engine import InferenceEngine, build_engine
from astrai.inference.network import get_app, run_server
from astrai.inference.runtime.executor import Executor
from astrai.inference.runtime.sample import sample
from astrai.inference.scheduler import InferenceScheduler
from astrai.inference.task import (
    STOP,
    BatchedStreamCallback,
    GenerationResult,
    Task,
    TaskManager,
    TaskStatus,
)

__all__ = [
    "InferenceEngine",
    "build_engine",
    "InferenceScheduler",
    "BatchedStreamCallback",
    "GenerationResult",
    "Executor",
    "STOP",
    "Task",
    "TaskManager",
    "TaskStatus",
    "sample",
    "get_app",
    "run_server",
]
