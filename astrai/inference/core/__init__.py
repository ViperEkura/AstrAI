"""Core layer (vLLM: EngineCore process — scheduler + KV accounting).

Single-writer request lifecycle: waiting/running queues, the busy loop,
step scheduling, KV cache management and the policy-version protocol for
weights shared with the trainer.  The core emits output events; it never
detokenizes and never runs consumer callbacks.
"""

from astrai.inference.core.engine_core import EngineCore
from astrai.inference.core.planner import StepPlanner
from astrai.inference.core.request import (
    GenerationResult,
    Request,
    RequestStatus,
    StreamDecoder,
)
from astrai.inference.core.scheduler import OutputEventSink, Scheduler

__all__ = [
    "EngineCore",
    "GenerationResult",
    "OutputEventSink",
    "Request",
    "RequestStatus",
    "Scheduler",
    "StepPlanner",
    "StreamDecoder",
]
