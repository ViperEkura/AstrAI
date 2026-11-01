"""Inference module for continuous batching, in three layers.

Layering (mirrors vLLM v1's process topology, in-process):

- ``frontend/``  user-facing: engine facade, input/output processors,
                 output events, direct engine-core calls   (vLLM: API proc)
- ``core/``      engine core: scheduler, request lifecycle, stepper,
                 KV cache accounting, policy versioning   (vLLM: EngineCore)
- ``worker/``    model execution: model runner, pending steps, CUDA
                 graphs, sampler, workspace                (vLLM: Worker)
- ``network/``   HTTP protocol adapters (OpenAI/Anthropic) — part of the
                 frontend deployment, kept as its own package.

Dependency rule (one-way): frontend → core → worker → model/KV ABI.
"""

from astrai.inference.contracts import ModelRunnerOutput, SchedulerOutput
from astrai.inference.core.engine_core import EngineCore
from astrai.inference.core.events import (
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.core.request import (
    GenerationResult,
    Request,
    RequestStatus,
)
from astrai.inference.core.scheduler import Scheduler
from astrai.inference.frontend.engine import InferenceEngine, build_engine
from astrai.inference.frontend.input_processor import InputProcessor
from astrai.inference.frontend.output_processor import OutputProcessor
from astrai.inference.network import get_app, run_server
from astrai.inference.worker.model_runner import GPUModelRunner
from astrai.inference.worker.sample import sample

__all__ = [
    "EngineCore",
    "GPUModelRunner",
    "GenerationResult",
    "InferenceEngine",
    "InputProcessor",
    "ModelRunnerOutput",
    "OutputProcessor",
    "Request",
    "RequestError",
    "RequestFinished",
    "RequestStatus",
    "Scheduler",
    "SchedulerOutput",
    "TokenDelta",
    "build_engine",
    "get_app",
    "run_server",
    "sample",
]
