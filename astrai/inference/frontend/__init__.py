"""Frontend layer (vLLM: API process — AsyncLLM, Input/OutputProcessor).

Everything that faces the user: the engine facade, request-id minting,
tokenization, detokenization/stop handling, the output-event contract and
direct engine-core calls.  Frontend code never touches KV state or
model tensors directly.
"""

from astrai.inference.core.events import (
    FINISH_ABORTED,
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_REJECTED,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.frontend.engine import InferenceEngine, build_engine
from astrai.inference.frontend.input_processor import InputProcessor
from astrai.inference.frontend.output_processor import (
    OutputProcessor,
    StopSequenceChecker,
)

__all__ = [
    "FINISH_ABORTED",
    "FINISH_CANCELLED",
    "FINISH_LENGTH",
    "FINISH_REJECTED",
    "FINISH_STOP_TOKEN",
    "InferenceEngine",
    "InputProcessor",
    "OutputProcessor",
    "RequestError",
    "RequestFinished",
    "StopSequenceChecker",
    "TokenDelta",
    "build_engine",
]
