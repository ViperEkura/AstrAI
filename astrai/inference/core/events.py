"""Compatibility imports for inference output events."""

from astrai.inference.contracts import (
    FINISH_ABORTED,
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_REJECTED,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)

__all__ = [
    "FINISH_ABORTED",
    "FINISH_CANCELLED",
    "FINISH_LENGTH",
    "FINISH_REJECTED",
    "FINISH_STOP_TOKEN",
    "RequestError",
    "RequestFinished",
    "TokenDelta",
]
