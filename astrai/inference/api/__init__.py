"""Inference API: protocol handler, stop checker, and FastAPI server.

``app`` is no longer a module-level global. Use :func:`get_app` to access the
lazy singleton FastAPI instance.
"""

from astrai.inference.api.protocol import GenContext, ProtocolHandler, StopChecker
from astrai.inference.api.server import (
    AnthropicMessage,
    ChatCompletionRequest,
    ChatMessage,
    MessagesRequest,
    get_app,
    run_server,
)

__all__ = [
    "ProtocolHandler",
    "StopChecker",
    "GenContext",
    "AnthropicMessage",
    "ChatCompletionRequest",
    "ChatMessage",
    "MessagesRequest",
    "get_app",
    "run_server",
]
