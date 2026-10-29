"""Inference API: protocol handler, stop checker, tool parsers, and FastAPI server.

``app`` is no longer a module-level global. Use :func:`get_app` to access the
lazy singleton FastAPI instance.
"""

from astrai.inference.network.app import (
    AnthropicMessage,
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    MessagesRequest,
    ToolDef,
    get_app,
    run_server,
)
from astrai.inference.network.protocol import GenContext, ProtocolHandler
from astrai.inference.network.tool_parser import (
    BaseToolParser,
    SimpleJsonToolParser,
)

__all__ = [
    "AnthropicMessage",
    "BaseToolParser",
    "ChatCompletionRequest",
    "ChatMessage",
    "FunctionDef",
    "GenContext",
    "MessagesRequest",
    "ProtocolHandler",
    "SimpleJsonToolParser",
    "ToolDef",
    "get_app",
    "run_server",
]
