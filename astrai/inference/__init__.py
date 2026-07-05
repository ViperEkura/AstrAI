"""Inference module for continuous batching.

Layers:
  - core/:        Core inference loop (cache, executor, scheduler, task)
  - api/:         HTTP orchestration (ProtocolHandler, server)
  - protocols/:   Response builders (OpenAI, Anthropic)
  - transport/:   SSE transport utilities
  - engine.py:    Facade (InferenceEngine), Value Object (GenerationRequest)
  - sample.py:    Strategy pattern (TemperatureStrategy, TopKStrategy, TopPStrategy)
"""

from astrai.inference.api import (
    AnthropicMessage,
    BaseToolParser,
    ChatCompletionRequest,
    ChatMessage,
    FunctionDef,
    GenContext,
    MessagesRequest,
    ProtocolHandler,
    SimpleJsonToolParser,
    StopChecker,
    ToolDef,
    ToolParserFactory,
    get_app,
    run_server,
)
from astrai.inference.api.anthropic import AnthropicResponseBuilder
from astrai.inference.api.openai import OpenAIResponseBuilder
from astrai.inference.core import (
    STOP,
    Allocator,
    CacheView,
    ContiguousCache,
    ContiguousCacheView,
    Executor,
    InferenceScheduler,
    KVCache,
    PageCache,
    PageCacheView,
    PagePool,
    PrefixCache,
    Storage,
    Task,
    TaskManager,
    TaskStatus,
    TaskTable,
    page_hash,
)
from astrai.inference.engine import GenerationRequest, InferenceEngine
from astrai.inference.sample import (
    BaseSamplingStrategy,
    SamplingPipeline,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
    sample,
)

__all__ = [
    "InferenceEngine",
    "GenerationRequest",
    "InferenceScheduler",
    "Executor",
    "STOP",
    "Task",
    "TaskManager",
    "TaskStatus",
    "Allocator",
    "CacheView",
    "KVCache",
    "ContiguousCache",
    "ContiguousCacheView",
    "PageCache",
    "PageCacheView",
    "PagePool",
    "PrefixCache",
    "Storage",
    "TaskTable",
    "page_hash",
    "sample",
    "BaseSamplingStrategy",
    "TemperatureStrategy",
    "TopKStrategy",
    "TopPStrategy",
    "SamplingPipeline",
    "ProtocolHandler",
    "StopChecker",
    "GenContext",
    "BaseToolParser",
    "SimpleJsonToolParser",
    "ToolParserFactory",
    "OpenAIResponseBuilder",
    "AnthropicResponseBuilder",
    "ChatMessage",
    "ChatCompletionRequest",
    "FunctionDef",
    "ToolDef",
    "AnthropicMessage",
    "MessagesRequest",
    "get_app",
    "run_server",
]
