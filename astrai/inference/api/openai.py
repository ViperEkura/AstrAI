"""OpenAI chat completion response builder."""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from astrai.inference.api.protocol import (
    GenContext,
    ResponseBuilder,
    StopInfo,
    sse_event,
)
from astrai.inference.api.tool_parser import BaseToolParser, ToolParserFactory
from astrai.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)

_UNSUPPORTED_PARAMS = (
    "n",
    "presence_penalty",
    "logit_bias",
    "user",
)


def _resolve_tool_choice(
    request: BaseModel,
) -> Union[str, Dict[str, Any]]:
    tc = getattr(request, "tool_choice", None)
    if tc is None:
        return "auto"
    if isinstance(tc, str):
        return tc
    if isinstance(tc, dict):
        return tc
    return "auto"


def _resolve_tools(request: BaseModel) -> Optional[List[Dict[str, Any]]]:
    raw = getattr(request, "tools", None)
    if not raw:
        return None
    if isinstance(raw, list):
        return [t.model_dump() if hasattr(t, "model_dump") else t for t in raw]
    return None


class OpenAIResponseBuilder(ResponseBuilder):
    def prepare(
        self, request: BaseModel, engine: InferenceEngine
    ) -> Tuple[str, GenContext, List[str]]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        tools = _resolve_tools(request)
        prompt = engine.tokenizer.apply_chat_template(
            messages, tokenize=False, tools=tools or []
        )

        self._resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self._model = request.model

        for param in _UNSUPPORTED_PARAMS:
            value = getattr(request, param, None)
            fields = getattr(type(request), "model_fields", {})
            default = fields[param].default if param in fields else None
            if value is not None and value != default:
                logger.warning(
                    "ChatCompletionRequest param '%s'=%r is not supported"
                    " and will be ignored",
                    param,
                    value,
                )

        self._parser: Optional[BaseToolParser] = None
        if tools:
            tool_choice = _resolve_tool_choice(request)
            self._parser = ToolParserFactory.create(
                "simple_json", tools=tools, tool_choice=tool_choice
            )
        self._content_started = False

        ctx = GenContext(
            resp_id=self._resp_id,
            created=int(time.time()),
            model=self._model,
        )
        stop = request.stop
        stop_sequences = (
            [] if stop is None else [stop] if isinstance(stop, str) else stop
        )
        return prompt, ctx, stop_sequences

    def format_stream_start(self, ctx: GenContext) -> List[str]:
        return [
            sse_event(
                {
                    "id": self._resp_id,
                    "object": "chat.completion.chunk",
                    "created": ctx.created,
                    "model": self._model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        ]

    def format_chunk(self, token: str, **kwargs) -> List[str]:
        body = kwargs.get("body", "")
        if self._parser is not None:
            return self._format_tool_chunk(body, **kwargs)

        return [
            sse_event(
                {
                    "id": self._resp_id,
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": self._model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        ]

    def _format_tool_chunk(self, body: str, **kwargs) -> List[str]:
        deltas = self._parser.feed(
            body,
            current_token_ids=kwargs.get("current_token_ids"),
            delta_token_ids=kwargs.get("delta_token_ids"),
        )
        events: List[str] = []
        for d in deltas:
            if "content" in d:
                if not self._content_started:
                    events.append(self._role_chunk())
                    self._content_started = True
                events.append(
                    sse_event(
                        {
                            "id": self._resp_id,
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": self._model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": d["content"]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                )
            elif "tool_calls" in d:
                if not self._content_started:
                    events.append(self._role_chunk())
                    self._content_started = True
                events.append(
                    sse_event(
                        {
                            "id": self._resp_id,
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": self._model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": d["tool_calls"]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                )
        return events

    def _role_chunk(self) -> str:
        return sse_event(
            {
                "id": self._resp_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self._model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
        )

    def format_stream_end(self, ctx: GenContext, stop: StopInfo) -> List[str]:
        finish_reason = "stop"
        if self._parser is not None and self._parser.has_tool_calls:
            finish_reason = "tool_calls"
        return [
            sse_event(
                {
                    "id": self._resp_id,
                    "object": "chat.completion.chunk",
                    "created": ctx.created,
                    "model": self._model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                }
            ),
            sse_event(
                {
                    "prompt_tokens": ctx.prompt_tokens,
                    "completion_tokens": ctx.completion_tokens,
                    "total_tokens": ctx.prompt_tokens + ctx.completion_tokens,
                }
            ),
        ]

    def format_response(
        self, ctx: GenContext, content: str, stop: StopInfo
    ) -> Dict[str, Any]:
        if self._parser is not None:
            parsed = self._parser.parse_complete(content)
            if parsed and parsed.get("tool_calls"):
                return {
                    "id": self._resp_id,
                    "object": "chat.completion",
                    "created": ctx.created,
                    "model": self._model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": parsed.get("content"),
                                "tool_calls": parsed["tool_calls"],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": ctx.prompt_tokens,
                        "completion_tokens": ctx.completion_tokens,
                        "total_tokens": ctx.prompt_tokens + ctx.completion_tokens,
                    },
                }

        return {
            "id": self._resp_id,
            "object": "chat.completion",
            "created": ctx.created,
            "model": self._model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": ctx.prompt_tokens,
                "completion_tokens": ctx.completion_tokens,
                "total_tokens": ctx.prompt_tokens + ctx.completion_tokens,
            },
        }
