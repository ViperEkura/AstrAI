"""Anthropic message completion response builder."""

import time
import uuid
from typing import Any, Dict, List, Tuple, Union

from pydantic import BaseModel

from astrai.inference.api.protocol import (
    GenContext,
    ResponseBuilder,
    StopInfo,
    sse_event,
)
from astrai.inference.engine import InferenceEngine


def _extract_text(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return ""


class AnthropicResponseBuilder(ResponseBuilder):
    def prepare(
        self, request: BaseModel, engine: InferenceEngine
    ) -> Tuple[str, GenContext, List[str]]:
        messages: List[Dict[str, str]] = []
        system = getattr(request, "system", None)
        if system:
            messages.append({"role": "system", "content": system})
        for m in request.messages:
            text = _extract_text(m.content)
            if text:
                messages.append({"role": m.role, "content": text})
        prompt = engine.tokenizer.apply_chat_template(messages, tokenize=False)
        ctx = GenContext(
            resp_id=f"msg_{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=request.model,
            prompt_tokens=0,
        )
        stop_sequences = getattr(request, "stop_sequences", None) or []
        return prompt, ctx, stop_sequences

    def format_stream_start(self, ctx: GenContext) -> List[str]:
        return [
            sse_event(
                {
                    "type": "message_start",
                    "message": {
                        "id": ctx.resp_id,
                        "type": "message",
                        "role": "assistant",
                        "model": ctx.model,
                        "content": [],
                        "usage": {"input_tokens": ctx.prompt_tokens},
                    },
                },
                event="message_start",
            ),
            sse_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                event="content_block_start",
            ),
        ]

    def format_chunk(self, token: str, **kwargs) -> List[str]:
        return [
            sse_event(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": token},
                },
                event="content_block_delta",
            )
        ]

    def format_stream_end(self, ctx: GenContext, stop: StopInfo) -> List[str]:
        events: List[str] = []
        if stop.matched:
            trimmed = stop.body[: stop.body.rfind(stop.matched)]
            unyielded = trimmed[len(stop.yielded) :]
            if unyielded:
                events.append(
                    sse_event(
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": unyielded},
                        },
                        event="content_block_delta",
                    )
                )
        events.append(
            sse_event(
                {"type": "content_block_stop", "index": 0},
                event="content_block_stop",
            )
        )
        events.append(
            sse_event(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "stop_sequence" if stop.matched else "end_turn",
                        "stop_sequence": stop.matched,
                    },
                    "usage": {"output_tokens": ctx.completion_tokens},
                },
                event="message_delta",
            )
        )
        events.append(sse_event({"type": "message_stop"}, event="message_stop"))
        return events

    def format_response(
        self, ctx: GenContext, content: str, stop: StopInfo
    ) -> Dict[str, Any]:
        if stop.matched:
            content = content[: content.rfind(stop.matched)]
        return {
            "id": ctx.resp_id,
            "type": "message",
            "role": "assistant",
            "model": ctx.model,
            "content": [{"type": "text", "text": content}],
            "stop_reason": "stop_sequence" if stop.matched else "end_turn",
            "stop_sequence": stop.matched,
            "usage": {
                "input_tokens": ctx.prompt_tokens,
                "output_tokens": ctx.completion_tokens,
            },
        }
