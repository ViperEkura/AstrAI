"""Orchestration layer: ProtocolHandler, GenContext, StopInfo, ResponseBuilder, SSE utils.

ProtocolHandler orchestrates the async generation loop and delegates
protocol-specific formatting to a ResponseBuilder.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from astrai.inference.frontend.engine import InferenceEngine


def sse_event(data: Dict[str, Any], event: Optional[str] = None) -> str:
    lines: List[str] = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    # The SSE spec dispatches an event only at a blank line, so the frame
    # must end with "\n\n" (a single trailing newline keeps clients waiting
    # and concatenates consecutive events into one corrupt payload).
    return "\n".join(lines) + "\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


@dataclass
class GenContext:
    """Per-generation metadata passed to builder format methods."""

    resp_id: str
    created: int
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StopInfo:
    """Terminal metadata; body has already been trimmed by OutputProcessor."""

    matched: Optional[str] = None
    finish_reason: Optional[str] = None


class ResponseBuilder(ABC):
    """Interface for protocol-specific response formatting.

    A new protocol requires one concrete builder implementing 5 methods.
    """

    @abstractmethod
    def prepare(
        self, request: BaseModel, engine: InferenceEngine
    ) -> Tuple[str, GenContext, List[str]]:
        """Return (prompt, ctx, stop_sequences) for a generation request."""

    @abstractmethod
    def format_stream_start(self, ctx: GenContext) -> List[str]:
        """SSE events that open the stream."""

    @abstractmethod
    def format_chunk(self, token: str, **kwargs) -> List[str]:
        """SSE events for a single generated token.

        ``delta_token_ids`` contains new token IDs
        since the previous chunk, including tokens that decoded to no text.
        Returns a list of SSE event strings (may be empty).
        """

    @abstractmethod
    def format_stream_end(self, ctx: GenContext, stop: StopInfo) -> List[str]:
        """SSE events that close the stream."""

    @abstractmethod
    def format_response(
        self, ctx: GenContext, content: str, stop: StopInfo
    ) -> Dict[str, Any]:
        """JSON response body for non-streaming mode."""


class ProtocolHandler:
    """Orchestrates the generation loop, delegates formatting to a builder.

    Usage::

        handler = ProtocolHandler(request, engine, OpenAIResponseBuilder())
        response = await handler.handle()
    """

    def __init__(
        self, request: BaseModel, engine: InferenceEngine, builder: ResponseBuilder
    ):
        self.request = request
        self.engine = engine
        self.builder = builder

    async def handle(self) -> Union[StreamingResponse, Dict[str, Any]]:
        prompt, ctx, stop_sequences = self.builder.prepare(self.request, self.engine)
        ctx.prompt_tokens = len(self.engine.tokenizer.encode(prompt))

        agen = self.engine.generate_events(
            prompt,
            max_tokens=self.request.max_tokens,
            temperature=self.request.temperature,
            top_p=self.request.top_p,
            top_k=self.request.top_k,
            frequency_penalty=getattr(self.request, "frequency_penalty", 0.0),
            stop_sequences=stop_sequences,
        )

        if self.request.stream:
            return self._handle_stream(agen, ctx)
        else:
            return await self._handle_non_stream(agen, ctx)

    def _handle_stream(
        self, agen: AsyncGenerator, ctx: GenContext
    ) -> StreamingResponse:
        async def event_stream():
            for event in self.builder.format_stream_start(ctx):
                yield event

            final = None
            try:
                async for chunk in agen:
                    ctx.completion_tokens += len(chunk.delta_token_ids)
                    if chunk.text or chunk.delta_token_ids:
                        for event in self.builder.format_chunk(
                            chunk.text,
                            delta_token_ids=chunk.delta_token_ids,
                        ):
                            yield event
                    if chunk.stopped or chunk.is_final:
                        final = chunk
                        break
            finally:
                await agen.aclose()

            # Terminal facts come from the chunk's folded state (usage was
            # counted from token ids, stop was matched incrementally).
            if final is not None:
                stop = StopInfo(
                    matched=final.stop_sequence,
                    finish_reason=final.finish_reason,
                )
                ctx.prompt_tokens = final.prompt_tokens
                ctx.completion_tokens = final.completion_tokens
            else:
                stop = StopInfo()
            for event in self.builder.format_stream_end(ctx, stop):
                yield event
            yield sse_done()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _handle_non_stream(
        self, agen: AsyncGenerator, ctx: GenContext
    ) -> Dict[str, Any]:
        parts = []
        final = None

        try:
            async for chunk in agen:
                parts.append(chunk.text)
                ctx.completion_tokens += len(chunk.delta_token_ids)
                if chunk.stopped or chunk.is_final:
                    final = chunk
                    break
        finally:
            await agen.aclose()

        stop = (
            StopInfo(
                matched=final.stop_sequence,
                finish_reason=final.finish_reason,
            )
            if final is not None
            else StopInfo()
        )
        if final is not None:
            ctx.prompt_tokens = final.prompt_tokens
            ctx.completion_tokens = final.completion_tokens
        return self.builder.format_response(ctx, "".join(parts), stop)
