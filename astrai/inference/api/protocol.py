"""Orchestration layer: ProtocolHandler, StopChecker, GenContext, StopInfo, ResponseBuilder, SSE utils.

ProtocolHandler orchestrates the async generation loop and delegates
protocol-specific formatting to a ResponseBuilder.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from astrai.inference.engine import InferenceEngine


def sse_event(data: Dict[str, Any], event: Optional[str] = None) -> str:
    lines: List[str] = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    lines.append("")
    return "\n".join(lines)


def sse_done() -> str:
    return "data: [DONE]\n\n"


@dataclass
class GenContext:
    """Per-generation metadata passed to builder format methods."""

    resp_id: str
    created: int
    model: str
    prompt_tokens: int
    completion_tokens: int = 0


@dataclass
class StopInfo:
    """Stop-check result passed to format_stream_end / format_response."""

    matched: Optional[str] = None
    body: str = ""
    yielded: str = ""


class StopChecker:
    """Scans accumulated text for stop sequence matches."""

    def __init__(self, sequences: List[str]):
        self._sequences = [s for s in sequences if s]

    def check(self, text: str) -> Optional[str]:
        for seq in self._sequences:
            if seq in text:
                return seq
        return None


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

        ``body`` (the full accumulated text so far) is always provided
        as a keyword argument. Additional keyword arguments such as
        ``current_token_ids`` and ``delta_token_ids`` may be included
        for tool parsers that need token-level information.
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

        agen = self.engine.generate_async(
            prompt=prompt,
            max_tokens=self.request.max_tokens,
            temperature=self.request.temperature,
            top_p=self.request.top_p,
            top_k=self.request.top_k,
        )

        if self.request.stream:
            return self._handle_stream(agen, ctx, stop_sequences)
        else:
            return await self._handle_non_stream(agen, ctx, stop_sequences)

    def _handle_stream(
        self, agen: AsyncGenerator, ctx: GenContext, stop_sequences: List[str]
    ) -> StreamingResponse:
        checker = StopChecker(stop_sequences)

        async def event_stream():
            for event in self.builder.format_stream_start(ctx):
                yield event

            body = ""
            yielded = ""
            matched = None
            token_ids: List[int] = []
            async for token in agen:
                body += token

                new_ids = self.engine.tokenizer.encode(token)
                token_ids.extend(new_ids)

                matched = checker.check(body)
                if matched:
                    break

                ctx.completion_tokens += 1
                for event in self.builder.format_chunk(
                    token,
                    body=body,
                    current_token_ids=token_ids,
                    delta_token_ids=new_ids,
                ):
                    yield event
                yielded += token

            stop = StopInfo(matched=matched, body=body, yielded=yielded)
            for event in self.builder.format_stream_end(ctx, stop):
                yield event
            yield sse_done()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def _handle_non_stream(
        self, agen: AsyncGenerator, ctx: GenContext, stop_sequences: List[str]
    ) -> Dict[str, Any]:
        checker = StopChecker(stop_sequences)
        chunks: List[str] = []
        body = ""
        matched = None

        async for token in agen:
            chunks.append(token)
            body += token

            matched = checker.check(body)
            if matched:
                break

            ctx.completion_tokens += 1

        content = "".join(chunks)
        stop = StopInfo(matched=matched, body=body)
        return self.builder.format_response(ctx, content, stop)
