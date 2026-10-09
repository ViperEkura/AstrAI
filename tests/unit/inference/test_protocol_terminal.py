import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from astrai.inference.frontend.tracking import StreamChunk
from astrai.inference.network.anthropic import AnthropicResponseBuilder
from astrai.inference.network.app import ChatCompletionRequest, MessagesRequest
from astrai.inference.network.openai import OpenAIResponseBuilder
from astrai.inference.network.protocol import ProtocolHandler


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", ["length", "stop_token", "stop"])
def test_protocol_delivers_terminal_text_and_authoritative_usage(
    protocol, stream, reason
):
    state = {"closed": False}
    first = StreamChunk("Hello ", [1], [1], False)
    final = StreamChunk(
        "world" if reason == "stop" else "EN", [2], [1, 2], reason == "stop"
    )
    final.finish_reason = reason
    final.prompt_tokens = 3
    final.completion_tokens = 2
    final.stop_sequence = "END" if reason == "stop" else None

    async def generate_events(*args, **kwargs):
        try:
            yield first
            yield final
        finally:
            state["closed"] = True

    engine = SimpleNamespace(
        tokenizer=SimpleNamespace(
            encode=lambda prompt: list(range(20)),
            apply_chat_template=lambda *args, **kwargs: "long prompt",
        ),
        generate_events=generate_events,
    )
    options = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
        "stream": stream,
    }
    if protocol == "openai":
        request = ChatCompletionRequest(**options, stop=["END"])
        builder = OpenAIResponseBuilder()
    else:
        request = MessagesRequest(**options, stop_sequences=["END"])
        builder = AnthropicResponseBuilder()

    async def run():
        response = await ProtocolHandler(request, engine, builder).handle()
        if not stream:
            return response
        frames = [frame async for frame in response.body_iterator]
        assert frames[-1] == "data: [DONE]\n\n"
        payloads = []
        for frame in frames:
            for line in frame.splitlines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    payloads.append(json.loads(line[6:]))
        return payloads

    response = asyncio.run(run())
    assert state["closed"]
    expected_text = "Hello world" if reason == "stop" else "Hello EN"
    if protocol == "openai":
        expected_reason = "length" if reason == "length" else "stop"
        if stream:
            choices = [item["choices"][0] for item in response if "choices" in item]
            text = "".join(item["delta"].get("content", "") for item in choices)
            assert choices[-1]["finish_reason"] == expected_reason
            usage = response[-1]
        else:
            text = response["choices"][0]["message"]["content"]
            assert response["choices"][0]["finish_reason"] == expected_reason
            usage = response["usage"]
        assert usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    else:
        expected_reason = {
            "length": "max_tokens",
            "stop_token": "end_turn",
            "stop": "stop_sequence",
        }[reason]
        if stream:
            text = "".join(
                item["delta"]["text"]
                for item in response
                if item["type"] == "content_block_delta"
            )
            terminal = next(
                item for item in response if item["type"] == "message_delta"
            )
            assert terminal["delta"]["stop_reason"] == expected_reason
            usage = terminal["usage"]
        else:
            text = response["content"][0]["text"]
            assert response["stop_reason"] == expected_reason
            usage = response["usage"]
        assert usage == {"input_tokens": 3, "output_tokens": 2}
    assert text == expected_text


def test_openai_tool_parser_receives_terminal_body_once():
    builder = OpenAIResponseBuilder()
    builder._resp_id = "response"
    builder._model = "test"
    builder._content_started = False
    builder._parser = SimpleNamespace(feed=Mock(return_value=[{"content": "tail"}]))
    events = builder.format_chunk(
        "tail", body="full tail", current_token_ids=[1, 2], delta_token_ids=[2]
    )
    assert events
    builder._parser.feed.assert_called_once_with(
        "full tail", current_token_ids=[1, 2], delta_token_ids=[2]
    )
