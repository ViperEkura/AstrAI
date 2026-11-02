import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from astrai.inference.core.events import (
    FINISH_ABORTED,
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_STOP_TOKEN,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.frontend.output_processor import (
    OutputProcessor,
    StopSequenceChecker,
)
from astrai.inference.frontend.tracking import StreamChunk
from astrai.inference.network.anthropic import AnthropicResponseBuilder
from astrai.inference.network.app import ChatCompletionRequest, MessagesRequest
from astrai.inference.network.openai import OpenAIResponseBuilder
from astrai.inference.network.protocol import ProtocolHandler
from tests.support.tokenizers import FakeTokenizer


def test_stop_sequence_checker_finds_match():
    sc = StopSequenceChecker(["stop", "end"])
    text, stopped = sc.push("hello stop world")
    assert stopped and sc.matched == "stop"


def test_stop_sequence_checker_buffers_ambiguous_tail():
    sc = StopSequenceChecker(["stop"])
    text, stopped = sc.push("hello sto")
    assert not stopped and text == "hello "  # tail kept for straddle matching


def test_stop_sequence_checker_empty_sequences():
    sc = StopSequenceChecker([])
    text, stopped = sc.push("hello")
    assert not stopped and text == "hello"


def test_stop_sequence_checker_single_character_never_repeats_text():
    checker = StopSequenceChecker(["!"])
    assert checker.push("a") == ("a", False)
    assert checker.push("b") == ("b", False)
    assert checker.flush() == ""
    assert checker.push("c!ignored") == ("c", True)
    assert checker.flush() == ""


def test_stop_sequence_checker_cross_token_match_discards_only_stop_and_suffix():
    checker = StopSequenceChecker(["END"])
    assert checker.push("hello EN") == ("hello ", False)
    assert checker.push("D ignored") == ("", True)
    assert checker.push("more ignored") == ("", True)
    assert checker.flush() == ""


def test_stop_sequence_checker_flushes_unmatched_tail_once():
    checker = StopSequenceChecker(["END"])
    assert checker.push("hello EN") == ("hello ", False)
    assert checker.flush() == "EN"
    assert checker.flush() == ""


def test_stop_sequence_checker_uses_earliest_match_not_stop_list_order():
    checker = StopSequenceChecker(["later", "early"])
    assert checker.push("before early and later") == ("before ", True)
    assert checker.matched == "early"
    assert checker.flush() == ""


@pytest.mark.parametrize("reason", [FINISH_STOP_TOKEN, FINISH_LENGTH, FINISH_CANCELLED])
def test_output_processor_flushes_unmatched_tail_on_core_terminal(reason):
    processor = OutputProcessor(
        "r", FakeTokenizer(), stop_sequences=["END"], prompt_tokens=7
    )
    processor._decoder = MagicMock()
    processor._decoder.push.side_effect = ["hello ", "EN", ""]
    chunks = [processor.push(TokenDelta("r", i, i))[0] for i in range(1, 4)]
    terminal = RequestFinished("r", reason, 7, 3)
    tail, stopped = processor.push(terminal)
    assert not stopped
    assert tail == "EN"
    assert "".join(chunks) + tail == "hello EN"
    assert processor.finished
    assert processor.state.finish_reason == reason
    assert processor.state.stop_sequence is None
    assert processor.usage() == (7, 3)
    assert processor.push(terminal) == ("", False)
    assert processor.push(TokenDelta("r", 4, 4)) == ("", False)


def test_output_processor_text_stop_keeps_prompt_usage_without_core_terminal():
    processor = OutputProcessor(
        "r", FakeTokenizer(), stop_sequences=["END"], prompt_tokens=7
    )
    processor._decoder = MagicMock()
    processor._decoder.push.side_effect = ["hello E", "ND ignored"]
    first, _ = processor.push(TokenDelta("r", 11, 1))
    last, stopped = processor.push(TokenDelta("r", 12, 2))
    assert stopped
    assert processor.finished
    assert first + last == "hello "
    assert processor.state.stop_sequence == "END"
    assert processor.usage() == (7, 2)
    assert processor.push(RequestFinished("r", FINISH_CANCELLED, 7, 2)) == (
        "",
        False,
    )
    assert processor.state.finish_reason == FINISH_STOP_TOKEN


def test_output_processor_error_is_terminal_and_preserves_partial_text():
    processor = OutputProcessor(
        "r", FakeTokenizer(), stop_sequences=["END"], prompt_tokens=7
    )
    processor._decoder = MagicMock()
    processor._decoder.push.return_value = "EN"
    assert processor.push(TokenDelta("r", 11, 1)) == ("", False)
    assert processor.push(RequestError("r", "test", "failed")) == ("EN", False)
    assert processor.finished
    assert processor.state.finish_reason == FINISH_ABORTED
    assert processor.usage() == (7, 1)


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", ["length", "stop_token", "stop"])
def test_protocol_delivers_terminal_text_and_authoritative_usage(
    protocol, stream, reason
):
    state = {"closed": False}
    first = StreamChunk("Hello ", [1], False)
    final = StreamChunk("world" if reason == "stop" else "EN", [2], reason == "stop")
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
            if reason == "stop":
                assert terminal["delta"]["stop_sequence"] == "END"
            usage = terminal["usage"]
        else:
            text = response["content"][0]["text"]
            assert response["stop_reason"] == expected_reason
            if reason == "stop":
                assert response["stop_sequence"] == "END"
            usage = response["usage"]
        assert usage == {"input_tokens": 3, "output_tokens": 2}
    assert text == expected_text


def test_openai_tool_parser_receives_terminal_body_once():
    builder = OpenAIResponseBuilder()
    builder._resp_id = "response"
    builder._model = "test"
    builder._content_started = False
    builder._body = ""
    builder._parser = SimpleNamespace(
        feed=Mock(side_effect=[[], [{"content": "tail"}]])
    )
    assert not builder.format_chunk("full ", delta_token_ids=[1])
    events = builder.format_chunk("tail", delta_token_ids=[2])
    assert events
    assert builder._parser.feed.call_args_list == [
        call("full ", delta_token_ids=[1]),
        call("full tail", delta_token_ids=[2]),
    ]
