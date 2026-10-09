"""Unit tests for protocol builders, GenContext, StopInfo."""

import json
from unittest.mock import MagicMock

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
from astrai.inference.network.anthropic import AnthropicResponseBuilder
from astrai.inference.network.openai import OpenAIResponseBuilder
from astrai.inference.network.protocol import GenContext, StopInfo
from tests.support.tokenizers import FakeTokenizer


def _make_ctx(**kwargs):
    defaults = {
        "resp_id": "test-123",
        "created": 1000,
        "model": "test-model",
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    defaults.update(kwargs)
    return GenContext(**defaults)


def _sse_payloads(events):
    payloads = []
    for chunk in events:
        for line in chunk.strip().split("\n"):
            if line.startswith("data: "):
                try:
                    payloads.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    return payloads


def _make_openai_builder():
    builder = OpenAIResponseBuilder()
    req = MagicMock()
    req.messages = [MagicMock(role="user", content="Hello")]
    req.stop = None
    req.model = "astrai"
    engine = MagicMock()
    engine.tokenizer.apply_chat_template.return_value = "Hello"
    builder.prepare(req, engine)
    return builder


def _make_anthropic_builder():
    builder = AnthropicResponseBuilder()
    req = MagicMock()
    req.messages = [MagicMock(role="user", content="Hello")]
    req.model = "claude"
    req.system = None
    engine = MagicMock()
    engine.tokenizer.apply_chat_template.return_value = "Hello"
    builder.prepare(req, engine)
    return builder


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
    assert "".join(chunks) + tail == processor.state.text == "hello EN"
    assert processor.finished
    assert processor.state.finish_reason == reason
    assert processor.state.stop_sequence is None
    assert processor.usage() == (7, 3)
    assert processor.push(terminal) == ("", False)
    assert processor.push(TokenDelta("r", 4, 4)) == ("", False)
    assert processor.state.text == "hello EN"


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
    assert first + last == processor.state.text == "hello "
    assert processor.state.stop_sequence == "END"
    assert processor.usage() == (7, 2)
    assert processor.push(RequestFinished("r", FINISH_CANCELLED, 7, 2)) == (
        "",
        False,
    )
    assert processor.state.finish_reason == FINISH_STOP_TOKEN
    assert processor.state.text == "hello "


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


def test_openai_prepare_returns_prompt_ctx_stops():
    builder = _make_openai_builder()
    req = MagicMock()
    req.messages = [MagicMock(role="user", content="Hi")]
    req.stop = ["END"]
    req.model = "gpt"
    engine = MagicMock()
    engine.tokenizer.apply_chat_template.return_value = "Hi"
    prompt, ctx, stops = builder.prepare(req, engine)
    assert prompt == "Hi"
    assert ctx.model == "gpt"
    assert ctx.prompt_tokens == 0
    assert stops == ["END"]


def test_openai_prepare_no_stop_returns_empty_list():
    builder = _make_openai_builder()
    req = MagicMock()
    req.messages = []
    req.stop = None
    req.model = "x"
    engine = MagicMock()
    engine.tokenizer.apply_chat_template.return_value = ""
    _, _, stops = builder.prepare(req, engine)
    assert stops == []


def test_openai_format_stream_start():
    builder = _make_openai_builder()
    ctx = _make_ctx()
    events = builder.format_stream_start(ctx)
    payloads = _sse_payloads(events)
    assert len(payloads) == 1
    p = payloads[0]
    assert p["object"] == "chat.completion.chunk"
    assert p["choices"][0]["delta"]["role"] == "assistant"
    assert p["choices"][0]["finish_reason"] is None


def test_openai_format_chunk():
    builder = _make_openai_builder()
    events = builder.format_chunk("hello", body="hello")
    payload = json.loads(events[0].split("data: ", 1)[1])
    assert payload["choices"][0]["delta"]["content"] == "hello"
    assert payload["choices"][0]["finish_reason"] is None


def test_openai_format_stream_end():
    builder = _make_openai_builder()
    ctx = _make_ctx(completion_tokens=5)
    stop = StopInfo(matched="stop")
    events = builder.format_stream_end(ctx, stop)
    payloads = _sse_payloads(events)
    finish = payloads[0]
    assert finish["choices"][0]["finish_reason"] == "stop"
    usage = payloads[1]
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15


def test_openai_format_response():
    builder = _make_openai_builder()
    ctx = _make_ctx()
    stop = StopInfo()
    resp = builder.format_response(ctx, "hello", stop)
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["content"] == "hello"
    assert resp["usage"]["prompt_tokens"] == 10


def test_anthropic_prepare_messages():
    builder = _make_anthropic_builder()
    req = MagicMock()
    req.messages = [MagicMock(role="user", content="Hi")]
    req.model = "claude"
    req.system = None
    req.stop_sequences = None
    engine = MagicMock()
    engine.tokenizer.apply_chat_template.return_value = "Hi"
    prompt, ctx, stops = builder.prepare(req, engine)
    assert prompt == "Hi"
    assert stops == []


def test_anthropic_prepare_with_stop_sequences():
    builder = _make_anthropic_builder()
    req = MagicMock()
    req.messages = []
    req.model = "x"
    req.stop_sequences = ["stop", "end"]
    req.system = None
    engine = MagicMock()
    engine.tokenizer.apply_chat_template.return_value = ""
    _, _, stops = builder.prepare(req, engine)
    assert stops == ["stop", "end"]


def test_anthropic_format_stream_start():
    builder = _make_anthropic_builder()
    ctx = _make_ctx(prompt_tokens=3)
    events = builder.format_stream_start(ctx)
    payloads = _sse_payloads(events)
    assert len(payloads) == 2
    assert payloads[0]["type"] == "message_start"
    assert payloads[0]["message"]["usage"]["input_tokens"] == 3
    assert payloads[1]["type"] == "content_block_start"


def test_anthropic_format_chunk():
    builder = _make_anthropic_builder()
    events = builder.format_chunk("tok", body="tok")
    payload = json.loads(events[0].split("data: ", 1)[1])
    assert payload["type"] == "content_block_delta"
    assert payload["delta"]["text"] == "tok"


def test_anthropic_format_stream_end_no_stop():
    builder = _make_anthropic_builder()
    ctx = _make_ctx(completion_tokens=3)
    stop = StopInfo()
    events = builder.format_stream_end(ctx, stop)
    payloads = _sse_payloads(events)
    types = [p["type"] for p in payloads]
    assert types == ["content_block_stop", "message_delta", "message_stop"]
    assert payloads[1]["delta"]["stop_reason"] == "end_turn"


def test_anthropic_format_stream_end_with_stop_trims_and_emits_remaining():
    builder = _make_anthropic_builder()
    ctx = _make_ctx(completion_tokens=7)
    stop = StopInfo(
        matched="END",
        body="Hello world ",
        yielded="Hello ",
    )
    events = builder.format_stream_end(ctx, stop)
    payloads = _sse_payloads(events)
    types = [p["type"] for p in payloads]
    assert types == [
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert payloads[0]["delta"]["text"] == "world "
    assert payloads[2]["delta"]["stop_reason"] == "stop_sequence"
    assert payloads[2]["delta"]["stop_sequence"] == "END"


def test_anthropic_format_stream_end_stop_trimmed_already_yielded():
    builder = _make_anthropic_builder()
    ctx = _make_ctx()
    stop = StopInfo(
        matched="END",
        body="Hello ",
        yielded="Hello ",
    )
    events = builder.format_stream_end(ctx, stop)
    payloads = _sse_payloads(events)
    types = [p["type"] for p in payloads]
    assert types == ["content_block_stop", "message_delta", "message_stop"]


def test_anthropic_format_response_with_stop_preserves_trimmed_content():
    builder = _make_anthropic_builder()
    ctx = _make_ctx()
    stop = StopInfo(matched="STOP", body="text ", yielded="text ")
    resp = builder.format_response(ctx, "text ", stop)
    assert resp["content"][0]["text"] == "text "
    assert resp["stop_reason"] == "stop_sequence"
    assert resp["stop_sequence"] == "STOP"


def test_anthropic_format_response_no_stop():
    builder = _make_anthropic_builder()
    ctx = _make_ctx()
    stop = StopInfo()
    resp = builder.format_response(ctx, "full text", stop)
    assert resp["content"][0]["text"] == "full text"
    assert resp["stop_reason"] == "end_turn"
