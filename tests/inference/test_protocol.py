"""Unit tests for protocol builders, StopChecker, GenContext, StopInfo."""

import json
from unittest.mock import MagicMock

import pytest

from astrai.inference.api.anthropic import AnthropicResponseBuilder
from astrai.inference.api.openai import OpenAIResponseBuilder
from astrai.inference.api.protocol import GenContext, StopChecker, StopInfo
from astrai.inference.engine import GenerationRequest


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


class TestStopChecker:
    def test_check_finds_match(self):
        sc = StopChecker(["stop", "end"])
        assert sc.check("hello stop world") == "stop"

    def test_check_returns_none_when_no_match(self):
        sc = StopChecker(["stop"])
        assert sc.check("hello world") is None

    def test_check_empty_sequences(self):
        sc = StopChecker([])
        assert sc.check("hello") is None


class TestGenContext:
    def test_defaults(self):
        ctx = GenContext(resp_id="a", created=1, model="m", prompt_tokens=10)
        assert ctx.completion_tokens == 0

    def test_fields_mutable(self):
        ctx = GenContext(resp_id="a", created=1, model="m", prompt_tokens=10)
        ctx.completion_tokens = 42
        assert ctx.completion_tokens == 42


class TestStopInfo:
    def test_defaults(self):
        s = StopInfo()
        assert s.matched is None
        assert s.body == ""
        assert s.yielded == ""

    def test_with_values(self):
        s = StopInfo(matched="stop", body="hello stop", yielded="hello ")
        assert s.matched == "stop"
        assert s.body == "hello stop"
        assert s.yielded == "hello "


class TestOpenAIResponseBuilder:
    @pytest.fixture
    def builder(self):
        builder = OpenAIResponseBuilder()
        req = MagicMock()
        req.messages = [MagicMock(role="user", content="Hello")]
        req.stop = None
        req.model = "astrai"
        engine = MagicMock()
        engine.tokenizer.apply_chat_template.return_value = "Hello"
        builder.prepare(req, engine)
        return builder

    def test_prepare_returns_prompt_ctx_stops(self, builder):
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

    def test_prepare_no_stop_returns_empty_list(self, builder):
        req = MagicMock()
        req.messages = []
        req.stop = None
        req.model = "x"
        engine = MagicMock()
        engine.tokenizer.apply_chat_template.return_value = ""
        _, _, stops = builder.prepare(req, engine)
        assert stops == []

    def test_format_stream_start(self, builder):
        ctx = _make_ctx()
        events = builder.format_stream_start(ctx)
        payloads = _sse_payloads(events)
        assert len(payloads) == 1
        p = payloads[0]
        assert p["object"] == "chat.completion.chunk"
        assert p["choices"][0]["delta"]["role"] == "assistant"
        assert p["choices"][0]["finish_reason"] is None

    def test_format_chunk(self, builder):
        events = builder.format_chunk("hello", body="hello")
        payload = json.loads(events[0].split("data: ", 1)[1])
        assert payload["choices"][0]["delta"]["content"] == "hello"
        assert payload["choices"][0]["finish_reason"] is None

    def test_format_stream_end(self, builder):
        ctx = _make_ctx(completion_tokens=5)
        stop = StopInfo(matched="stop")
        events = builder.format_stream_end(ctx, stop)
        payloads = _sse_payloads(events)
        finish = payloads[0]
        assert finish["choices"][0]["finish_reason"] == "stop"
        usage = payloads[1]
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15

    def test_format_response(self, builder):
        ctx = _make_ctx()
        stop = StopInfo()
        resp = builder.format_response(ctx, "hello", stop)
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["content"] == "hello"
        assert resp["usage"]["prompt_tokens"] == 10


class TestAnthropicResponseBuilder:
    @pytest.fixture
    def builder(self):
        builder = AnthropicResponseBuilder()
        req = MagicMock()
        req.messages = [MagicMock(role="user", content="Hello")]
        req.model = "claude"
        engine = MagicMock()
        engine.tokenizer.apply_chat_template.return_value = "Hello"
        req.system = None
        builder.prepare(req, engine)
        return builder

    def test_prepare_messages(self, builder):
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

    def test_prepare_with_stop_sequences(self, builder):
        req = MagicMock()
        req.messages = []
        req.model = "x"
        req.stop_sequences = ["stop", "end"]
        req.system = None
        engine = MagicMock()
        engine.tokenizer.apply_chat_template.return_value = ""
        _, _, stops = builder.prepare(req, engine)
        assert stops == ["stop", "end"]

    def test_format_stream_start(self, builder):
        ctx = _make_ctx(prompt_tokens=3)
        events = builder.format_stream_start(ctx)
        payloads = _sse_payloads(events)
        assert len(payloads) == 2
        assert payloads[0]["type"] == "message_start"
        assert payloads[0]["message"]["usage"]["input_tokens"] == 3
        assert payloads[1]["type"] == "content_block_start"

    def test_format_chunk(self, builder):
        events = builder.format_chunk("tok", body="tok")
        payload = json.loads(events[0].split("data: ", 1)[1])
        assert payload["type"] == "content_block_delta"
        assert payload["delta"]["text"] == "tok"

    def test_format_stream_end_no_stop(self, builder):
        ctx = _make_ctx(completion_tokens=3)
        stop = StopInfo()
        events = builder.format_stream_end(ctx, stop)
        payloads = _sse_payloads(events)
        # content_block_stop, message_delta, message_stop
        types = [p["type"] for p in payloads]
        assert types == ["content_block_stop", "message_delta", "message_stop"]
        assert payloads[1]["delta"]["stop_reason"] == "end_turn"

    def test_format_stream_end_with_stop_trims_and_emits_remaining(self, builder):
        ctx = _make_ctx(completion_tokens=7)
        stop = StopInfo(
            matched="END",
            body="Hello world END extra",
            yielded="Hello ",
        )
        events = builder.format_stream_end(ctx, stop)
        payloads = _sse_payloads(events)
        # unyielded delta, content_block_stop, message_delta, message_stop
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

    def test_format_stream_end_stop_trimmed_already_yielded(self, builder):
        ctx = _make_ctx()
        stop = StopInfo(
            matched="END",
            body="Hello END",
            yielded="Hello ",
        )
        events = builder.format_stream_end(ctx, stop)
        payloads = _sse_payloads(events)
        # No unyielded delta (everything already sent)
        types = [p["type"] for p in payloads]
        assert types == ["content_block_stop", "message_delta", "message_stop"]

    def test_format_response_with_stop_trims_content(self, builder):
        ctx = _make_ctx()
        stop = StopInfo(matched="STOP", body="text STOP extra", yielded="text ")
        resp = builder.format_response(ctx, "text STOP extra", stop)
        assert resp["content"][0]["text"] == "text "
        assert resp["stop_reason"] == "stop_sequence"
        assert resp["stop_sequence"] == "STOP"

    def test_format_response_no_stop(self, builder):
        ctx = _make_ctx()
        stop = StopInfo()
        resp = builder.format_response(ctx, "full text", stop)
        assert resp["content"][0]["text"] == "full text"
        assert resp["stop_reason"] == "end_turn"


class TestGenerationRequestValidation:
    def test_valid_params(self):
        gr = GenerationRequest(
            messages=[{"role": "user", "content": "hi"}],
            top_k=50,
            top_p=0.9,
            temperature=0.7,
        )
        assert gr.top_k == 50

    def test_invalid_top_p_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            GenerationRequest(messages=[{"role": "user", "content": "hi"}], top_p=1.5)

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k"):
            GenerationRequest(messages=[{"role": "user", "content": "hi"}], top_k=-1)

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            GenerationRequest(
                messages=[{"role": "user", "content": "hi"}], temperature=-0.1
            )

    def test_top_k_zero_valid(self):
        gr = GenerationRequest(messages=[{"role": "user", "content": "hi"}], top_k=0)
        assert gr.top_k == 0
