"""Unit tests for the inference HTTP server."""

from pathlib import Path

import pytest
import torch

from astrai.inference import build_engine, get_app
from astrai.model.transformer import AutoRegressiveLM
from astrai.serialization import save_model
from tests.helpers import CHAT_TEMPLATE, build_test_tokenizer, make_tiny_config


def test_health_no_model(client):
    """GET /health should return 200 even when engine not loaded."""
    get_app().state.engine = None
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert not data["model_loaded"]


def test_health_with_model(client, loaded_model):
    """GET /health should return 200 when engine is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_chat_completions_non_stream(client, loaded_model):
    """POST /v1/chat/completions with stream=false returns OpenAI-style JSON."""

    async def async_gen():
        yield "Assistant reply"

    get_app().state.engine = loaded_model
    loaded_model.generate_async.return_value = async_gen()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]


def test_chat_completions_stream(client, loaded_model):
    """POST /v1/chat/completions with stream=true returns SSE stream."""

    async def async_gen():
        yield "cumulative1"
        yield "cumulative2"

    get_app().state.engine = loaded_model
    loaded_model.generate_async.return_value = async_gen()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": True,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    lines = [
        line.strip() for line in response.content.decode("utf-8").split("\n") if line
    ]
    assert any("cumulative1" in line for line in lines)
    assert any("cumulative2" in line for line in lines)
    assert any("[DONE]" in line for line in lines)


def test_messages_non_stream(client, loaded_model):
    """POST /v1/messages with stream=false returns Anthropic-style JSON."""

    async def async_gen():
        yield "Assistant reply"

    get_app().state.engine = loaded_model
    loaded_model.generate_async.return_value = async_gen()
    response = client.post(
        "/v1/messages",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    assert "usage" in data
    assert "input_tokens" in data["usage"]


def test_messages_stream(client, loaded_model):
    """POST /v1/messages with stream=true returns Anthropic SSE stream."""

    async def async_gen():
        yield "cumulative1"
        yield "cumulative2"

    get_app().state.engine = loaded_model
    loaded_model.generate_async.return_value = async_gen()
    response = client.post(
        "/v1/messages",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": True,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "message_start" in content
    assert "content_block_start" in content
    assert "content_block_delta" in content
    assert "cumulative1" in content
    assert "cumulative2" in content
    assert "content_block_stop" in content
    assert "message_delta" in content
    assert "message_stop" in content


def test_messages_with_system(client, loaded_model):
    """POST /v1/messages with system prompt."""

    async def async_gen():
        yield "Reply"

    get_app().state.engine = loaded_model
    loaded_model.generate_async.return_value = async_gen()
    response = client.post(
        "/v1/messages",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are a helpful assistant.",
            "max_tokens": 100,
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"


def test_chat_completions_stop_sequence(client, loaded_model):
    """POST /v1/chat/completions with stop parameter truncates at stop sequence."""
    closed = []

    async def async_gen():
        try:
            yield "Hello"
            yield "X"
            yield "world"
        finally:
            closed.append(True)

    get_app().state.engine = loaded_model
    loaded_model.generate_async.return_value = async_gen()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": False,
            "stop": ["X"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    assert "X" in content
    assert "world" not in content
    assert closed == [True]


def test_chat_completions_stop_sequence_stream(client, loaded_model):
    """POST /v1/chat/completions with stop parameter truncates SSE stream."""
    closed = []

    async def async_gen():
        try:
            yield "Hello"
            yield "X"
            yield "world"
        finally:
            closed.append(True)

    get_app().state.engine = loaded_model
    loaded_model.generate_async.return_value = async_gen()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stream": True,
            "stop": ["X"],
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    content = response.content.decode("utf-8")
    assert "Hello" in content
    assert "world" not in content
    assert any(
        "finish_reason" in line for line in content.split("\n") if "stop" in line
    )
    assert closed == [True]


def test_chat_completions_real_engine(tmp_path, client):
    """POST /v1/chat/completions with a real tiny model and tokenizer."""
    cfg = make_tiny_config(vocab_size=256)
    model = AutoRegressiveLM(cfg).eval()
    save_model(cfg.to_dict(), model.state_dict(), str(tmp_path))

    tokenizer = build_test_tokenizer(vocab_size=256, chat_template=CHAT_TEMPLATE)
    tokenizer.save_pretrained(str(tmp_path))

    engine = build_engine(
        Path(tmp_path),
        device="cpu",
        dtype=torch.float32,
        max_batch_size=1,
        max_seq_len=64,
    )
    try:
        get_app().state.engine = engine
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 4,
                "temperature": 0.0,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert data["usage"]["completion_tokens"] > 0
    finally:
        engine.shutdown()
        get_app().state.engine = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
