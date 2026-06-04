"""Unit tests for the inference HTTP server."""

import pytest

from astrai.inference import get_app


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

    async def async_gen():
        yield "Hello"
        yield "X"
        yield "world"

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


def test_chat_completions_stop_sequence_stream(client, loaded_model):
    """POST /v1/chat/completions with stop parameter truncates SSE stream."""

    async def async_gen():
        yield "Hello"
        yield "X"
        yield "world"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
