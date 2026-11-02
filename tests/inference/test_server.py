"""HTTP protocol conversion and one real CPU inference roundtrip."""

import json
from pathlib import Path

import pytest
import torch

from astrai.inference import build_engine, get_app
from astrai.inference.frontend.tracking import StreamChunk
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.serialization import save_model
from tests.support.models import make_tiny_config
from tests.support.tokenizers import CHAT_TEMPLATE, build_test_tokenizer


@pytest.mark.parametrize("loaded", [False, True])
def test_health_reports_model_availability(client, mock_engine, loaded):
    get_app().state.engine = mock_engine if loaded else None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": loaded}


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("stream", [False, True])
def test_http_protocol_preserves_text_usage_and_terminal_sequence(
    client, loaded_model, protocol, stream
):
    closed = []

    async def events():
        try:
            yield StreamChunk("Hello ", [1], False)
            final = StreamChunk("world", [2], False)
            final.finish_reason = "length"
            final.prompt_tokens = 3
            final.completion_tokens = 2
            yield final
        finally:
            closed.append(True)

    loaded_model.generate_events.return_value = events()
    endpoint = "/v1/chat/completions" if protocol == "openai" else "/v1/messages"
    response = client.post(
        endpoint,
        json={"messages": [{"role": "user", "content": "Hi"}], "stream": stream},
    )
    assert response.status_code == 200
    assert closed == [True]
    if stream:
        lines = response.text.splitlines()
        assert lines.count("data: [DONE]") == 1
        payloads = [
            json.loads(line[6:])
            for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        if protocol == "openai":
            choices = [
                payload["choices"][0] for payload in payloads if "choices" in payload
            ]
            assert choices[0]["delta"]["role"] == "assistant"
            assert (
                "".join(choice["delta"].get("content", "") for choice in choices)
                == "Hello world"
            )
            assert choices[-1]["finish_reason"] == "length"
            usage = payloads[-1]
        else:
            assert [payload["type"] for payload in payloads] == [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
            assert (
                "".join(
                    payload["delta"]["text"]
                    for payload in payloads
                    if payload["type"] == "content_block_delta"
                )
                == "Hello world"
            )
            assert payloads[-2]["delta"]["stop_reason"] == "max_tokens"
            usage = payloads[-2]["usage"]
    else:
        payload = response.json()
        usage = payload["usage"]
        if protocol == "openai":
            assert payload["object"] == "chat.completion"
            assert payload["choices"][0]["message"]["content"] == "Hello world"
            assert payload["choices"][0]["finish_reason"] == "length"
        else:
            assert payload["type"] == "message"
            assert payload["role"] == "assistant"
            assert payload["content"] == [{"type": "text", "text": "Hello world"}]
            assert payload["stop_reason"] == "max_tokens"
    expected_usage = (
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        if protocol == "openai"
        else {"input_tokens": 3, "output_tokens": 2}
    )
    assert usage == expected_usage


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
