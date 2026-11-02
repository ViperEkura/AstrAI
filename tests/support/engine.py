"""Unit tests for InferenceEngine.generate() and the frontend event protocol."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from astrai.inference.core.events import (
    FINISH_CANCELLED,
    RequestError,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.frontend.engine import InferenceEngine
from tests.support.tokenizers import FakeTokenizer


def make_engine_mocks(decode=None):
    """Build the standard mock model/tokenizer pair used by engine tests."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.eos_token_id = 0
    if decode is not None:
        mock_tokenizer.decode.return_value = decode
    return mock_model, mock_tokenizer


class _CharacterDecoder:
    """Deterministic CPU decoder for frontend lifecycle tests."""

    def __init__(self, tokenizer):
        pass

    def push(self, token_id):
        return chr(token_id) if token_id > 2 else ""


@pytest.fixture
def event_engine(monkeypatch):
    monkeypatch.setattr(
        "astrai.inference.frontend.output_processor.StreamDecoder", _CharacterDecoder
    )
    with patch("astrai.inference.frontend.engine.EngineCore") as core_cls:
        core_cls.return_value.max_seq_len = 32
        engine = InferenceEngine(MagicMock(), FakeTokenizer(), max_seq_len=32)
    engine._core = MagicMock()
    engine._core.cancel_request.side_effect = lambda rid: (
        engine._tracker([RequestFinished(rid, FINISH_CANCELLED)]) or True
    )
    emit_on_submission(engine)
    return engine


def emit_on_submission(engine, text="", reason=None, error=False):
    def emit(rid, prompt_ids):
        events = [TokenDelta(rid, ord(char), i + 1) for i, char in enumerate(text)]
        if error:
            events.append(RequestError(rid, "test", "failed"))
        elif reason is not None:
            events.append(RequestFinished(rid, reason, len(prompt_ids), len(text)))
        engine._tracker(events)

    def add_request(**kwargs):
        rid = kwargs["request_id"]
        emit(rid, kwargs["prompt_ids"])
        return rid

    def add_requests(**kwargs):
        for rid, ids in zip(kwargs["request_ids"], kwargs["prompts_ids"]):
            emit(rid, ids)
        return kwargs["request_ids"]

    engine._core.add_request.side_effect = add_request
    engine._core.add_requests.side_effect = add_requests


def assert_frontend_released(engine):
    tracker = engine._tracker
    assert not tracker._requests


def collect_async(stream):
    async def collect():
        return [chunk async for chunk in stream]

    return asyncio.run(asyncio.wait_for(collect(), timeout=2))
