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


def _make_engine_mocks(decode=None):
    """Build the standard mock model/tokenizer pair used by engine tests."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.stop_ids = [0]
    if decode is not None:
        mock_tokenizer.decode.return_value = decode
    return mock_model, mock_tokenizer


def _drive_events(engine, events_by_id):
    """Push output events through the engine's installed sink (mock schedulers
    emit nothing on their own; tests drive the event protocol directly)."""
    sink = engine.scheduler._event_sink
    flat = []
    for rid, events in events_by_id.items():
        flat.extend(events)
    sink(flat)


def _mock_add_requests(events_by_id, tokenizer_ids=None):
    """Build an add_requests side effect that returns ids and, on the NEXT
    scheduler tick (i.e. when the test drives the sink), replays events."""

    def fake(prompts, **kw):
        # Return core-minted ids matching the ids the engine registered.
        return list(events_by_id.keys())[: len(prompts)]

    return fake


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
    with patch("astrai.inference.frontend.engine.Scheduler"):
        engine = InferenceEngine(MagicMock(), FakeTokenizer(), max_seq_len=32)
    engine._core = MagicMock()
    engine._core.abort_request.side_effect = lambda rid: (
        engine._tracker.sink([RequestFinished(rid, FINISH_CANCELLED)]) or True
    )
    _emit_on_submission(engine)
    return engine


def _emit_on_submission(engine, text="", reason=None, error=False):
    def emit(rid, prompt_ids):
        events = [TokenDelta(rid, ord(char), i + 1) for i, char in enumerate(text)]
        if error:
            events.append(RequestError(rid, "test", "failed"))
        elif reason is not None:
            events.append(RequestFinished(rid, reason, len(prompt_ids), len(text)))
        engine._tracker.sink(events)

    def send_request(**kwargs):
        rid = kwargs["request_id"]
        emit(rid, kwargs["prompt_ids"])
        return rid

    def send_requests(**kwargs):
        for rid, ids in zip(kwargs["request_ids"], kwargs["prompts_ids"]):
            emit(rid, ids)
        return kwargs["request_ids"]

    engine._core.send_request.side_effect = send_request
    engine._core.send_requests.side_effect = send_requests


def _assert_frontend_released(engine):
    tracker = engine._tracker
    assert not tracker._finished
    assert not tracker.sink._queues
    assert not tracker.sink._async_subscribers
    assert not tracker.sink._terminal


def _collect_async(stream):
    async def collect():
        return [chunk async for chunk in stream]

    return asyncio.run(asyncio.wait_for(collect(), timeout=2))
