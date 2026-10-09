"""Unit tests for InferenceEngine.generate() and the frontend event protocol."""

import threading
import time
from unittest.mock import patch

import pytest

from astrai.extension import TorchNativeBackend, attn_backend
from astrai.inference.core.events import (
    FINISH_LENGTH,
    RequestFinished,
    TokenDelta,
)
from astrai.inference.frontend.engine import InferenceEngine, build_engine
from tests.support.engine import (
    _make_engine_mocks,
)
from tests.support.models import make_model
from tests.support.tokenizers import FakeTokenizer


def test_engine_generate_non_streaming_single():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="response")
    # Mock tokenizer returns a flat list for both single and batch shapes.
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_requests.side_effect = lambda prompts, **kw: [
            f"request-{i}" for i in range(len(prompts))
        ]
        instance.add_request.side_effect = lambda prompt, **kw: "request-0"
        instance.remove_request.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)

        # Drive generation from another thread: emit events after submit.
        def run():
            result = eng.generate("hello")
            return result

        # The events must arrive after generate() registered the request but
        # the mock scheduler never runs a loop, so a helper thread polls the
        # tracker and injects the terminal sequence.
        def inject():
            deadline = time.time() + 5
            while time.time() < deadline:
                if eng._tracker._sink._queues:
                    rid = next(iter(eng._tracker._sink._queues))
                    eng._tracker.sink(
                        [
                            TokenDelta(rid, 11, 1),
                            TokenDelta(rid, 12, 2),
                            RequestFinished(
                                rid, FINISH_LENGTH, prompt_tokens=3, completion_tokens=2
                            ),
                        ]
                    )
                    return
                time.sleep(0.01)

        t = threading.Thread(target=inject, daemon=True)
        t.start()
        result = run()
        t.join(timeout=5)
        assert not t.is_alive()
        assert result != ""


def test_engine_generate_streaming_yields_token_ids():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="tok")
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_requests.side_effect = lambda prompts, **kw: [
            f"request-{i}" for i in range(len(prompts))
        ]
        instance.cancel_request.return_value = True

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        gen = eng.generate("hello", stream=True)

        def inject():
            deadline = time.time() + 5
            while time.time() < deadline:
                if eng._tracker._sink._queues:
                    rid = next(iter(eng._tracker._sink._queues))
                    eng._tracker.sink(
                        [
                            TokenDelta(rid, 11, 1),
                            TokenDelta(rid, 12, 2),
                            RequestFinished(rid, FINISH_LENGTH, 3, 2),
                        ]
                    )
                    return
                time.sleep(0.01)

        t = threading.Thread(target=inject, daemon=True)
        t.start()
        tokens = list(gen)
        t.join(timeout=5)
        assert not t.is_alive()
        assert len(tokens) >= 1  # decoder is a mock; at least one fragment


def test_engine_generate_non_streaming_batch():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="r")
    mock_tokenizer.encode.return_value = [1, 2, 3]

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_requests.side_effect = lambda prompts, **kw: [
            f"request-{i}" for i in range(len(prompts))
        ]
        instance.remove_request.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=2)

        def inject():
            deadline = time.time() + 5
            while time.time() < deadline:
                queues = eng._tracker._sink._queues
                if len(queues) >= 2:
                    ids = list(queues)
                    events = []
                    for rid in ids:
                        events.extend(
                            [
                                TokenDelta(rid, 11, 1),
                                RequestFinished(rid, FINISH_LENGTH, 3, 1),
                            ]
                        )
                    eng._tracker.sink(events)
                    return
                time.sleep(0.01)

        t = threading.Thread(target=inject, daemon=True)
        t.start()
        results = eng.generate(["hello", "world"])
        t.join(timeout=5)
        assert not t.is_alive()
        assert len(results) == 2


def test_engine_generate_zero_max_tokens_returns_empty():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.remove_request.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=2)
        assert eng.generate(["hello", "world"], max_tokens=0) == ["", ""]
        instance.add_requests.assert_not_called()


def test_engine_generate_zero_max_tokens_stream_is_empty():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        assert list(eng.generate("hello", stream=True, max_tokens=0)) == []
        instance.add_requests.assert_not_called()


def test_engine_passes_backend_to_scheduler():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        InferenceEngine(
            mock_model,
            mock_tokenizer,
            max_batch_size=1,
            backend="torch_native",
        )

    assert MockSched.call_args.kwargs["backend"] == "torch_native"


@pytest.mark.parametrize("enable_overlap", [False, True])
def test_engine_passes_overlap_to_scheduler(enable_overlap):
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        InferenceEngine(
            mock_model,
            mock_tokenizer,
            max_batch_size=1,
            enable_overlap=enable_overlap,
        )

    assert MockSched.call_args.kwargs["enable_overlap"] is enable_overlap


def test_generate_captures_calling_backend_context():
    mock_model, mock_tokenizer = _make_engine_mocks()
    captured = []

    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.cancel_request.return_value = True

        def fake_add_tasks(prompts, **kwargs):
            captured.append(kwargs["backend"])
            return [f"request-{i}" for i in range(len(prompts))]

        instance.add_requests.side_effect = fake_add_tasks
        engine = InferenceEngine(mock_model, mock_tokenizer)
        with attn_backend("torch_native"):
            gen = engine.generate("hello", stream=True)

            # Terminate via the event protocol (the mock core has no loop).
            def inject():
                for _ in range(500):
                    qs = engine._tracker._sink._queues
                    if qs:
                        rid = next(iter(qs))
                        engine._tracker.sink(
                            [RequestFinished(rid, FINISH_LENGTH, 1, 0)]
                        )
                        return
                    time.sleep(0.01)

            t = threading.Thread(target=inject, daemon=True)
            t.start()
            assert list(gen) == []

    assert len(captured) == 1
    assert isinstance(captured[0], TorchNativeBackend)


def test_build_engine_from_live_objects_starts_scheduler():
    model, _ = make_model("cpu", max_position_embeddings=64)
    tokenizer = FakeTokenizer()
    engine = build_engine(
        model=model,
        tokenizer=tokenizer,
        device=None,
        dtype=None,
        max_batch_size=2,
    )
    try:
        assert isinstance(engine, InferenceEngine)
        assert engine.tokenizer is tokenizer
        assert engine.scheduler._stop_event.is_set() is False
    finally:
        engine.shutdown()


def test_build_engine_passes_engine_kwargs_through():
    model, _ = make_model("cpu", max_position_embeddings=64)
    backend = TorchNativeBackend()
    with patch("astrai.inference.frontend.engine.Scheduler") as MockSched:
        instance = MockSched.return_value
        instance.cancel_request.return_value = True

        def fake_add_tasks(*args, **k):
            return [
                f"request-{i}"
                for i in range(
                    len(args[0]) if args else (len(k.get("prompts", [])) or 1)
                )
            ]

        instance.add_requests.side_effect = fake_add_tasks
        engine = build_engine(
            model=model,
            tokenizer=FakeTokenizer(),
            device=None,
            dtype=None,
            cache=object(),
            enable_cuda_graph=False,
            backend=backend,
            enable_overlap=True,
        )
        gen = engine.generate("hi", stream=True)

        def inject():
            for _ in range(500):
                qs = engine._tracker._sink._queues
                if qs:
                    rid = next(iter(qs))
                    engine._tracker.sink([RequestFinished(rid, FINISH_LENGTH, 1, 0)])
                    return
                time.sleep(0.01)

        t = threading.Thread(target=inject, daemon=True)
        t.start()
        assert list(gen) == []

    kwargs = MockSched.call_args.kwargs
    assert kwargs["cache"] is not None
    assert kwargs["enable_cuda_graph"] is False
    assert kwargs["backend"] is backend
    assert kwargs["enable_overlap"] is True


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {"param_path": "x", "model": object()},
            ValueError,
            "not both",
        ),
        ({}, ValueError, "requires param_path"),
        ({"param_path": "/nonexistent-dir-xyz"}, FileNotFoundError, "not found"),
    ],
)
def test_build_engine_rejects_invalid_arguments(kwargs, error, message):
    with pytest.raises(error, match=message):
        build_engine(**kwargs)
