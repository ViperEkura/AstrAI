"""Unit tests for InferenceEngine.generate() and the frontend event protocol."""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from astrai.extension import op_backend
from astrai.inference.core.events import (
    FINISH_CANCELLED,
    FINISH_LENGTH,
    FINISH_STOP_TOKEN,
    RequestFinished,
)
from astrai.inference.frontend.engine import build_engine
from tests.support.engine import (
    _assert_frontend_released,
    _collect_async,
    _emit_on_submission,
)
from tests.support.engine import event_engine as event_engine
from tests.support.models import make_model
from tests.support.tokenizers import FakeTokenizer


@pytest.mark.parametrize(
    ("text", "stop"), [("ab!ignored", "!"), ("abENDignored", "END")]
)
def test_generate_events_text_stop_aborts_before_yielding_final(
    event_engine, text, stop
):
    engine = event_engine
    _emit_on_submission(engine, text)
    prompt = "p" * 40  # Usage must reflect the actual, left-truncated prompt.

    async def consume():
        stream = engine.generate_events(prompt, stop_sequences=[stop])
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            if chunk.is_final:
                engine._core.cancel_request.assert_called_once()
                _assert_frontend_released(engine)
                break
        await stream.aclose()
        return chunks

    chunks = asyncio.run(asyncio.wait_for(consume(), timeout=2))
    final = chunks[-1]
    assert "".join(chunk.text for chunk in chunks) == "ab"
    assert final.is_final and final.stopped
    assert final.stop_sequence == stop
    assert final.finish_reason == "stop"
    assert final.prompt_tokens == 32
    assert final.completion_tokens == len("ab" + stop)
    assert [token for chunk in chunks for token in chunk.delta_token_ids] == list(
        ("ab" + stop).encode()
    )
    assert sum(chunk.is_final for chunk in chunks) == 1
    engine._core.cancel_request.assert_called_once()


@pytest.mark.parametrize("reason", [FINISH_STOP_TOKEN, FINISH_LENGTH, FINISH_CANCELLED])
def test_generate_events_core_terminal_flushes_tail_without_abort(event_engine, reason):
    _emit_on_submission(event_engine, "hello EN", reason)
    chunks = _collect_async(event_engine.generate_events("hi", stop_sequences=["END"]))
    assert "".join(chunk.text for chunk in chunks) == "hello EN"
    final = chunks[-1]
    assert final.text == "EN"
    assert final.finish_reason == ("stop" if reason == FINISH_STOP_TOKEN else reason)
    assert final.prompt_tokens == 2
    assert final.completion_tokens == len("hello EN")
    assert [token for chunk in chunks for token in chunk.delta_token_ids] == list(
        b"hello EN"
    )
    assert not final.stopped
    assert sum(chunk.is_final for chunk in chunks) == 1
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


def test_generate_events_stop_after_core_terminal_needs_no_abort(event_engine):
    _emit_on_submission(event_engine, "ab!ignored", FINISH_LENGTH)
    chunks = _collect_async(event_engine.generate_events("hi", stop_sequences=["!"]))
    assert chunks[-1].stopped
    assert "".join(chunk.text for chunk in chunks) == "ab"
    assert chunks[-1].prompt_tokens == 2
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
@pytest.mark.parametrize("exit_kind", ["close", "exception", "cancel"])
def test_async_consumer_exit_cancels_and_releases(event_engine, method, exit_kind):
    _emit_on_submission(event_engine, "a")

    async def consume():
        stream = getattr(event_engine, method)("hi")
        await anext(stream)
        if exit_kind == "close":
            await stream.aclose()
        elif exit_kind == "exception":
            with pytest.raises(RuntimeError, match="consumer failed"):
                await stream.athrow(RuntimeError("consumer failed"))
        else:
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await stream.aclose()

    asyncio.run(asyncio.wait_for(consume(), timeout=2))
    event_engine._core.cancel_request.assert_called_once()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_async_close_after_core_terminal_is_not_cancelled(event_engine, method):
    _emit_on_submission(event_engine, "abc", FINISH_LENGTH)

    async def consume():
        stream = getattr(event_engine, method)("hi")
        await anext(stream)  # Terminal is queued, but not folded yet.
        await stream.aclose()

    asyncio.run(asyncio.wait_for(consume(), timeout=2))
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_unstarted_async_stream_does_not_submit_or_leak(event_engine, method):
    stream = getattr(event_engine, method)("hi")
    asyncio.run(stream.aclose())
    event_engine._core.add_request.assert_not_called()
    event_engine._core.add_requests.assert_not_called()
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_async_stream_does_not_send_per_request_attention_choice(event_engine, method):
    _emit_on_submission(event_engine, reason=FINISH_LENGTH)
    with op_backend(attention="torch"):
        stream = getattr(event_engine, method)("hi")
    _collect_async(stream)
    assert "backend" not in event_engine._core.add_requests.call_args.kwargs


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_async_submission_preserves_scalar_only_tokenizer(
    event_engine, monkeypatch, method
):
    def encode(prompt, *, add_special_tokens):
        assert prompt == "hi" and add_special_tokens is True
        return [[10, 11]]

    encode = MagicMock(side_effect=encode)
    monkeypatch.setattr(event_engine.tokenizer, "encode", encode)
    _emit_on_submission(event_engine, reason=FINISH_LENGTH)

    _collect_async(getattr(event_engine, method)("hi", temperature=0.7, top_k=3))

    encode.assert_called_once_with("hi", add_special_tokens=True)
    event_engine._core.add_request.assert_not_called()
    event_engine._core.add_requests.assert_called_once()
    submitted = event_engine._core.add_requests.call_args.kwargs
    assert submitted["prompts"] == ["hi"]
    assert submitted["prompts_ids"] == [[10, 11]]
    assert len(submitted["request_ids"]) == 1
    assert submitted["temperature"] == 0.7 and submitted["top_k"] == 3
    _assert_frontend_released(event_engine)


def test_unstarted_sync_stream_does_not_submit_or_leak(event_engine):
    stream = event_engine.generate("hi", stream=True)
    stream.close()
    event_engine._core.add_requests.assert_not_called()
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("core_finished", [False, True])
def test_sync_stream_close_cancels_only_unfinished_core(event_engine, core_finished):
    _emit_on_submission(event_engine, "abc", FINISH_LENGTH if core_finished else None)
    stream = event_engine.generate("hi", stream=True)
    assert next(stream) == "a"
    stream.close()
    assert event_engine._core.cancel_request.call_count == int(not core_finished)
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize(
    "method", ["generate", "stream", "generate_events", "generate_async"]
)
def test_request_error_terminates_every_frontend_path(
    event_engine, monkeypatch, method
):
    _emit_on_submission(event_engine, "ab", error=True)
    original_wait = event_engine._tracker.wait

    def wait(rid, timeout=None):
        assert event_engine._tracker.is_finished(rid), "unexpected wait for terminal"
        return original_wait(rid, timeout=timeout)

    monkeypatch.setattr(event_engine._tracker, "wait", wait)
    if method == "generate":
        assert event_engine.generate("hi") == "ab"
    elif method == "stream":
        # A fold that fails to mark RequestError terminal must fail, not hang.
        monkeypatch.setattr(
            event_engine._tracker,
            "wait",
            MagicMock(side_effect=AssertionError("stream hung")),
        )
        assert "".join(event_engine.generate("hi", stream=True)) == "ab"
    else:
        chunks = _collect_async(getattr(event_engine, method)("hi"))
        if method == "generate_events":
            assert "".join(chunk.text for chunk in chunks) == "ab"
            assert chunks[-1].finish_reason == "aborted"
            assert chunks[-1].prompt_tokens == 2
            assert chunks[-1].completion_tokens == 2
        else:
            assert "".join(chunks) == "ab"
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("max_tokens", [0, -1, None])
@pytest.mark.parametrize(
    "method", ["generate", "stream", "generate_events", "generate_async"]
)
def test_zero_token_terminal_does_not_wait_for_token_delta(
    event_engine, monkeypatch, max_tokens, method
):
    _emit_on_submission(event_engine, reason=FINISH_LENGTH)
    prompt = "p" * 32 if max_tokens is None else "hi"
    if method in ("generate", "stream"):
        if method == "stream":
            monkeypatch.setattr(
                event_engine._tracker,
                "wait",
                MagicMock(side_effect=AssertionError("stream hung")),
            )
        result = event_engine.generate(
            prompt, stream=method == "stream", max_tokens=max_tokens
        )
        assert (list(result) if method == "stream" else result) == (
            [] if method == "stream" else ""
        )
        if max_tokens is not None:
            event_engine._core.add_requests.assert_not_called()
    else:
        chunks = _collect_async(
            getattr(event_engine, method)(prompt, max_tokens=max_tokens)
        )
        if method == "generate_events":
            assert len(chunks) == 1
            assert chunks[0].text == ""
            assert chunks[0].finish_reason == "length"
            assert chunks[0].prompt_tokens == len(prompt)
            assert chunks[0].completion_tokens == 0
        else:
            assert chunks == []
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


def test_non_streaming_still_bulk_decodes_once_without_helper_threads(
    event_engine, monkeypatch
):
    _emit_on_submission(event_engine, "abc", FINISH_LENGTH)
    event_engine.tokenizer._tokenizer = object()
    event_engine.tokenizer.decode = MagicMock(wraps=event_engine.tokenizer.decode)
    monkeypatch.setattr(
        "astrai.inference.frontend.engine.OutputProcessor",
        MagicMock(side_effect=AssertionError("incremental fold used")),
    )
    monkeypatch.setattr(
        threading,
        "Thread",
        MagicMock(side_effect=AssertionError("helper thread started")),
    )
    assert event_engine.generate(["hi", "there"]) == ["abc", "abc"]
    assert event_engine.tokenizer.decode.call_count == 2
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


def test_blocking_timeout_cancels_only_unfinished_requests(event_engine, monkeypatch):
    def submit(**kwargs):
        event_engine._tracker.sink(
            [RequestFinished(kwargs["request_ids"][0], FINISH_LENGTH, 2, 0)]
        )
        return kwargs["request_ids"]

    event_engine._core.add_requests.side_effect = submit
    monkeypatch.setattr("astrai.inference.frontend.engine._GENERATE_TIMEOUT_S", 0.01)
    with pytest.raises(TimeoutError, match="1/2 completed"):
        event_engine.generate(["hi", "there"])
    request_ids = event_engine._core.add_requests.call_args.kwargs["request_ids"]
    event_engine._core.cancel_request.assert_called_once_with(request_ids[1])
    _assert_frontend_released(event_engine)


def test_blocking_wait_exception_cancels_and_releases_batch(event_engine, monkeypatch):
    monkeypatch.setattr(
        event_engine._tracker, "wait", MagicMock(side_effect=KeyboardInterrupt)
    )
    with pytest.raises(KeyboardInterrupt):
        event_engine.generate(["hi", "there"])
    assert event_engine._core.cancel_request.call_count == 2
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize(
    "method", ["generate", "stream", "generate_events", "generate_async"]
)
def test_submission_failure_releases_frontend_state(event_engine, method):
    event_engine._core.add_request.side_effect = RuntimeError("submit failed")
    event_engine._core.add_requests.side_effect = RuntimeError("submit failed")
    with pytest.raises(RuntimeError, match="submit failed"):
        if method == "generate":
            event_engine.generate("hi")
        elif method == "stream":
            list(event_engine.generate("hi", stream=True))
        else:
            _collect_async(getattr(event_engine, method)("hi"))
    event_engine._core.cancel_request.assert_called_once()
    _assert_frontend_released(event_engine)


def test_batch_submission_failure_releases_all_registered_consumers(event_engine):
    def submit(**kwargs):
        request_ids = kwargs["request_ids"]
        assert set(request_ids) == set(event_engine._tracker._finished)
        event_engine._tracker.sink([RequestFinished(request_ids[0], FINISH_LENGTH)])
        raise RuntimeError("submit failed after first completion")

    event_engine._core.add_requests.side_effect = submit
    with pytest.raises(RuntimeError, match="submit failed"):
        event_engine.generate(["first", "second"])

    request_ids = event_engine._core.add_requests.call_args.kwargs["request_ids"]
    event_engine._core.cancel_request.assert_called_once_with(request_ids[1])
    _assert_frontend_released(event_engine)


def test_output_fold_exception_cancels_and_releases_stream(event_engine, monkeypatch):
    _emit_on_submission(event_engine, "a")
    monkeypatch.setattr(
        "astrai.inference.frontend.engine.OutputProcessor.push",
        MagicMock(side_effect=RuntimeError("fold failed")),
    )
    with pytest.raises(RuntimeError, match="fold failed"):
        _collect_async(event_engine.generate_events("hi"))
    event_engine._core.cancel_request.assert_called_once()
    _assert_frontend_released(event_engine)


def test_async_backlog_is_bounded_for_full_context_without_token_loss(event_engine):
    event_engine._max_seq_len = 5000
    _emit_on_submission(event_engine, "a" * 4500, FINISH_LENGTH)
    chunks = _collect_async(event_engine.generate_events("hi"))
    assert "".join(chunk.text for chunk in chunks) == "a" * 4500
    assert chunks[-1].completion_tokens == 4500
    event_engine._core.cancel_request.assert_not_called()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize("method", ["generate_events", "generate_async"])
def test_cancellation_before_first_token_releases_request(event_engine, method):
    async def consume():
        stream = getattr(event_engine, method)("hi")
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        event_engine._core.add_requests.assert_called_once()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await stream.aclose()

    asyncio.run(asyncio.wait_for(consume(), timeout=2))
    event_engine._core.cancel_request.assert_called_once()
    _assert_frontend_released(event_engine)


@pytest.mark.parametrize(
    "method", ["generate", "stream", "generate_events", "generate_async"]
)
@pytest.mark.parametrize("full_prompt", [False, True])
def test_real_cpu_engine_zero_output_boundaries(monkeypatch, method, full_prompt):
    model, _ = make_model("cpu", max_position_embeddings=16)
    monkeypatch.setattr("astrai.inference.frontend.engine._GENERATE_TIMEOUT_S", 2)
    with build_engine(
        model=model,
        tokenizer=FakeTokenizer(),
        device=None,
        dtype=None,
        max_seq_len=16,
        max_batch_size=1,
        enable_cuda_graph=False,
    ) as engine:
        original_wait = engine._tracker.wait
        waits = 0

        def bounded_wait(rid, timeout=None):
            nonlocal waits
            waits += 1
            assert waits < 40, "zero-output stream waited without a terminal"
            return original_wait(rid, timeout=timeout)

        monkeypatch.setattr(engine._tracker, "wait", bounded_wait)
        engine._core.cancel_request = MagicMock(wraps=engine._core.cancel_request)
        prompt, max_tokens = ("p" * 32, None) if full_prompt else ("hi", 0)
        if method == "generate":
            assert engine.generate(prompt, max_tokens=max_tokens) == ""
        elif method == "stream":
            assert (
                list(engine.generate(prompt, stream=True, max_tokens=max_tokens)) == []
            )
        else:
            chunks = _collect_async(
                getattr(engine, method)(prompt, max_tokens=max_tokens)
            )
            if method == "generate_events":
                assert len(chunks) == 1
                assert chunks[0].text == ""
                assert chunks[0].finish_reason == "length"
                assert chunks[0].prompt_tokens == (16 if full_prompt else 2)
                assert chunks[0].completion_tokens == 0
            else:
                assert chunks == []
        engine._core.cancel_request.assert_not_called()
        _assert_frontend_released(engine)


@pytest.mark.parametrize("error", [False, True])
def test_stream_deltas_keep_tokens_with_delayed_or_empty_text(
    event_engine, monkeypatch, error
):
    decoder = MagicMock()
    decoder.push.side_effect = ["", "ab", ""]
    monkeypatch.setattr(
        "astrai.inference.frontend.output_processor.StreamDecoder",
        lambda tokenizer: decoder,
    )
    _emit_on_submission(event_engine, "abc", FINISH_LENGTH, error=error)
    chunks = _collect_async(event_engine.generate_events("hi"))
    assert [chunk.text for chunk in chunks] == ["ab", ""]
    assert [chunk.delta_token_ids for chunk in chunks] == [[97, 98], [99]]
    assert chunks[-1].completion_tokens == 3
    assert chunks[-1].finish_reason == ("aborted" if error else "length")
    _assert_frontend_released(event_engine)
