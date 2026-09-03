"""Unit tests for GenerateResult accumulator and InferenceEngine.generate()."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

from astrai.extension import TorchNativeBackend, attn_backend
from astrai.inference import STOP
from astrai.inference.engine import GenerateResult, InferenceEngine


def _make_engine_mocks(decode=None):
    """Build the standard mock model/tokenizer pair used by engine tests."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.stop_ids = [0]
    if decode is not None:
        mock_tokenizer.decode.return_value = decode
    return mock_model, mock_tokenizer


def test_result_append_single():
    r = GenerateResult(count=1)
    r.append("hello", 0)
    assert r.results[0] == "hello"


def test_result_append_multiple_tasks():
    r = GenerateResult(count=3)
    r.append("a", 0)
    r.append("b", 1)
    r.append("c", 2)
    assert r.results[0] == "a"
    assert r.results[1] == "b"
    assert r.results[2] == "c"


def test_result_stop_marks_complete():
    r = GenerateResult(count=2)
    r.append("text", 0)
    r.append(STOP, 0)
    r.append("more", 1)
    assert r._done[0] is True
    assert r._done[1] is False
    assert r._completed == 1


def test_result_stop_does_not_double_count():
    r = GenerateResult(count=1)
    r.append(STOP, 0)
    r.append(STOP, 0)
    assert r._completed == 1


def test_result_pop_all_returns_and_clears():
    r = GenerateResult(count=2)
    r.append("a", 0)
    r.append("b", 1)
    out = r.pop_all()
    assert len(out) == 2
    assert out[0] == (0, "a")
    assert out[1] == (1, "b")
    assert r.pop_all() == []


def test_result_wait_blocks_until_data():
    r = GenerateResult(count=1)

    def delayed_append():
        import time

        time.sleep(0.05)
        r.append("delayed", 0)

    t = threading.Thread(target=delayed_append)
    t.start()
    ok = r.wait(timeout=5.0)
    t.join()
    assert ok
    assert r.results[0] == "delayed"


def test_result_wait_timeout():
    r = GenerateResult(count=1)
    ok = r.wait(timeout=0.01)
    assert not ok


def test_result_wait_completion_non_streaming():
    r = GenerateResult(count=2)

    def finish_later():
        import time

        time.sleep(0.05)
        r.append(STOP, 0)
        time.sleep(0.05)
        r.append(STOP, 1)

    t = threading.Thread(target=finish_later)
    t.start()
    r.wait_completion()
    t.join()
    assert r._completed == 2


def test_result_get_results():
    r = GenerateResult(count=2)
    r.append("hello", 0)
    r.append("world", 1)
    results = r.get_results()
    assert results == ["hello", "world"]


def test_engine_generate_non_streaming_single():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="response")

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value

        def fake_add(prompt, **kw):
            cb = kw["stream_callback"]
            cb("response")
            cb(STOP)

        instance.add_task.side_effect = fake_add
        instance.remove_task.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        result = eng.generate("hello")
        assert result == "response"


def test_engine_generate_streaming_yields_tokens():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="tok")

    callbacks_saved = []

    def capture_cb(prompt, **kw):
        callbacks_saved.append(kw.get("stream_callback"))

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_task.side_effect = capture_cb
        instance.remove_task.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        gen = eng.generate("hello", stream=True)

        cb = callbacks_saved[0]
        cb("t1")
        cb("t2")
        cb(STOP)

        tokens = list(gen)
        assert tokens == ["t1", "t2"]


def test_engine_stream_close_cancels_unfinished_task():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="tok")
    callbacks_saved = []

    def capture_cb(prompt, **kwargs):
        callbacks_saved.append(kwargs["stream_callback"])
        return "task-1"

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_task.side_effect = capture_cb

        engine = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        stream = engine.generate("hello", stream=True)

        callbacks_saved[0]("t1")
        assert next(stream) == "t1"
        stream.close()

        instance.cancel_task.assert_called_once_with("task-1")


def test_engine_generate_async_yields_tokens_until_stop():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="tok")
    callbacks_saved = []

    def capture_cb(prompt, **kw):
        callbacks_saved.append(kw.get("stream_callback"))

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_task.side_effect = capture_cb
        instance.remove_task.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        agen = eng.generate_async("hello")

        async def collect():
            out = []
            async for token in agen:
                out.append(token)
            return out

        cb = callbacks_saved[0]
        cb("t1")
        cb("t2")
        cb(STOP)

        assert asyncio.run(collect()) == ["t1", "t2"]


def test_engine_async_close_cancels_unfinished_task():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="tok")
    callbacks_saved = []

    def capture_cb(prompt, **kwargs):
        callbacks_saved.append(kwargs["stream_callback"])
        return "task-1"

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_task.side_effect = capture_cb

        engine = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        stream = engine.generate_async("hello")
        callbacks_saved[0]("t1")

        async def consume_then_close():
            assert await anext(stream) == "t1"
            await stream.aclose()

        asyncio.run(consume_then_close())

        instance.cancel_task.assert_called_once_with("task-1")


def test_engine_generate_non_streaming_batch():
    mock_model, mock_tokenizer = _make_engine_mocks(decode="r")

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value

        def fake_add(prompt, **kw):
            cb = kw["stream_callback"]
            cb("r")
            cb(STOP)

        instance.add_task.side_effect = fake_add
        instance.remove_task.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=2)
        results = eng.generate(["hello", "world"])
        assert results == ["r", "r"]


def test_engine_generate_zero_max_tokens_returns_empty():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value
        instance.remove_task.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=2)
        assert eng.generate(["hello", "world"], max_tokens=0) == ["", ""]
        instance.add_task.assert_not_called()


def test_engine_generate_zero_max_tokens_stream_is_empty():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value
        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        assert list(eng.generate("hello", stream=True, max_tokens=0)) == []
        instance.add_task.assert_not_called()


def test_engine_exposes_release_resume_lifecycle():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        scheduler = MockSched.return_value
        scheduler.runtime_released = False
        scheduler.release.return_value = True
        scheduler.resume.return_value = True
        engine = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)

        assert engine.runtime_released is False
        assert engine.release() is True
        assert engine.resume() is True
        scheduler.release.assert_called_once_with()
        scheduler.resume.assert_called_once_with()


def test_engine_passes_backend_to_scheduler():
    mock_model, mock_tokenizer = _make_engine_mocks()

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        InferenceEngine(
            mock_model,
            mock_tokenizer,
            max_batch_size=1,
            backend="torch_native",
        )

    assert MockSched.call_args.kwargs["backend"] == "torch_native"


def test_generate_captures_calling_backend_context():
    mock_model, mock_tokenizer = _make_engine_mocks()
    captured = []

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value

        def fake_add(prompt, **kwargs):
            captured.append(kwargs["backend"])
            kwargs["stream_callback"](STOP)
            return "task"

        instance.add_task.side_effect = fake_add
        engine = InferenceEngine(mock_model, mock_tokenizer)
        with attn_backend("torch_native"):
            assert engine.generate("hello") == ""

    assert len(captured) == 1
    assert isinstance(captured[0], TorchNativeBackend)
