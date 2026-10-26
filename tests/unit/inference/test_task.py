"""Unit tests for Request and RequestManager."""

from unittest.mock import Mock, call

import pytest

from astrai.inference import (
    Request,
    RequestManager,
    RequestStatus,
)
from tests.support.task import _make_mock_tokenizer


def test_task_default_status_is_pending():
    request = Request("id1", [1, 2, 3])
    assert request.status == RequestStatus.PENDING


def test_task_next_pos():
    request = Request("id1", [1, 2, 3])
    request.input_tokens = 5
    request.num_computed_tokens = 5
    assert request.next_pos == 5
    request.num_computed_tokens += 1
    assert request.next_pos == 6
    request.num_computed_tokens += 1
    assert request.next_pos == 7


def test_task_is_finished_max_tokens():
    request = Request("id1", [1, 2, 3], max_tokens=2)
    request.output_tokens = 2
    assert request.is_finished([])


def test_task_is_finished_stop_id():
    request = Request("id1", [1, 2, 3])
    request.output_ids = [5, 0]
    assert request.is_finished([0])


def test_task_is_finished_not_yet():
    request = Request("id1", [1, 2, 3], max_tokens=10)
    request.output_ids = [1, 2]
    assert not request.is_finished([0])


def test_task_manager_add_task():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_request("hello")
    assert tid.startswith("req_")
    assert tm._total_requests == 1
    assert len(tm.waiting) == 1


@pytest.mark.parametrize("encoded", [[1, 2, 3], [[1, 2, 3]]])
@pytest.mark.parametrize("request_id", [None, "", "supplied-id"])
def test_single_request_preserves_scalar_encoding_and_request_id(encoded, request_id):
    tokenizer = _make_mock_tokenizer()

    def encode(prompt):
        assert isinstance(prompt, str)
        return encoded

    tokenizer.encode.side_effect = encode
    manager = RequestManager(tokenizer, max_seq_len=5)
    actual_id = manager.add_request("hello", request_id=request_id)

    tokenizer.encode.assert_called_once_with("hello")
    if request_id:
        assert actual_id == request_id
    else:
        assert actual_id.startswith("req_")
    request = manager.get_request(actual_id)
    assert request.prompt_ids == [1, 2, 3]
    assert request.max_tokens == 2
    assert manager.get_stats()["total_tasks"] == 1


@pytest.mark.parametrize(("max_tokens", "expected"), [(None, 2), (1, 1), (10, 2)])
def test_single_tokenized_request_keeps_budget_and_sampling(max_tokens, expected):
    tokenizer = _make_mock_tokenizer()
    manager = RequestManager(tokenizer, max_seq_len=5)
    request_id = manager.add_request(
        "unused text",
        prompt_ids=[1, 2, 3],
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=4,
        frequency_penalty=0.2,
        rep_window=7,
    )

    tokenizer.encode.assert_not_called()
    request = manager.get_request(request_id)
    assert request.max_tokens == expected
    assert (request.temperature, request.top_p, request.top_k) == (0.7, 0.8, 4)
    assert (request.frequency_penalty, request.rep_window) == (0.2, 7)


@pytest.mark.parametrize(
    ("method", "request_ids"),
    [
        ("single", ["existing"]),
        ("batch", ["new", "existing"]),
        ("batch", ["new", "new"]),
        ("tokenized", ["new", "existing"]),
        ("tokenized", ["new", "new"]),
    ],
)
def test_task_manager_duplicate_registration_is_atomic(method, request_ids):
    metrics = Mock()
    metrics.get_stats.return_value = {}
    tm = RequestManager(tokenizer=_make_mock_tokenizer(), metrics=metrics)
    tm.add_request("original", request_id="existing")
    original = tm.get_request("existing")
    metrics.reset_mock()
    tm._request_event.clear()

    with pytest.raises(ValueError, match="duplicate"):
        if method == "single":
            tm.add_request("duplicate", request_id=request_ids[0])
        elif method == "batch":
            tm.add_requests(
                ["prompt"] * len(request_ids),
                request_ids=request_ids,
                prompts_ids=[[1]] * len(request_ids),
            )
        else:
            tm.enqueue_tokenized_requests(
                [Request(request_id, [1]) for request_id in request_ids]
            )

    assert list(tm.waiting) == [original]
    assert tm._requests == {"existing": original}
    assert tm.get_stats()["total_tasks"] == 1
    assert not tm._request_event.is_set()
    metrics.register.assert_not_called()


def test_task_manager_batch_encodes_once_and_registers_each_request_once():
    tokenizer = _make_mock_tokenizer()
    tokenizer.encode.return_value = [[1, 2], [3, 4]]
    metrics = Mock()
    metrics.get_stats.return_value = {}
    tm = RequestManager(tokenizer=tokenizer, metrics=metrics)

    assert tm.add_requests(["first", "second"], request_ids=["a", "b"]) == ["a", "b"]

    tokenizer.encode.assert_called_once_with(["first", "second"])
    assert [request.request_id for request in tm.waiting] == ["a", "b"]
    assert metrics.register.call_args_list == [call("a"), call("b")]
    assert tm.get_stats()["total_tasks"] == 2
    assert tm._request_event.is_set()


def test_empty_prompt_later_in_batch_does_not_register_valid_prefix():
    tokenizer = _make_mock_tokenizer()
    metrics = Mock()
    manager = RequestManager(tokenizer, metrics=metrics)
    with pytest.raises(ValueError, match="zero tokens"):
        manager.add_requests(
            ["valid", "empty"],
            request_ids=["first", "second"],
            prompts_ids=[[1], []],
        )

    tokenizer.encode.assert_not_called()
    metrics.register.assert_not_called()
    assert not manager.waiting and not manager._requests
    assert manager._total_requests == 0
    assert not manager._request_event.is_set()


def test_task_manager_long_prompt_truncated_not_stopped():
    t = _make_mock_tokenizer()
    t.encode.return_value = list(range(9000))

    tm = RequestManager(tokenizer=t, max_seq_len=16)
    tm.add_request("long")
    assert len(tm.waiting) == 1
    assert len(tm.waiting[0].prompt_ids) == 16


def test_task_manager_remove_request():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_request("test")
    tm.cancel_request(tid)
    assert len(tm.waiting) == 0


def test_task_manager_cancel_active_task_defers_removal():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_request("test")
    requests = tm.pull_waiting(1)
    tm.activate(requests[0])
    assert len(tm.running) == 1
    immediate, cancelled = tm.cancel_request(tid)
    assert cancelled is True
    assert immediate == []
    assert tm.running[0].status == RequestStatus.ABORTED

    tm.discard(requests[0])
    assert len(tm.running) == 0
    assert tm.get_stats()["cancelled_total"] == 1


def test_task_manager_pull_candidates_fifo():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("a")
    tm.add_request("b")
    tm.add_request("c")
    pulled = tm.pull_waiting(2)
    assert len(pulled) == 2
    assert pulled[0].prompt_ids == [1, 2, 3, 4, 5]
    assert len(tm.waiting) == 1


def test_task_manager_activate():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("test")
    request = tm.pull_waiting(1)[0]
    tm.activate(request)
    assert request.status == RequestStatus.RUNNING
    assert request in tm.running


def test_task_manager_return_to_waiting():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("a")
    tm.add_request("b")
    t1 = tm.pull_waiting(1)[0]
    tm.return_to_waiting([t1])
    assert len(tm.waiting) == 2
    assert tm.waiting[0] == t1


def test_task_manager_has_work():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    assert not tm.has_requests()
    tm.add_request("test")
    assert tm.has_requests()


def test_task_manager_get_stats():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.add_request("test")
    stats = tm.get_stats()
    assert stats["total_tasks"] == 1
    assert stats["waiting"] == 1
    assert stats["running"] == 0


def test_task_manager_add_task_rejects_empty_prompt():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    tm.tokenizer.encode.return_value = []

    with pytest.raises(ValueError, match="zero tokens"):
        tm.add_request("")


def test_task_manager_cancel_unknown_request_is_noop():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    immediate, cancelled = tm.cancel_request("does-not-exist")
    assert not cancelled and immediate == []


def test_task_manager_cancel_running_request_defers_removal():
    tm = RequestManager(tokenizer=_make_mock_tokenizer())
    request_id = tm.add_request("test")
    request = tm._requests[request_id]
    tm.waiting.clear()
    tm.running.append(request)
    request.status = RequestStatus.RUNNING

    immediate, cancelled = tm.cancel_request(request_id)
    assert cancelled and immediate == []
    assert request.status == RequestStatus.ABORTED
