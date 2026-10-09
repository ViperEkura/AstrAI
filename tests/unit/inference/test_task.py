"""Unit tests for Request and RequestManager."""

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
