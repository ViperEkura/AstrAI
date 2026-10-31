"""Unit tests for Request and Scheduler."""

from unittest.mock import Mock, call

import pytest

from astrai.inference import (
    Request,
    RequestStatus,
)
from tests.support.task import _make_mock_tokenizer, _make_scheduler


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


@pytest.mark.parametrize("encoded", [[1, 2, 3], [[1, 2, 3]]])
@pytest.mark.parametrize("request_id", [None, "", "supplied-id"])
def test_single_request_preserves_scalar_encoding_and_request_id(encoded, request_id):
    tokenizer = _make_mock_tokenizer()

    def encode(prompt):
        assert isinstance(prompt, str)
        return encoded

    tokenizer.encode.side_effect = encode
    manager = _make_scheduler(tokenizer, max_seq_len=5)
    actual_id = manager.add_request("hello", request_id=request_id)

    tokenizer.encode.assert_called_once_with("hello")
    if request_id:
        assert actual_id == request_id
    else:
        assert actual_id.startswith("req_")
    request = manager._states[actual_id]
    assert request.prompt_ids == [1, 2, 3]
    assert request.max_tokens == 2
    assert list(manager.waiting) == [request]
    stats = manager.get_stats()
    assert stats["total_tasks"] == 1
    assert stats["waiting"] == 1
    assert stats["running"] == 0


@pytest.mark.parametrize(("max_tokens", "expected"), [(None, 2), (1, 1), (10, 2)])
def test_single_tokenized_request_keeps_budget_and_sampling(max_tokens, expected):
    tokenizer = _make_mock_tokenizer()
    manager = _make_scheduler(tokenizer, max_seq_len=5)
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
    request = manager._states[request_id]
    assert request.max_tokens == expected
    assert (
        request.sampling.temperature,
        request.sampling.top_p,
        request.sampling.top_k,
    ) == (0.7, 0.8, 4)
    assert (request.sampling.frequency_penalty, request.sampling.rep_window) == (0.2, 7)


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
def test_scheduler_duplicate_registration_is_atomic(method, request_ids):
    metrics = Mock()
    metrics.get_stats.return_value = {}
    tm = _make_scheduler(tokenizer=_make_mock_tokenizer(), metrics=metrics)
    tm.add_request("original", request_id="existing")
    original = tm._states["existing"]
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
    assert tm._states == {"existing": original}
    assert tm.get_stats()["total_tasks"] == 1
    assert not tm._request_event.is_set()
    metrics.register.assert_not_called()


def test_scheduler_batch_encodes_once_and_registers_each_request_once():
    tokenizer = _make_mock_tokenizer()
    tokenizer.encode.return_value = [[1, 2], [3, 4]]
    metrics = Mock()
    metrics.get_stats.return_value = {}
    tm = _make_scheduler(tokenizer=tokenizer, metrics=metrics)

    assert tm.add_requests(["first", "second"], request_ids=["a", "b"]) == ["a", "b"]

    tokenizer.encode.assert_called_once_with(["first", "second"])
    assert [request.request_id for request in tm.waiting] == ["a", "b"]
    assert metrics.register.call_args_list == [call("a"), call("b")]
    assert tm.get_stats()["total_tasks"] == 2
    assert tm._request_event.is_set()


def test_empty_prompt_later_in_batch_does_not_register_valid_prefix():
    tokenizer = _make_mock_tokenizer()
    metrics = Mock()
    manager = _make_scheduler(tokenizer, metrics=metrics)
    with pytest.raises(ValueError, match="zero tokens"):
        manager.add_requests(
            ["valid", "empty"],
            request_ids=["first", "second"],
            prompts_ids=[[1], []],
        )

    tokenizer.encode.assert_not_called()
    metrics.register.assert_not_called()
    assert not manager.waiting and not manager._states
    assert manager._total_requests == 0
    assert not manager._request_event.is_set()


def test_scheduler_long_prompt_truncates_and_finishes_at_sequence_limit():
    tokenizer = _make_mock_tokenizer()
    tokenizer.encode.return_value = list(range(9000))
    scheduler = _make_scheduler(tokenizer, max_seq_len=16)
    events = []
    scheduler.set_event_sink(events.extend)
    scheduler.add_request("long")
    assert len(events) == 1
    assert events[0].prompt_tokens == 16
    assert events[0].completion_tokens == 0
    assert not scheduler.has_requests()


def test_scheduler_remove_request():
    tm = _make_scheduler(tokenizer=_make_mock_tokenizer())
    tid = tm.add_request("test")
    tm.cancel_request(tid)
    assert len(tm.waiting) == 0


def test_scheduler_cancel_preserves_inflight_request_until_retirement():
    scheduler = _make_scheduler()
    rid = scheduler.add_request("test")
    scheduler.admit_requests()
    request = scheduler.running[0]
    plan = scheduler.schedule()
    assert scheduler.cancel_request(rid)
    assert not scheduler.cancel_request(rid)
    assert request.status == RequestStatus.ABORTED
    assert scheduler.running == [request]
    assert scheduler.kv_manager.request_count == 1
    with pytest.raises(ValueError, match="duplicate"):
        scheduler.add_request("replacement", request_id=rid)
    scheduler.rollback_schedule(plan)
    scheduler.release_finished()
    assert not scheduler.running and not scheduler._states
    assert scheduler.kv_manager.request_count == 0
    assert scheduler.get_stats()["cancelled_total"] == 1
    scheduler.add_request("replacement", request_id=rid)


def test_scheduler_admits_waiting_requests_in_order():
    scheduler = _make_scheduler(max_batch_size=2)
    ids = [scheduler.add_request(text) for text in ("a", "b", "c")]
    scheduler.admit_requests()
    assert [r.request_id for r in scheduler.running] == ids[:2]
    assert [r.request_id for r in scheduler.waiting] == ids[2:]


def test_scheduler_preserves_waiting_order_under_cache_pressure():
    scheduler = _make_scheduler()
    ids = [scheduler.add_request(text) for text in ("a", "b")]
    allocate = scheduler.kv_manager.alloc_slots
    scheduler.kv_manager.alloc_slots = Mock(return_value=False)
    scheduler.admit_requests()
    assert not scheduler.running
    assert [r.request_id for r in scheduler.waiting] == ids
    scheduler.kv_manager.alloc_slots = allocate
    scheduler.admit_requests()
    assert [r.request_id for r in scheduler.running] == ids
    assert not scheduler.waiting


def test_scheduler_has_work():
    tm = _make_scheduler(tokenizer=_make_mock_tokenizer())
    assert not tm.has_requests()
    tm.add_request("test")
    assert tm.has_requests()


def test_scheduler_add_task_rejects_empty_prompt():
    tm = _make_scheduler(tokenizer=_make_mock_tokenizer())
    tm.tokenizer.encode.return_value = []

    with pytest.raises(ValueError, match="zero tokens"):
        tm.add_request("")


def test_scheduler_cancel_unknown_request_is_noop():
    scheduler = _make_scheduler()
    assert not scheduler.cancel_request("does-not-exist")
    assert scheduler.get_stats()["cancelled_total"] == 0


def test_scoring_and_generation_share_duplicate_id_validation():
    scheduler = _make_scheduler()
    scored = Request("scored", [1, 2])
    assert scheduler.reserve_tokenized_request(scored)
    with pytest.raises(ValueError, match="duplicate"):
        scheduler.reserve_tokenized_request(Request("scored", [3]))
    with pytest.raises(ValueError, match="duplicate"):
        scheduler.enqueue_tokenized_requests(
            [Request("new", [3]), Request("scored", [4])]
        )
    assert scheduler._states == {"scored": scored}
    assert not scheduler.waiting
    assert scheduler.kv_manager.request_count == 1
    scheduler.release_tokenized_request(scored)
    assert not scheduler._states
    assert scheduler.kv_manager.request_count == 0
    scheduler.add_request("generation", request_id="scored")
