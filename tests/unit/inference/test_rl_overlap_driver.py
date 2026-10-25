"""CPU gates for the shared serving and synchronous rollout step driver."""

from types import SimpleNamespace

import pytest
import torch

from astrai.inference.core.request import Request, RequestStatus
from astrai.inference.worker.pending import PendingExecution
from tests.support.inference import make_cpu_model, make_cpu_scheduler


def _record_execution(monkeypatch, engine, calls, token_for=None):
    plans = []

    def execute(plan):
        plans.append(plan)
        calls.append(("submit", plan.step_id))
        sampled = tuple(entry for entry in plan.requests if entry.samples)
        pending = PendingExecution(
            plan,
            tokens=torch.tensor(
                [
                    token_for(entry) if token_for else entry.position + 10
                    for entry in sampled
                ],
                dtype=torch.long,
            ),
            logprobs=torch.full((len(sampled),), -0.25)
            if plan.return_logprobs
            else None,
            sampled_identities=tuple(entry.identity for entry in sampled),
            completion_event=SimpleNamespace(
                synchronize=lambda: calls.append(("commit", plan.step_id))
            ),
        )
        engine.model_runner._pending = pending
        yield pending

    monkeypatch.setattr(engine.model_runner, "execute_model", execute)
    return plans


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("return_logprobs", [False, True])
def test_generate_uses_shared_pipeline_and_drains(
    monkeypatch, overlap, return_logprobs
):
    engine = make_cpu_scheduler(make_cpu_model(), enable_overlap=overlap)
    calls = []
    plans = _record_execution(monkeypatch, engine, calls)
    depths = []
    tick = engine.tick

    def observe_tick(**kwargs):
        result = tick(**kwargs)
        depths.append(len(engine.pending))
        return result

    monkeypatch.setattr(engine, "tick", observe_tick)
    outputs = engine.generate(
        [[1, 2]],
        max_tokens=6,
        return_logprobs=return_logprobs,
    )

    assert depths and max(depths) == int(overlap)
    assert all(plan.return_logprobs == return_logprobs for plan in plans)
    tokens = outputs[0].token_ids
    assert tokens == [10, 12, 13, 14, 15, 16]
    assert outputs[0].logprobs == ([-0.25] * len(tokens) if return_logprobs else [])
    assert outputs[0].finish_reason == "length"
    assert outputs[0].error_reason is None
    submit_next = calls.index(("submit", 3))
    commit_prior = calls.index(("commit", 2))
    assert (submit_next < commit_prior) == overlap
    assert {step for kind, step in calls if kind == "submit"} == {
        step for kind, step in calls if kind == "commit"
    }
    assert not engine.pending
    assert engine.model_runner.peek_pending() is None
    assert not engine.scheduler.has_outstanding()
    assert engine.scheduler.kv_manager.request_count == 0


def test_overlap_replan_preserves_logprobs_when_kv_extension_shrinks_batch(
    monkeypatch,
):
    engine = make_cpu_scheduler(make_cpu_model(), enable_overlap=True)
    calls = []
    submitted = _record_execution(monkeypatch, engine, calls)
    scheduled = []
    schedule = engine.scheduler.schedule

    def record_schedule(*args, **kwargs):
        plan = schedule(*args, **kwargs)
        scheduled.append(plan)
        return plan

    monkeypatch.setattr(engine.scheduler, "schedule", record_schedule)
    extend = engine.scheduler.kv_manager.extend_slots_batch
    extensions = 0

    def shrink_second_decode(ids, positions):
        nonlocal extensions
        extensions += 1
        if extensions >= 2 and "b" in ids:
            return [
                False if rid == "b" else extend([rid], [position])[0]
                for rid, position in zip(ids, positions)
            ]
        return extend(ids, positions)

    monkeypatch.setattr(
        engine.scheduler.kv_manager, "extend_slots_batch", shrink_second_decode
    )
    requests = [Request(rid, [1, 2], max_tokens=6) for rid in ("a", "b")]

    engine._execute_requests(requests, return_logprobs=True)

    assert [plan.step_id for plan in scheduled[:4]] == [1, 2, 3, 4]
    assert [plan.step_id for plan in submitted[:3]] == [1, 2, 4]
    assert all(plan.return_logprobs for plan in scheduled)
    assert len(requests[0].output_ids) == len(requests[0].output_logprobs) == 6
    assert requests[1].output_ids == [10, 12]
    assert requests[1].output_logprobs == [-0.25, -0.25]
    assert requests[1].status == RequestStatus.ABORTED
    assert requests[1].error_reason == "kv_cache_extension_failed"
    assert not engine.pending and not engine.scheduler.has_outstanding()
    assert engine.scheduler.kv_manager.request_count == 0


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("all_rows", [False, True], ids=["one-row", "all-rows"])
def test_pending_eos_wins_over_next_decode_kv_failure(monkeypatch, overlap, all_rows):
    engine = make_cpu_scheduler(make_cpu_model(), enable_overlap=overlap)
    calls = []
    eos_ids = {"a", "b"} if all_rows else {"a"}
    plans = _record_execution(
        monkeypatch,
        engine,
        calls,
        token_for=lambda entry: (
            2
            if entry.request_id in eos_ids and entry.position == 2
            else entry.position + 10
        ),
    )
    extend = engine.scheduler.kv_manager.extend_slots_batch
    failures = []

    def fail_after_pending_eos(ids, positions):
        result = []
        for rid, position in zip(ids, positions):
            if rid in eos_ids and position >= 3:
                failures.append(rid)
                assert ("commit", 2) not in calls
                result.append(False)
            else:
                result.append(extend([rid], [position])[0])
        return result

    monkeypatch.setattr(
        engine.scheduler.kv_manager, "extend_slots_batch", fail_after_pending_eos
    )
    free = engine.scheduler.kv_manager.free_slots

    def record_free(rid):
        calls.append(("free", rid))
        free(rid)

    monkeypatch.setattr(engine.scheduler.kv_manager, "free_slots", record_free)
    requests = [Request(rid, [1, 2], max_tokens=6) for rid in ("a", "b")]
    engine._execute_requests(requests, return_logprobs=True)

    assert set(failures) == (eos_ids if overlap else set())
    for request in requests:
        assert request.status == RequestStatus.FINISHED
        assert request.error_reason is None
        if request.request_id in eos_ids:
            assert request.output_ids == [10, 2]
            assert request.output_logprobs == [-0.25, -0.25]
            assert request.finish_reason == "stop_token"
        else:
            assert request.output_ids == [10, 12, 13, 14, 15, 16]
            assert len(request.output_logprobs) == 6
            assert request.finish_reason == "length"
        assert calls.index(("commit", 2)) < calls.index(("free", request.request_id))
    assert not any(
        entry.request_id in eos_ids and entry.position >= 3
        for plan in plans
        for entry in plan.requests
    )
    assert not engine.pending and not engine.scheduler.has_outstanding()
    assert engine.model_runner.peek_pending() is None
    assert engine.scheduler.kv_manager.request_count == 0


def test_rollout_rejects_waiting_batch_when_capacity_cannot_make_progress(
    monkeypatch,
):
    engine = make_cpu_scheduler(make_cpu_model(), enable_overlap=True)
    monkeypatch.setattr(engine.scheduler.kv_manager, "alloc_slots", lambda *_: False)
    requests = [Request("waiting", [1, 2], max_tokens=6)]

    engine._execute_requests(requests)

    assert requests[0].status == RequestStatus.ABORTED
    assert requests[0].finish_reason == "rejected"
    assert requests[0].error_reason == "kv_cache_allocation_failed"
    assert not engine.scheduler.has_requests()
    assert not engine.pending


def test_rollout_failed_submit_fences_before_releasing_pending_kv(monkeypatch):
    engine = make_cpu_scheduler(make_cpu_model(), enable_overlap=True)
    calls = []
    _record_execution(monkeypatch, engine, calls)
    execute = engine.model_runner.execute_model

    def fail_third_step(plan):
        if plan.step_id == 3:
            raise RuntimeError("forward failed before yielding its handle")
        yield from execute(plan)

    monkeypatch.setattr(engine.model_runner, "execute_model", fail_third_step)
    monkeypatch.setattr(
        engine.model_runner, "synchronize", lambda: calls.append(("fence", None))
    )
    free = engine.scheduler.kv_manager.free_slots

    def record_free(rid):
        calls.append(("free", rid))
        free(rid)

    monkeypatch.setattr(engine.scheduler.kv_manager, "free_slots", record_free)
    request = Request("failed", [1, 2], max_tokens=6)

    with pytest.raises(RuntimeError, match="forward failed"):
        engine._execute_requests([request], return_logprobs=True)

    assert calls.index(("fence", None)) < calls.index(("free", "failed"))
    assert request.status == RequestStatus.ABORTED
    assert request.finish_reason == "aborted"
    assert request.error_reason == "execution_failed"
    assert not engine.pending and not engine.scheduler.has_outstanding()
    assert engine.model_runner.peek_pending() is None
    assert engine.scheduler.kv_manager.request_count == 0
