"""Deterministic CPU gates for execution identity, lifecycle and resource ownership."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from astrai.inference.contracts import (
    ModelRunnerOutput,
    RequestIdentity,
    RequestOutput,
    SchedulerOutput,
)
from astrai.inference.core.events import RequestError, RequestFinished, TokenDelta
from astrai.inference.core.request import Request
from astrai.inference.worker.pending import PendingExecution, ResultRing
from tests.support.core_contracts import admit, output, pending
from tests.support.core_contracts import scheduler as scheduler
from tests.support.inference import make_cpu_model, make_cpu_scheduler
from tests.support.tokenizers import FakeTokenizer


def test_execution_freezes_state_and_reuses_large_immutable_inputs():
    request = Request("r", list(range(1024)), frequency_penalty=0)
    request.output_ids = [7, 8]
    first = request.execution("decode", 1025, 1)
    second = request.execution("decode", 1026, 1)
    assert first.prompt_ids is second.prompt_ids
    assert first.sampling is second.sampling
    assert first.output_ids == ()
    assert first.input_token_id == 8
    request.output_ids.append(9)
    request.prompt_ids[0] = 99
    assert first.input_token_id == 8 and first.prompt_ids[0] == 0
    with pytest.raises(FrozenInstanceError):
        first.position = 999
    freq = Request("freq", [1, 2], frequency_penalty=0.2)
    freq.output_ids = [3, 4]
    snapshot = freq.execution("decode", 3, 1)
    freq.output_ids.append(5)
    assert snapshot.output_ids == (3, 4)


def test_policy_and_reordered_result_rows_apply_by_identity_once(scheduler):
    sched, events = scheduler
    sched.update_weights(7)
    a, b = admit(sched, "a"), admit(sched, "b")
    plan = sched.schedule([a, b])
    assert plan.policy_version == 7
    rows = ModelRunnerOutput(
        plan.step_id,
        7,
        (
            RequestOutput(b.identity, plan.requests[1].materialized_end, 22),
            RequestOutput(a.identity, plan.requests[0].materialized_end, 11),
        ),
    )
    sched.update_from_output(rows)
    sched.update_from_output(rows)
    assert a.output_ids == [11] and b.output_ids == [22]
    assert [(e.request_id, e.sequence_no) for e in events] == [("b", 1), ("a", 1)]
    assert not sched._planned


def test_future_results_wait_for_prior_step(scheduler):
    sched, events = scheduler
    request = admit(sched)
    first = sched.schedule([request])
    second = sched.schedule([request])
    sched.update_from_output(output(second, 22))
    assert request.output_ids == []
    sched.update_from_output(output(first, 11))
    assert request.output_ids == [11, 22]
    assert [e.sequence_no for e in events] == [1, 2]
    assert not sched._ready and not sched._planned


def test_wrong_version_or_incarnation_does_not_consume_valid_plan(scheduler):
    sched, _events = scheduler
    request = admit(sched)
    plan = sched.schedule([request])
    valid = output(plan, 19)
    sched.update_from_output(replace(valid, policy_version=999))
    bad = replace(valid.results[0], identity=RequestIdentity("r", "stale"))
    sched.update_from_output(replace(valid, results=(bad,)))
    assert request.output_ids == [] and sched._planned
    sched.update_from_output(valid)
    assert request.output_ids == [19]


def test_reused_public_id_cannot_receive_predecessor_output(scheduler):
    sched, events = scheduler
    old = admit(sched, max_tokens=1)
    plan = sched.schedule([old])
    result = output(plan, 17)
    sched.update_from_output(result)
    sched.release_finished()
    new = admit(sched)
    assert new.identity != old.identity
    before = len(events)
    sched.update_from_output(result)
    assert new.output_ids == [] and len(events) == before


def test_late_output_after_policy_epoch_change_is_reclaimed_not_applied(scheduler):
    sched, events = scheduler
    request = admit(sched)
    plan = sched.schedule([request])
    # Simulate a world transition; the real guard prevents it while queued.
    sched._policy_guard._policy_version = 1
    sched.update_from_output(output(plan))
    sched.release_finished()
    assert request.output_ids == []
    assert request.error_reason == "stale_execution"
    assert len(events) == 1 and events[0].finish_reason == "aborted"
    assert sched._kv_manager.request_count == 0


@pytest.mark.parametrize("active", [False, True])
def test_cancel_has_one_event_terminal_and_no_late_token(scheduler, active):
    sched, events = scheduler
    rid = sched.add_request("input", request_id="r", prompt_ids=[10, 11], max_tokens=8)
    if active:
        sched.admit_requests()
        plan = sched.schedule()
        sched.engine_core._pending.append(pending(plan))
    assert sched.cancel_request(rid)
    assert not sched.cancel_request(rid)
    assert len(events) == 1 and events[0].finish_reason == "cancelled"
    if active:
        assert sched._kv_manager.request_count == 1  # pending still owns KV
    sched.engine_core.drain()
    sched.release_finished()
    sched.stop()
    assert len(events) == 1
    assert sched._kv_manager.request_count == 0


@pytest.mark.parametrize("limit,prompt_len", [(0, 2), (-1, 2), (8, 64), (None, 64)])
def test_zero_output_admission_terminates_without_forward(scheduler, limit, prompt_len):
    sched, events = scheduler
    with patch.object(sched._executor, "execute_model") as execute:
        rid = sched.add_request("input", prompt_ids=[1] * prompt_len, max_tokens=limit)
        execute.assert_not_called()
    assert len(events) == 1
    assert events[0] == RequestFinished(rid, "length", prompt_len, 0)
    assert not sched._requests.has_requests()
    assert sched._kv_manager.request_count == 0


def test_eos_discards_later_inflight_output_but_retires_all_resources(scheduler):
    sched, events = scheduler
    request = admit(sched)
    first, second = sched.schedule([request]), sched.schedule([request])
    sched.update_from_output(output(first, 0))
    sched.release_finished()
    assert sched._kv_manager.request_count == 1
    sched.update_from_output(output(second, 77))
    sched.release_finished()
    assert request.output_ids == [0]
    assert [type(e) for e in events] == [TokenDelta, RequestFinished]
    assert events[-1].finish_reason == "stop_token"
    assert sched._kv_manager.request_count == 0


def test_stop_drains_every_group_before_free_and_never_duplicates_terminal(scheduler):
    sched, events = scheduler
    a, b = admit(sched, "a", 1), admit(sched, "b", 1)
    plan = sched.schedule([a, b])
    calls = []
    for entry in plan.requests:
        fence = SimpleNamespace(
            synchronize=lambda rid=entry.request_id: calls.append(("fence", rid))
        )
        sched.engine_core._pending.append(pending(plan.select([entry]), fence=fence))
    original = sched._kv_manager.free_slots

    def free(rid):
        calls.append(("free", rid))
        original(rid)

    with patch.object(sched._kv_manager, "free_slots", side_effect=free):
        assert sched.stop()
        assert sched.stop()
    assert calls[:2] == [("fence", "a"), ("fence", "b")]
    assert len([e for e in events if isinstance(e, RequestFinished)]) == 2
    assert all(
        e.finish_reason == "length" for e in events if isinstance(e, RequestFinished)
    )
    assert not sched._planned and not sched.engine_core.pending


def test_bad_pending_fences_then_errors_and_frees(scheduler):
    sched, events = scheduler
    request = admit(sched)
    plan = sched.schedule([request])
    bad = PendingExecution(plan, tokens=torch.tensor([1]))  # no row identity
    sched.engine_core._pending.append(bad)
    calls = []
    free = sched._kv_manager.free_slots
    with (
        patch.object(
            sched._executor, "synchronize", side_effect=lambda: calls.append("fence")
        ),
        patch.object(
            sched._kv_manager,
            "free_slots",
            side_effect=lambda rid: (calls.append("free"), free(rid)),
        ),
    ):
        assert sched.stop()
    assert calls == ["fence", "free"]
    assert len(events) == 1 and isinstance(events[0], RequestError)
    assert sched._kv_manager.request_count == 0 and not sched._planned


def test_failed_fence_retains_kv_until_successful_retry(scheduler):
    sched, _events = scheduler
    request = admit(sched)
    plan = sched.schedule([request])
    sched.engine_core._pending.append(PendingExecution(plan, tokens=torch.tensor([1])))
    with patch.object(
        sched._executor, "synchronize", side_effect=RuntimeError("fence failed")
    ):
        with pytest.raises(RuntimeError, match="fence failed"):
            sched.stop()
        assert sched._kv_manager.request_count == 1 and sched._planned
        with pytest.raises(RuntimeError, match="drain safely"):
            sched.start()
    assert sched.stop()
    assert sched._kv_manager.request_count == 0


def test_chunk_cache_publication_uses_confirmed_not_optimistic_watermark(scheduler):
    sched, events = scheduler
    sched._stepper._token_budget = 1
    request = admit(sched, prompt=(10, 11, 12))
    first, second = sched.schedule([request]), sched.schedule([request])
    assert request.num_computed_tokens == 2
    calls = []
    fence = SimpleNamespace(synchronize=lambda: calls.append("completed"))
    p = pending(first, fence=fence)
    with patch.object(sched._kv_manager, "record_block_hashes") as record:
        sched.update_from_output(p.commit())
        assert calls == ["completed"]
        assert record.call_args.kwargs["materialized_end"] == 1
    assert request.num_materialized_tokens == 1 and events == []
    sched.update_from_output(output(second))


def test_result_ring_never_overwrites_an_unconsumed_completed_copy():
    ring = object.__new__(ResultRing)
    ring._enabled = True
    ring._slots = [
        {"in_use": True, "tokens": torch.tensor([19]), "event": MagicMock()},
        {"in_use": True, "tokens": torch.tensor([29]), "event": MagicMock()},
    ]
    entry = Request("r", [1]).execution("prefill", 0, 1)
    p = pending(SchedulerOutput(1, 0, (entry,)))
    assert ring.post(p) is False
    assert [s["tokens"].item() for s in ring._slots] == [19, 29]
    for slot in ring._slots:
        slot["event"].synchronize.assert_not_called()


def test_sync_execution_refuses_live_online_requests(scheduler):
    sched, _events = scheduler
    sched.add_request("input", prompt_ids=[10, 11])
    with pytest.raises(RuntimeError, match="queued requests"):
        sched.score_ids([[10]], [[11]])
    with pytest.raises(RuntimeError, match="queued requests"):
        sched.run_batch([[10]], max_tokens=1)


def test_overlap_rebuilds_shrunken_plan_after_committing_old_history():
    outputs = []
    for overlap in (False, True):
        model = make_cpu_model()

        def forward(ids, *, logits_positions=None, **kwargs):
            ids = ids.reshape(-1)
            if logits_positions is not None:
                ids = ids[logits_positions]
            logits = torch.full((ids.numel(), 256), -100.0)
            logits.scatter_(1, (ids + 1).unsqueeze(1), 100.0)
            return {"logits": logits}

        model.forward = forward
        tokenizer = FakeTokenizer()
        tokenizer.stop_ids = []
        sched = make_cpu_scheduler(
            model,
            tokenizer,
            max_batch_size=2,
            page_size=1,
            kv_tokens=5,
            enable_overlap=overlap,
        )
        try:
            sched.add_request(
                "a", request_id="a", prompt_ids=[1], max_tokens=5, temperature=0
            )
            sched.add_request(
                "b", request_id="b", prompt_ids=[10], max_tokens=5, temperature=0
            )
            with sched._weight_lock:
                sched.engine_core.tick()  # prefill: 2 and 11
                a = sched._states["a"]
                sched.engine_core.tick()  # one decode may stay in flight
                sched.engine_core.tick()  # only one of two extensions fits
                sched.engine_core.drain()
                outputs.append(list(a.output_ids[:3]))
        finally:
            sched.stop()
    assert outputs == [[2, 3, 4], [2, 3, 4]]


def test_online_forward_and_fence_failure_still_emits_one_error(scheduler):
    sched, events = scheduler
    sched.add_request("input", request_id="r", prompt_ids=[10, 11])
    with (
        patch.object(
            sched._executor, "execute_model", side_effect=RuntimeError("forward failed")
        ),
        patch.object(
            sched._executor, "synchronize", side_effect=RuntimeError("fence failed")
        ),
    ):
        sched.run_busy_loop()
        request = sched._states["r"]
        assert request.terminal_emitted
        assert len(events) == 1 and isinstance(events[0], RequestError)
        assert "forward failed" in events[0].message
        assert sched.engine_core._shutdown_failed
        assert sched._kv_manager.request_count == 1 and sched._planned
        sched.release_finished()
        assert sched._kv_manager.request_count == 1
        with pytest.raises(RuntimeError, match="fence failed"):
            sched.stop()
        assert len(events) == 1
    assert sched.stop()
    assert sched._kv_manager.request_count == 0
    assert len(events) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_ids": ["dup", "dup"]},
        {"prompts_ids": [[10]]},
    ],
)
def test_bad_batch_is_atomic_and_does_not_orphan_registry(scheduler, kwargs):
    sched, _events = scheduler
    options = {"prompts_ids": [[10], [11]]}
    options.update(kwargs)
    with pytest.raises(ValueError):
        sched.add_requests(["a", "b"], **options)
    assert sched._states == {}
    assert not sched._requests.has_requests()
    assert sched.stop()
    assert sched.update_weights(1) == 1


def test_duplicate_existing_batch_id_does_not_enqueue_valid_prefix(scheduler):
    sched, _events = scheduler
    rid = sched.add_request("first", request_id="live", prompt_ids=[10])
    with pytest.raises(ValueError, match="duplicate"):
        sched.add_requests(
            ["a", "b"], request_ids=["new", "live"], prompts_ids=[[10], [11]]
        )
    assert set(sched._states) == {rid}
    assert [r.request_id for r in sched._requests.waiting] == [rid]
    assert sched.cancel_request(rid)
