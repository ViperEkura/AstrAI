"""Tests for scheduler concurrency."""

import threading
import time
import uuid
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch

from astrai.extension.runtime.context import use_op_selection
from astrai.inference import EngineCore
from astrai.inference.contracts import SchedulerOutput
from astrai.inference.core.events import RequestFinished, TokenDelta
from astrai.inference.core.request import Request
from astrai.inference.worker.pending import (
    PendingExecution,
)
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import make_rollout_config
from tests.support.scheduler import (
    _make_contract_scheduler,
    _make_real_scheduler,
    _UnwrappingTokenizer,
)
from tests.support.tokenizers import FakeTokenizer


def test_pending_step_commit_is_idempotent(device):
    """A committed step cannot double-append tokens to its requests."""
    tokens = torch.tensor([5, 6], dtype=torch.long)
    logprobs = torch.tensor([-1.5, -2.5], dtype=torch.float32)
    entries = tuple(Request(rid, [1]).execution("prefill", 0, 1) for rid in ("a", "b"))
    pending = PendingExecution(
        snapshot=SchedulerOutput(7, 3, entries),
        sampled_identities=tuple(r.identity for r in entries),
        tokens=tokens,
        logprobs=logprobs,
    )
    first = pending.commit()
    second = pending.commit()
    assert first is second
    assert [(r.token_id, r.logprob) for r in first.results] == [(5, -1.5), (6, -2.5)]
    assert pending.committed


def test_submit_decode_returns_pending_without_touching_tasks(device):
    """The submit half leaves request output state untouched.

    Core of the B1 contract: after submit, no token is appended — the
    scheduler may still roll the batch back.  (The KV write cursor is a
    submit-side property — see SchedulerStep._submit_decoded — because the
    launched work owns its write slot; commit only materialises results.)
    """
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30, 40]]
        scheduler.generate(prompts, max_tokens=2, temperature=0.7)
        requests = [
            Request(
                request_id=f"probe_{uuid.uuid4().hex[:8]}",
                prompt_ids=[10, 20, 30, 40],
                max_tokens=4,
                temperature=0.7,
            )
        ]
        request = requests[0]
        if not scheduler.scheduler.kv_manager.alloc_slots(
            request.request_id, request.prompt_ids
        ):
            pytest.skip("KV allocation failed")
        request.input_tokens = len(request.prompt_ids)
        # Prefill through the real stepper path so KV state is complete.
        with use_op_selection(scheduler.op_selection):
            scheduler.execute_synchronously([request])
            assert request.prefill_complete
            plan = scheduler.scheduler.schedule([request])
            pending = scheduler.model_runner.submit_decode(
                list(plan.requests), plan=plan
            )
        assert pending is not None
        try:
            n_prefilled = len(request.output_ids)
            assert request.output_tokens == n_prefilled
            scheduler.resolve(pending)
            assert len(request.output_ids) == n_prefilled + 1
            assert request.output_tokens == n_prefilled + 1
        finally:
            scheduler.scheduler.kv_manager.free_slots(request.request_id)
    finally:
        scheduler.stop()


def test_online_loop_overlap_generates_and_admits_midstream(device):
    """End-to-end online generation rides the overlap pipeline.

    A real model serves one request through the scheduler thread, then a
    second request joins mid-decode.  The overlap branch (submit current,
    commit previous) must deliver every token of both requests in order,
    handle the mid-run batch change through its drain-and-sync fallback,
    and leave no in-flight step behind at stop.
    """
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tok = FakeTokenizer()
    scheduler = EngineCore(
        model=model,
        tokenizer=tok,
        max_batch_size=8,
        max_seq_len=64,
        enable_overlap=True,
    )

    scheduler.scheduler._requests.tokenizer = _UnwrappingTokenizer(tok)

    token_counts: Dict = {"first": 0, "second": 0}
    done: Dict = {"first": threading.Event(), "second": threading.Event()}
    tag_by_prompt = {"a" * 8: "first", "b" * 8: "second"}
    sink_rids: Dict = {}

    def sink(events):
        for event in events:
            tag = tag_by_prompt.get(
                next(
                    (p for p, rid in sink_rids.items() if rid == event.request_id), None
                )
            )
            if tag is None:
                return
            if isinstance(event, TokenDelta):
                token_counts[tag] += 1
            elif isinstance(event, RequestFinished):
                done[tag].set()

    scheduler.set_event_sink(sink)

    try:
        scheduler.start()
        rid_a = scheduler.add_request("a" * 8, max_tokens=12)
        sink_rids["a" * 8] = rid_a
        time.sleep(0.2)  # let the first request enter steady decode
        rid_b = scheduler.add_request("b" * 8, max_tokens=12)
        sink_rids["b" * 8] = rid_b
        assert done["first"].wait(timeout=10)
        assert done["second"].wait(timeout=10)
        assert token_counts["first"] == 12
        assert token_counts["second"] == 12
        # The overlap loop may still hold the FINAL submitted step in the
        # executor slot (the step launched past the terminal one) for one
        # more iteration. Wait for the drain instead of racing it.
        deadline = time.time() + 5
        while scheduler.model_runner.peek_pending() is not None:
            if time.time() > deadline:
                pytest.fail("overlap loop left a step pending after finish")
            time.sleep(0.01)
        stats = scheduler.get_stats()
        assert stats["in_flight_tasks"] == 0
        assert stats["kv_cache_tasks"] == 0
    finally:
        scheduler.stop()


def test_overlap_loop_matches_synchronous_tokens(device):
    """Overlap decode commits every step: token stream identical to sync.

    Regression gate for the clear_pending bug: the steady overlap branch
    used to detach the JUST-SUBMITTED pending step (the executor slot
    already held the new step, not the one being committed), so every
    other steady iteration's tokens never committed -- requests finished at
    the KV cap with half their tokens and interleaved-token detext.
    Token-count gates cannot see this on small configs (the requests still
    finish); only the full sequence comparison against the synchronous
    generate reference catches it.
    """
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = FakeTokenizer()
    scheduler = EngineCore(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
        enable_overlap=True,
    )

    scheduler.scheduler._requests.tokenizer = _UnwrappingTokenizer(tokenizer)

    prompts = ["a" * 8, "b" * 8, "c" * 8]
    scheduler.start()
    try:
        # Synchronous reference: same model, same prompts, greedy.  The tiny
        # random model may sample a stop id (FakeTokenizer: eos=2) mid-stream,
        # so accept an early stop-token termination on either side of the
        # comparison; what must match is the committed token stream itself.
        reference = scheduler.generate(
            [[ord(c) for c in p] for p in prompts],
            max_tokens=12,
            temperature=0,
        )
        stop_ids = set(scheduler.scheduler.stop_ids)
        assert all(
            len(result.token_ids) == 12
            or (result.token_ids and result.token_ids[-1] in stop_ids)
            for result in reference
        )

        # Collect the committed token stream through the event sink before
        # release_finished discards the finished requests.
        stream: Dict = {}
        lock = threading.Lock()

        def sink(events):
            with lock:
                for event in events:
                    if isinstance(event, TokenDelta):
                        stream.setdefault(event.request_id, []).append(event.token_id)

        scheduler.set_event_sink(sink)

        request_ids = [
            scheduler.add_request(p, max_tokens=12, temperature=0) for p in prompts
        ]
        deadline = time.time() + 30
        while time.time() < deadline:
            with lock:
                done = all(
                    len(stream.get(tid, [])) >= 12
                    or (
                        stream.get(tid, [None])[-1] in stop_ids
                        and not scheduler.scheduler._requests.get_request(tid)
                    )
                    for tid in request_ids
                )
            if done:
                break
            time.sleep(0.01)
        # Let the loop's release path drain fully before asserting.
        deadline = time.time() + 5
        while (
            time.time() < deadline and scheduler.model_runner.peek_pending() is not None
        ):
            time.sleep(0.01)

        by_task = {tid: stream.get(tid, []) for tid in request_ids}
        for i, tid in enumerate(request_ids):
            assert by_task[tid] == reference[i].token_ids, (
                f"request {i} overlap stream {by_task[tid]} != reference "
                f"{reference[i].token_ids}"
            )
        assert scheduler.model_runner.peek_pending() is None
    finally:
        scheduler.stop()


@pytest.mark.parametrize(
    "temperature,frequency_penalty",
    [(0.0, 0.0), (0.7, 0.0), (0.7, 0.5)],
    ids=["greedy", "seeded", "frequency-penalty"],
)
def test_rollout_overlap_matches_sync_with_queued_prefill_and_shrinking_batch(
    temperature, frequency_penalty, monkeypatch
):
    """Rollouts use overlap without changing samples or raw policy logprobs."""
    torch.manual_seed(3407)
    config = make_rollout_config(
        vocab_size=32, max_position_embeddings=14, num_hidden_layers=1
    )
    model = AutoRegressiveLM(config).to(device="cpu", dtype=torch.float32).eval()
    tokenizer = FakeTokenizer()
    tokenizer.stop_ids = []
    core = EngineCore(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_batch_size=2,
        max_seq_len=14,
        enable_cuda_graph=False,
        enable_overlap=False,
    )
    prompts = [[3, 4, 5], [6, 7, 6, 8, 6, 9, 6, 7, 6], [10, 11, 12, 13]]
    kwargs = dict(
        max_tokens=7,
        temperature=temperature,
        top_k=5,
        top_p=0.9,
        frequency_penalty=frequency_penalty,
        rep_window=3,
        seeds=[11, 22, 33],
        return_logprobs=True,
    )
    submissions = []
    submit = core.submit

    def record_submit(plan):
        submissions.append(
            (tuple(request.phase for request in plan.requests), len(core._pending))
        )
        return submit(plan)

    try:
        reference = core.generate(prompts, **kwargs)
        core.enable_overlap = True
        monkeypatch.setattr(core, "submit", record_submit)
        actual = core.generate(prompts, **kwargs)

        assert [len(result.token_ids) for result in actual] == [7, 5, 7]
        for prompt, expected, result in zip(prompts, reference, actual):
            assert result.token_ids == expected.token_ids
            assert result.finish_reason == expected.finish_reason == "length"
            assert result.error_reason is None
            torch.testing.assert_close(
                torch.tensor(result.logprobs),
                torch.tensor(expected.logprobs),
                rtol=1e-5,
                atol=1e-6,
            )
            # Temperature, top-k/p and penalties must not transform the
            # logprobs consumed by PPO/GRPO's policy-ratio calculation.
            with torch.no_grad(), use_op_selection(core.op_selection):
                inputs = torch.tensor([prompt + result.token_ids[:-1]])
                logits = model(inputs)["logits"][0, len(prompt) - 1 :]
                raw = (
                    logits.float()
                    .log_softmax(-1)
                    .gather(-1, torch.tensor(result.token_ids).unsqueeze(-1))
                )
            torch.testing.assert_close(
                torch.tensor(result.logprobs),
                raw.squeeze(-1),
                rtol=1e-5,
                atol=1e-6,
            )

        # The third prompt must enter after decoding has started, and the
        # final request must continue after the batch shrinks to one row.
        first_decode = next(
            index for index, (phases, _) in enumerate(submissions) if "decode" in phases
        )
        assert any("prefill" in phases for phases, _ in submissions[first_decode + 1 :])
        assert any(phases == ("decode",) for phases, _ in submissions)
        if frequency_penalty:
            assert all(pending == 0 for _, pending in submissions)
        else:
            assert any(pending == 1 for _, pending in submissions)

        assert not core._pending
        assert core.model_runner.peek_pending() is None
        assert not core.scheduler.has_outstanding()
        assert not core.scheduler.has_requests()
        assert core.get_stats()["kv_cache_tasks"] == 0

        def update(_version):
            with torch.no_grad():
                model.lm_head.weight.mul_(0.9)

        core.apply_weight_update(1, update)
        assert core.policy_version == 1
        assert (
            len(core.generate([prompts[0]], max_tokens=2, temperature=0)[0].token_ids)
            == 2
        )
        assert not core._pending
        assert core.get_stats()["kv_cache_tasks"] == 0
    finally:
        core.stop()


def test_stop_leaves_state_intact_when_loop_does_not_drain():
    """stop() must not clear queues under a live loop (KV double-free race).

    A loop stuck longer than the 2s join used to be followed blindly by
    _abort_and_clear + handle reset: the still-running thread would then
    free the same slots again, and a second start() could race the old
    loop.  The fixed contract: on drain failure stop() keeps the handle
    and the request state, and start() refuses to spawn a second loop.
    """
    scheduler = _make_contract_scheduler()
    scheduler.loop_thread = MagicMock()
    scheduler.loop_thread.is_alive.return_value = True
    scheduler.release_finished = MagicMock()

    assert scheduler.stop() is False
    assert scheduler.loop_thread is not None
    scheduler.release_finished.assert_not_called()

    with patch("astrai.inference.core.engine_core.threading.Thread") as TH:
        scheduler.start()
        TH.assert_not_called()


def test_stop_clears_state_when_loop_drains_normally():
    """After a clean drain the handle resets and terminal cleanup runs."""
    scheduler = _make_contract_scheduler()
    scheduler.add_request("waiting", max_tokens=8)
    scheduler.loop_thread = MagicMock()
    scheduler.loop_thread.is_alive.return_value = True

    def join_side_effect(timeout=None):
        scheduler.loop_thread.is_alive.return_value = False

    scheduler.loop_thread.join.side_effect = join_side_effect
    assert scheduler.stop() is True
    assert scheduler.loop_thread is None
    assert not scheduler.scheduler._requests.has_requests()
    assert not scheduler.scheduler._states
