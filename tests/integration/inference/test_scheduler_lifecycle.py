"""Tests for scheduler concurrency."""

import threading
import time
from unittest.mock import patch

import pytest
import torch

from astrai.extension import TorchNativeBackend, get_backend
from astrai.inference import GenerationResult, Scheduler
from astrai.inference.core.scheduler import OutputEventSink
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import make_rollout_config
from tests.support.scheduler import (
    _make_mock_scheduler,
    _make_real_scheduler,
    _run_threads,
)
from tests.support.scheduler import mock_model_and_tokenizer as mock_model_and_tokenizer
from tests.support.tokenizers import FakeTokenizer


def test_scheduler_concurrent_add_task(mock_model_and_tokenizer):
    """Test concurrent add_request operations."""
    scheduler = _make_mock_scheduler(mock_model_and_tokenizer)

    results = {"request_ids": [], "errors": []}
    lock = threading.Lock()

    def add_task_worker(worker_id):
        try:
            for i in range(10):
                request_id = scheduler.add_request(
                    f"prompt from worker {worker_id}-{i}"
                )
                with lock:
                    results["request_ids"].append(request_id)
        except Exception as e:
            results["errors"].append(str(e))

    _run_threads(*(lambda wid=i: add_task_worker(wid) for i in range(5)))

    scheduler.stop()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["request_ids"]) == 50


def test_generation_loop_activates_backend_in_worker_thread():
    scheduler, _, _ = _make_real_scheduler("cpu")
    scheduler._backend = TorchNativeBackend()
    observed = []

    def observe_backend(*args, **kwargs):
        observed.append(type(get_backend()))
        scheduler._stop_event.set()

    scheduler._requests.wait_for_requests = observe_backend
    scheduler.start()
    scheduler._loop_thread.join(timeout=5)
    assert not scheduler._loop_thread.is_alive()
    assert observed == [TorchNativeBackend]
    scheduler.stop()


def test_scheduler_concurrent_add_remove_task(mock_model_and_tokenizer):
    """Test concurrent add and remove request operations."""
    scheduler = _make_mock_scheduler(mock_model_and_tokenizer)

    results = {"added": [], "removed": [], "errors": []}
    add_ready = threading.Event()

    def add_worker():
        try:
            for i in range(20):
                request_id = scheduler.add_request(f"prompt {i}")
                results["added"].append(request_id)
                if len(results["added"]) >= 10:
                    add_ready.set()
        except Exception as e:
            results["errors"].append(f"Add: {str(e)}")

    def remove_worker():
        try:
            add_ready.wait(timeout=5.0)
            for request_id in results["added"][:10]:
                scheduler.cancel_request(request_id)
                results["removed"].append(request_id)
        except Exception as e:
            results["errors"].append(f"Remove: {str(e)}")

    _run_threads(add_worker, remove_worker)
    scheduler.stop()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["added"]) == 20


def test_scheduler_concurrent_get_stats(mock_model_and_tokenizer):
    """Test concurrent get_stats operations."""
    scheduler = _make_mock_scheduler(mock_model_and_tokenizer)

    results = {"stats": [], "errors": []}
    started = threading.Event()
    stats_done = threading.Event()

    def add_requests():
        try:
            for i in range(20):
                scheduler.add_request(f"prompt {i}")
                started.set()
        except Exception as e:
            results["errors"].append(f"Add: {str(e)}")

    def get_stats():
        try:
            started.wait(timeout=5.0)
            for _ in range(50):
                stats = scheduler.get_stats()
                results["stats"].append(stats)
            stats_done.set()
        except Exception as e:
            results["errors"].append(f"Get stats: {str(e)}")

    _run_threads(add_requests, get_stats)
    scheduler.stop()
    stats_done.wait(timeout=5.0)

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["stats"]) == 50

    for stats in results["stats"]:
        assert "total_tasks" in stats
        assert stats["total_tasks"] >= 0


def test_cancel_waiting_task_storm_returns_to_baseline(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        request_ids = [
            scheduler.add_request(f"waiting-{index}", max_tokens=32)
            for index in range(32)
        ]

        assert all(scheduler.cancel_request(request_id) for request_id in request_ids)
        stats = scheduler.get_stats()
        assert stats["running"] == 0
        assert stats["waiting_tasks"] == 0
        assert stats["in_flight_tasks"] == 0
        assert stats["kv_cache_tasks"] == 0
        assert stats["cancelled_total"] == len(request_ids)
    finally:
        scheduler.stop()


def test_cancel_active_task_releases_metrics_and_kv(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        request_id = scheduler.add_request("active", max_tokens=32)
        request = scheduler._requests.pull_waiting(1)[0]
        assert scheduler._kv_manager.alloc_slots(request.request_id, request.prompt_ids)
        assert scheduler._requests.activate(request)

        before = scheduler.get_stats()
        assert before["running"] == 1
        assert before["in_flight_tasks"] == 1
        assert before["kv_cache_tasks"] == 1

        assert scheduler.cancel_request(request_id)
        scheduler.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            after = scheduler.get_stats()
            if (
                after["running"] == 0
                and after["in_flight_tasks"] == 0
                and after["kv_cache_tasks"] == 0
            ):
                break
            time.sleep(0.01)

        assert after["running"] == 0
        assert after["waiting_tasks"] == 0
        assert after["in_flight_tasks"] == 0
        assert after["kv_cache_tasks"] == 0
        assert after["cancelled_total"] == 1
    finally:
        scheduler.stop()


def test_cancel_during_kv_allocation_releases_metrics_and_kv(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    allocation_started = threading.Event()
    continue_allocation = threading.Event()
    original_alloc = scheduler._kv_manager.alloc_slots

    def blocking_alloc(*args, **kwargs):
        allocation_started.set()
        assert continue_allocation.wait(timeout=5)
        return original_alloc(*args, **kwargs)

    try:
        with patch.object(
            scheduler._kv_manager,
            "alloc_slots",
            side_effect=blocking_alloc,
        ):
            scheduler.start()
            request_id = scheduler.add_request("allocation-race", max_tokens=32)
            assert allocation_started.wait(timeout=5)
            assert scheduler.cancel_request(request_id)
            continue_allocation.set()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stats = scheduler.get_stats()
                if (
                    stats["running"] == 0
                    and stats["waiting_tasks"] == 0
                    and stats["in_flight_tasks"] == 0
                    and stats["kv_cache_tasks"] == 0
                ):
                    break
                time.sleep(0.01)

        assert stats["running"] == 0
        assert stats["waiting_tasks"] == 0
        assert stats["in_flight_tasks"] == 0
        assert stats["kv_cache_tasks"] == 0
        assert stats["cancelled_total"] == 1
    finally:
        continue_allocation.set()
        scheduler.stop()


def test_run_batch_returns_token_sequences(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30], [5, 6, 7, 8]]
        results = scheduler.run_batch(prompts, max_tokens=4, temperature=1.0)
        assert len(results) == 2
        for ids in results:
            assert isinstance(ids, list)
            assert len(ids) <= 4
            assert all(0 <= i < 200 for i in ids)
    finally:
        scheduler.stop()


def test_run_batch_tokens_match_full_sequence_forward(device):
    scheduler, _tok, model = _make_real_scheduler(device)
    prompt = [10, 20, 30, 40]
    try:
        expected = []
        sequence = list(prompt)
        for _ in range(2):
            input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
            position_ids = torch.arange(len(sequence), device=device).unsqueeze(0)
            input_mask = torch.ones(
                1, len(sequence), len(sequence), dtype=torch.bool, device=device
            ).tril()
            with torch.inference_mode():
                logits = model(
                    input_ids,
                    input_mask=input_mask,
                    position_ids=position_ids,
                )["logits"][:, -1, :]
            token = logits.argmax(dim=-1).item()
            expected.append(token)
            sequence.append(token)

        result = scheduler.run_batch(
            prompt_ids_list=[prompt], max_tokens=2, temperature=0
        )
        assert result == [expected]
    finally:
        scheduler.stop()


def test_run_batch_return_logprobs_aligned(device):
    """return_logprobs=True gives (token_ids, logprobs) tuples with equal len."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30, 40]]
        results = scheduler.run_batch(
            prompts, max_tokens=5, temperature=1.0, return_logprobs=True
        )
        assert len(results) == 1
        token_ids, logprobs = results[0]
        assert len(token_ids) == len(logprobs)
        assert all(lp <= 1e-5 for lp in logprobs)  # logprobs ≤ 0
    finally:
        scheduler.stop()


def test_ragged_prefill_matches_sequential_greedy_tokens_and_logprobs(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    prompts = [
        [10, 20, 30],
        [5, 6, 7, 8],
        [40, 41, 42, 43, 44],
    ]
    try:
        ragged = scheduler.run_batch(
            prompts, max_tokens=1, temperature=0, return_logprobs=True
        )
        sequential = [
            scheduler.run_batch(
                [prompt], max_tokens=1, temperature=0, return_logprobs=True
            )[0]
            for prompt in prompts
        ]

        assert [result[0] for result in ragged] == [result[0] for result in sequential]
        for ragged_result, sequential_result in zip(ragged, sequential):
            assert ragged_result[1] == pytest.approx(sequential_result[1], abs=1e-6)
    finally:
        scheduler.stop()


def test_run_batch_zero_max_tokens_returns_empty(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.run_batch([[10, 20, 30]], max_tokens=0) == [[]]
    finally:
        scheduler.stop()


def test_run_batch_stop_id_terminates(device):
    """A token matching stop_ids terminates generation for that prompt."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        # Make every token a stop id: generation must end after exactly
        # one token (the stop token itself) instead of running to max_tokens.
        scheduler._requests.tokenizer.stop_ids = list(range(200))
        prompts = [[10, 20, 30]]
        results = scheduler.run_batch(prompts, max_tokens=32, temperature=1.0)
        assert len(results[0]) == 1
    finally:
        scheduler.stop()


def test_run_batch_empty_prompts(device):
    """Empty prompt list yields empty result list."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.run_batch([], max_tokens=4) == []
    finally:
        scheduler.stop()


def test_run_batch_too_long_prompt_skipped(device):
    """A prompt longer than max_seq_len yields an empty result slot."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        long = list(range(100))  # > max_seq_len=64
        results = scheduler.run_batch([long, [10, 20]], max_tokens=2)
        assert results[0] == []
        assert len(results[1]) <= 2
    finally:
        scheduler.stop()


def test_run_batch_details_distinguish_rejection_from_success(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        long_prompt = list(range(100))
        results = scheduler.run_batch(
            [long_prompt, [10, 20]],
            max_tokens=2,
            temperature=0,
            return_logprobs=True,
            return_details=True,
        )

        assert results[0] == GenerationResult(
            token_ids=[],
            logprobs=[],
            finish_reason="rejected",
            error_reason="prompt_too_long",
        )
        assert results[1].finish_reason in ("stop", "length")
        assert results[1].error_reason is None
        assert len(results[1].token_ids) == len(results[1].logprobs)
    finally:
        scheduler.stop()


def test_run_batch_details_report_non_positive_max_tokens(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        result = scheduler.run_batch([[10, 20]], max_tokens=0, return_details=True)[0]
        assert result.finish_reason == "rejected"
        assert result.error_reason == "max_tokens_non_positive"
    finally:
        scheduler.stop()


def test_run_batch_details_report_allocation_failure(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        with patch.object(scheduler._kv_manager, "alloc_slots", return_value=False):
            result = scheduler.run_batch([[10, 20]], max_tokens=2, return_details=True)[
                0
            ]
        assert result.finish_reason == "rejected"
        assert result.error_reason == "kv_cache_allocation_failed"
    finally:
        scheduler.stop()


def test_run_batch_details_report_extension_failure_and_cleanup(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        with patch.object(
            scheduler._stepper,
            "step",
            side_effect=lambda requests, **_kwargs: ([], list(requests)),
        ):
            result = scheduler.run_batch([[10, 20]], max_tokens=2, return_details=True)[
                0
            ]

        assert result.finish_reason == "rejected"
        assert result.error_reason == "kv_cache_extension_failed"
        assert scheduler._kv_manager._states == {}
        assert scheduler._metrics._timings == {}
    finally:
        scheduler.stop()


def test_admission_rejects_requests_that_can_never_fit(device):
    """The livelock guard: a prompt larger than the whole paged pool is
    terminated (FINISH_REJECTED) instead of retrying alloc forever."""
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    scheduler = Scheduler(
        model=model,
        tokenizer=FakeTokenizer(),
        max_batch_size=2,
        max_seq_len=64,
        enable_cuda_graph=False,
        page_size=2,
        kv_tokens=16,  # 8 pages × 2 tokens = 16 token slots total
    )
    sink_events = []

    class _Capture(OutputEventSink):
        def __call__(self, events):
            sink_events.extend(events)

    scheduler.set_event_sink(_Capture())
    try:
        scheduler.start()
        # 60-token prompt can never fit a 16-slot pool.
        rid = scheduler.add_request(prompt="x" * 60, max_tokens=4)
        deadline = time.time() + 10
        while time.time() < deadline:
            if any(
                getattr(e, "request_id", None) == rid
                and getattr(e, "finish_reason", None) == "rejected"
                for e in sink_events
            ):
                break
            time.sleep(0.05)
        else:
            pytest.fail("oversized request was never rejected")
        # And the queue drained: no spinning leftovers.
        deadline = time.time() + 5
        while time.time() < deadline and scheduler._requests.waiting:
            time.sleep(0.05)
        assert not scheduler._requests.waiting
    finally:
        scheduler.stop()
