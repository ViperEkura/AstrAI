"""Tests for scheduler concurrency."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from astrai.extension import CudaBackend, TorchNativeBackend, get_backend
from astrai.inference import GenerationResult, InferenceScheduler
from astrai.inference.metrics import MetricsCollector
from astrai.inference.runtime.executor import DecodeSteadyState, Executor
from astrai.inference.task import Task
from astrai.model.transformer import AutoRegressiveLM
from tests.helpers import FakeTokenizer, make_rollout_config


@pytest.fixture
def mock_model_and_tokenizer():
    """Create mock model and tokenizer."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_model.config.num_key_value_heads = 8
    mock_model.config.num_attention_heads = 8
    mock_model.config.hidden_size = 128
    mock_model.config.num_hidden_layers = 2
    mock_model.config.max_position_embeddings = 100
    mock_model.parameters.return_value = iter(
        [MagicMock(dtype=torch.float32, device=torch.device("cpu"))]
    )

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "token"
    mock_tokenizer.stop_ids = [0]
    mock_tokenizer.pad_id = None

    return mock_model, mock_tokenizer


def test_scheduler_concurrent_add_task(mock_model_and_tokenizer):
    """Test concurrent add_task operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"task_ids": [], "errors": []}
    lock = threading.Lock()

    def add_task_worker(worker_id):
        try:
            for i in range(10):
                task_id = scheduler.add_task(f"prompt from worker {worker_id}-{i}")
                with lock:
                    results["task_ids"].append(task_id)
        except Exception as e:
            results["errors"].append(str(e))

    threads = [threading.Thread(target=add_task_worker, args=(i,)) for i in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    scheduler.stop()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["task_ids"]) == 50


def test_generation_loop_activates_backend_in_worker_thread():
    scheduler = object.__new__(InferenceScheduler)
    scheduler._backend = TorchNativeBackend()
    scheduler._stop_event = threading.Event()
    scheduler._task_cache = MagicMock()

    observed = []
    task_mgr = MagicMock()
    task_mgr.tokenizer.stop_ids = [0]
    task_mgr.remove_finished_tasks.return_value = []
    task_mgr.get_active_tasks.return_value = []
    task_mgr.max_batch_size = 1
    task_mgr.pull_candidates.return_value = []
    task_mgr.has_work.return_value = False

    def observe_backend(*args, **kwargs):
        observed.append(type(get_backend()))
        scheduler._stop_event.set()

    task_mgr.wait_for_tasks.side_effect = observe_backend
    scheduler._task_mgr = task_mgr

    thread = threading.Thread(target=scheduler._run_generation_loop)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert observed == [TorchNativeBackend]


def test_step_splits_decode_batch_by_request_backend():
    scheduler = object.__new__(InferenceScheduler)
    scheduler._task_cache = MagicMock()
    scheduler._task_cache.task_extend.return_value = True
    scheduler._metrics = MetricsCollector()
    scheduler._executor = MagicMock()

    observed = []

    def execute(tasks, **kwargs):
        observed.append((type(get_backend()), [task.task_id for task in tasks]))
        return [1] * len(tasks)

    scheduler._executor.execute_decode.side_effect = execute

    torch_task = Task("torch", [1], backend=TorchNativeBackend())
    cuda_task = Task("cuda", [1], backend=CudaBackend())
    for task in (torch_task, cuda_task):
        task.input_tokens = 1
        task.output_ids = [1]
        task.mark_prefill_done()
        scheduler._metrics.register(task.task_id)

    produced, aborted = scheduler._step([torch_task, cuda_task])

    assert aborted == []
    assert produced == [torch_task, cuda_task]
    assert observed == [
        (TorchNativeBackend, ["torch"]),
        (CudaBackend, ["cuda"]),
    ]


def test_step_batches_ragged_prefill_with_shared_cache_start():
    scheduler = object.__new__(InferenceScheduler)
    scheduler._cache = SimpleNamespace(page_size=64)
    scheduler._task_cache = MagicMock()
    scheduler._task_cache.task_cached.return_value = 0
    scheduler._metrics = MetricsCollector()
    scheduler._executor = MagicMock()

    short = Task("short", [1, 2, 3])
    long = Task("long", [4, 5, 6, 7, 8])
    for task in (short, long):
        scheduler._metrics.register(task.task_id)

    scheduler._executor.execute_prefill.return_value = (
        [long, short],
        [11, 12],
    )

    produced, aborted = scheduler._step([short, long])

    assert aborted == []
    assert produced == [long, short]
    scheduler._executor.execute_prefill.assert_called_once_with(
        [short, long], start_pos=0, return_logprobs=False
    )
    assert long.output_ids == [11]
    assert short.output_ids == [12]


def test_execute_prefill_packs_ragged_prompts_and_selects_last_logits():
    executor = object.__new__(Executor)
    executor.device = torch.device("cpu")
    executor.task_cache = MagicMock()
    executor.task_cache.bind.return_value = MagicMock()
    executor._workspace = MagicMock()
    all_logits = torch.arange(42, dtype=torch.float32).reshape(6, 7)
    executor.model = MagicMock(return_value={"logits": all_logits})
    executor._sample_logits = MagicMock(
        return_value=([101, 102], torch.tensor([101, 102]))
    )

    task_b = Task("b", [20, 21, 22, 23, 24])
    task_a = Task("a", [10, 11, 12])

    tasks, output = executor.execute_prefill([task_b, task_a], start_pos=1)

    assert tasks == [task_a, task_b]
    assert output == [101, 102]
    model_args, model_kwargs = executor.model.call_args
    assert model_args[0].tolist() == [11, 12, 21, 22, 23, 24]
    assert model_kwargs["position_ids"].tolist() == [1, 2, 1, 2, 3, 4]
    executor.task_cache.bind.assert_called_once_with(
        ["a", "b"], executor._workspace, start_pos=1
    )
    sample_args, sample_kwargs = executor._sample_logits.call_args
    torch.testing.assert_close(sample_args[0], all_logits[[1, 5]])
    assert sample_args[1] == [task_a, task_b]
    assert sample_args[2] is False
    assert sample_kwargs == {}


def test_scheduler_concurrent_add_remove_task(mock_model_and_tokenizer):
    """Test concurrent add and remove task operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"added": [], "removed": [], "errors": []}
    add_ready = threading.Event()

    def add_worker():
        try:
            for i in range(20):
                task_id = scheduler.add_task(f"prompt {i}")
                results["added"].append(task_id)
                if len(results["added"]) >= 10:
                    add_ready.set()
        except Exception as e:
            results["errors"].append(f"Add: {str(e)}")

    def remove_worker():
        try:
            add_ready.wait(timeout=5.0)
            for task_id in results["added"][:10]:
                scheduler.remove_task(task_id)
                results["removed"].append(task_id)
        except Exception as e:
            results["errors"].append(f"Remove: {str(e)}")

    add_thread = threading.Thread(target=add_worker)
    remove_thread = threading.Thread(target=remove_worker)

    add_thread.start()
    remove_thread.start()

    add_thread.join()
    remove_thread.join()
    scheduler.stop()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["added"]) == 20


def test_scheduler_concurrent_get_stats(mock_model_and_tokenizer):
    """Test concurrent get_stats operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.scheduler.AutoModel"):
        with patch("astrai.inference.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    results = {"stats": [], "errors": []}
    started = threading.Event()
    stats_done = threading.Event()

    def add_tasks():
        try:
            for i in range(20):
                scheduler.add_task(f"prompt {i}")
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

    add_thread = threading.Thread(target=add_tasks)
    stats_thread = threading.Thread(target=get_stats)

    add_thread.start()
    stats_thread.start()

    add_thread.join()
    stats_done.wait(timeout=5.0)
    scheduler.stop()

    stats_thread.join()

    assert len(results["errors"]) == 0, f"Errors: {results['errors']}"
    assert len(results["stats"]) == 50

    for stats in results["stats"]:
        assert "total_tasks" in stats
        assert stats["total_tasks"] >= 0


def _make_real_scheduler(device):
    """Build a scheduler backed by a tiny real model for run_batch tests."""
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = FakeTokenizer()
    scheduler = InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
    )
    return scheduler, tokenizer, model


def test_cancel_waiting_task_storm_returns_to_baseline(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        task_ids = [
            scheduler.add_task(f"waiting-{index}", max_tokens=32) for index in range(32)
        ]

        assert all(scheduler.cancel_task(task_id) for task_id in task_ids)
        stats = scheduler.get_stats()
        assert stats["active_tasks"] == 0
        assert stats["waiting_tasks"] == 0
        assert stats["in_flight_tasks"] == 0
        assert stats["kv_cache_tasks"] == 0
        assert stats["cancelled_total"] == len(task_ids)
    finally:
        scheduler.stop()


def test_cancel_active_task_releases_metrics_and_kv(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        task_id = scheduler.add_task("active", max_tokens=32)
        task = scheduler._task_mgr.pull_candidates(1)[0]
        assert scheduler._task_cache.task_alloc(task.task_id, task.prompt_ids)
        assert scheduler._task_mgr.activate(task)

        before = scheduler.get_stats()
        assert before["active_tasks"] == 1
        assert before["in_flight_tasks"] == 1
        assert before["kv_cache_tasks"] == 1

        assert scheduler.cancel_task(task_id)
        scheduler.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            after = scheduler.get_stats()
            if (
                after["active_tasks"] == 0
                and after["in_flight_tasks"] == 0
                and after["kv_cache_tasks"] == 0
            ):
                break
            time.sleep(0.01)

        assert after["active_tasks"] == 0
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
    original_alloc = scheduler._task_cache.task_alloc

    def blocking_alloc(*args, **kwargs):
        allocation_started.set()
        assert continue_allocation.wait(timeout=5)
        return original_alloc(*args, **kwargs)

    try:
        with patch.object(
            scheduler._task_cache,
            "task_alloc",
            side_effect=blocking_alloc,
        ):
            scheduler.start()
            task_id = scheduler.add_task("allocation-race", max_tokens=32)
            assert allocation_started.wait(timeout=5)
            assert scheduler.cancel_task(task_id)
            continue_allocation.set()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                stats = scheduler.get_stats()
                if (
                    stats["active_tasks"] == 0
                    and stats["waiting_tasks"] == 0
                    and stats["in_flight_tasks"] == 0
                    and stats["kv_cache_tasks"] == 0
                ):
                    break
                time.sleep(0.01)

        assert stats["active_tasks"] == 0
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


def test_run_batch_respects_max_tokens(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30]]
        results = scheduler.run_batch(prompts, max_tokens=3, temperature=1.0)
        assert len(results[0]) <= 3
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
        prompts = [[10, 20, 30]]
        results = scheduler.run_batch(prompts, max_tokens=32, temperature=1.0)
        # If stop token 2 was produced, it is the last token
        if results[0] and results[0][-1] == 2:
            # No tokens after stop should exist (since we terminate)
            assert 2 not in results[0][:-1]
    finally:
        scheduler.stop()


def test_run_batch_empty_prompts(device):
    """Empty prompt list yields empty result list."""
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.run_batch([], max_tokens=4) == []
    finally:
        scheduler.stop()


def test_scheduler_weight_versions_are_monotonic_and_acknowledged(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.policy_version == 0
        assert scheduler.update_weights(1) == 1
        assert scheduler.policy_version == 1
        assert scheduler.get_stats()["policy_version"] == 1
        assert scheduler.update_weights(1) == 1
        with pytest.raises(ValueError, match="cannot move backwards"):
            scheduler.update_weights(0)
        with pytest.raises(ValueError, match="non-negative integer"):
            scheduler.update_weights(True)
    finally:
        scheduler.stop()


def test_scheduler_applies_weight_mutation_and_version_atomically(device):
    scheduler, _tok, model = _make_real_scheduler(device)
    before = next(model.parameters()).detach().clone()

    def mutate():
        with torch.no_grad():
            next(model.parameters()).add_(1)
        return "updated"

    try:
        assert scheduler.apply_weight_update(1, mutate) == "updated"
        assert scheduler.policy_version == 1
        assert not torch.equal(next(model.parameters()), before)
        with pytest.raises(ValueError, match="must advance"):
            scheduler.apply_weight_update(1, mutate)

        def failed_mutation():
            raise RuntimeError("optimizer failed")

        with pytest.raises(RuntimeError, match="optimizer failed"):
            scheduler.apply_weight_update(2, failed_mutation)
        assert scheduler.policy_version == 1
    finally:
        scheduler.stop()


def test_scheduler_serializes_policy_snapshot_and_direct_update(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    update_finished = threading.Event()
    errors = []

    def inspect(version):
        assert version == 0
        snapshot_started.set()
        assert release_snapshot.wait(timeout=5)

    def take_snapshot():
        try:
            scheduler.with_policy_snapshot(inspect)
        except BaseException as exc:
            errors.append(exc)

    def update():
        try:
            scheduler.update_weights(1)
            update_finished.set()
        except BaseException as exc:
            errors.append(exc)

    snapshot_thread = threading.Thread(target=take_snapshot)
    update_thread = threading.Thread(target=update)
    try:
        snapshot_thread.start()
        assert snapshot_started.wait(timeout=5)
        update_thread.start()
        assert not update_finished.wait(timeout=0.1)
        release_snapshot.set()
        snapshot_thread.join(timeout=5)
        update_thread.join(timeout=5)
        assert not snapshot_thread.is_alive()
        assert not update_thread.is_alive()
        assert errors == []
        assert scheduler.policy_version == 1
    finally:
        release_snapshot.set()
        snapshot_thread.join(timeout=5)
        update_thread.join(timeout=5)
        scheduler.stop()


def test_scheduler_rejects_weight_update_with_queued_tasks(device):
    scheduler, _tok, _model = _make_real_scheduler(device)
    task_id = scheduler.add_task("queued")
    try:
        with pytest.raises(RuntimeError, match="while tasks are queued"):
            scheduler.update_weights(1)
        scheduler.remove_task(task_id)
        assert scheduler.update_weights(1) == 1
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
        with patch.object(scheduler._task_cache, "task_alloc", return_value=False):
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
            scheduler,
            "_step",
            side_effect=lambda tasks, **_kwargs: ([], list(tasks)),
        ):
            result = scheduler.run_batch([[10, 20]], max_tokens=2, return_details=True)[
                0
            ]

        assert result.finish_reason == "rejected"
        assert result.error_reason == "kv_cache_extension_failed"
        assert scheduler._task_cache._states == {}
        assert scheduler._metrics._timings == {}
    finally:
        scheduler.stop()


def test_decode_does_not_reuse_previous_batch_state():
    executor = object.__new__(Executor)
    executor.device = torch.device("cpu")
    executor.task_cache = MagicMock()
    executor.task_cache.bind_was_steady = True
    executor.task_cache.bind.return_value = MagicMock()
    executor._graph_supported = False
    executor._graph_ctx = SimpleNamespace(enabled=False)

    workspace = MagicMock()
    workspace.position_ids = torch.tensor([2], dtype=torch.long)
    workspace.fill_input_ids.return_value = torch.tensor([7], dtype=torch.long)
    workspace.decode_mask.return_value = torch.ones(1, 1, 9, dtype=torch.bool)
    executor._workspace = workspace
    executor.model = MagicMock(
        return_value={"logits": torch.zeros(1, 1, 10, dtype=torch.float32)}
    )

    old_info = object()
    new_info = object()
    executor._decode_cache = DecodeSteadyState(("old",), [2], old_info)
    executor._sample_logits = MagicMock(
        return_value=([3], torch.tensor([3], dtype=torch.long))
    )

    task = Task("new", list(range(8)), temperature=0)
    task.input_tokens = 8
    task.output_ids = [7]
    task.mark_prefill_done()

    with patch(
        "astrai.inference.runtime.executor._build_sampling_batch_info",
        return_value=new_info,
    ):
        assert executor.execute_decode([task]) == [3]

    assert workspace.position_ids.tolist() == [8]
    assert executor._decode_cache.task_sig == ("new",)
    executor._sample_logits.assert_called_once()
    args, kwargs = executor._sample_logits.call_args
    assert args[1:] == ([task], False)
    assert kwargs["info"] is new_info


def test_decode_fills_input_ids_from_device_on_matching_signature():
    """Steady-state decode copies cached device tokens, skipping the host."""
    executor = object.__new__(Executor)
    executor.device = torch.device("cpu")
    executor.task_cache = MagicMock()
    executor.task_cache.bind_was_steady = True
    executor.task_cache.bind.return_value = MagicMock()
    executor._graph_supported = False
    executor._graph_ctx = SimpleNamespace(enabled=False)

    workspace = MagicMock()
    workspace.position_ids = torch.tensor([2], dtype=torch.long)
    workspace.fill_input_ids_from_device.return_value = torch.tensor(
        [9], dtype=torch.long
    )
    executor._workspace = workspace
    executor.model = MagicMock(
        return_value={"logits": torch.zeros(1, 1, 10, dtype=torch.float32)}
    )

    info = object()
    tokens = torch.tensor([3], dtype=torch.long)
    executor._decode_cache = DecodeSteadyState(("t1",), [2], info, last_tokens=tokens)
    executor._sample_logits = MagicMock(return_value=([3], tokens))

    task = Task("t1", list(range(8)), temperature=0)
    task.input_tokens = 8
    task.output_ids = [7]
    task.mark_prefill_done()

    with patch(
        "astrai.inference.runtime.executor._build_sampling_batch_info",
        return_value=info,
    ):
        assert executor.execute_decode([task]) == [3]

    workspace.fill_input_ids.assert_not_called()
    workspace.fill_input_ids_from_device.assert_called_once_with(tokens)
    assert workspace.position_ids.tolist() == [3]
    assert executor._decode_cache.task_sig == ("t1",)
    assert executor._decode_cache.last_tokens is tokens
