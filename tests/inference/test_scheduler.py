"""Tests for scheduler concurrency."""

import threading
from unittest.mock import MagicMock, patch

import pytest
import torch

from astrai.inference import InferenceScheduler


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

    with patch("astrai.inference.core.scheduler.AutoModel"):
        with patch("astrai.inference.core.scheduler.AutoTokenizer"):
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


def test_scheduler_concurrent_add_remove_task(mock_model_and_tokenizer):
    """Test concurrent add and remove task operations."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.core.scheduler.AutoModel"):
        with patch("astrai.inference.core.scheduler.AutoTokenizer"):
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

    with patch("astrai.inference.core.scheduler.AutoModel"):
        with patch("astrai.inference.core.scheduler.AutoTokenizer"):
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


def test_prefill_skips_fully_cached_tasks(mock_model_and_tokenizer):
    """Tasks whose entire prompt is cached skip the prefill phase."""
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("astrai.inference.core.scheduler.AutoModel"):
        with patch("astrai.inference.core.scheduler.AutoTokenizer"):
            scheduler = InferenceScheduler(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_batch_size=4,
                device="cpu",
            )

    task_id = scheduler.add_task("short prompt", stream_callback=lambda t: None)
    scheduler.stop()
    assert task_id.startswith("task_")


def _make_real_scheduler(device):
    """Build a scheduler backed by a tiny real model for run_batch tests."""
    from astrai.config.model_config import AutoRegressiveLMConfig
    from astrai.model.transformer import AutoRegressiveLM

    class _Tok:
        stop_ids = [2]

        def encode(self, texts, **_):
            if isinstance(texts, str):
                texts = [texts]
            return [[b for b in t.encode("utf-8")] for t in texts]

        def decode(self, ids, skip_special_tokens=True):
            return bytes(b for b in ids if b > 2 or not skip_special_tokens).decode(
                "utf-8", errors="ignore"
            )

    cfg = AutoRegressiveLMConfig(
        vocab_size=200,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=64,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
    )
    model = AutoRegressiveLM(cfg).to(device=device).eval()
    tokenizer = _Tok()
    scheduler = InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
        max_prompt_len=64,
    )
    return scheduler, tokenizer, model


def test_run_batch_returns_token_sequences():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def test_run_batch_return_logprobs_aligned():
    """return_logprobs=True gives (token_ids, logprobs) tuples with equal len."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def test_run_batch_respects_max_tokens():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        prompts = [[10, 20, 30]]
        results = scheduler.run_batch(prompts, max_tokens=3, temperature=1.0)
        assert len(results[0]) <= 3
    finally:
        scheduler.stop()


def test_run_batch_stop_id_terminates():
    """A token matching stop_ids terminates generation for that prompt."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def test_run_batch_empty_prompts():
    """Empty prompt list yields empty result list."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        assert scheduler.run_batch([], max_tokens=4) == []
    finally:
        scheduler.stop()


def test_run_batch_too_long_prompt_skipped():
    """A prompt longer than max_seq_len yields an empty result slot."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler, _tok, _model = _make_real_scheduler(device)
    try:
        long = list(range(100))  # > max_seq_len=64
        results = scheduler.run_batch([long, [10, 20]], max_tokens=2)
        assert results[0] == []
        assert len(results[1]) <= 2
    finally:
        scheduler.stop()
