import json
from types import SimpleNamespace

import pytest
import torch

from astrai.optimization.optimizer.precision import PrecisionOptimizer
from astrai.parallel.executor import NoneExecutor
from astrai.parallel.topology import ParallelTopology
from astrai.trainer.callbacks import GradientClippingCallback, MetricCallback
from astrai.trainer.callbacks.checkpoint import _write_metric_snapshot
from astrai.trainer.callbacks.metric_util import GradSNRTracker
from astrai.trainer.context import PreloadedState, TrainContext, TrainContextBuilder
from astrai.trainer.context.metric import MetricHistory
from astrai.trainer.trainer import Trainer, _invoke_callback
from tests.support.rollout import RandomTokenDataset
from tests.support.trainer import create_train_config


def _callback(val_step=0):
    return MetricCallback(metrics=["loss"], val_step=val_step)


def _context(step=0, samples=0, checkpoint=None, param_path=None, rollout=None):
    return TrainContext(
        checkpoint=checkpoint,
        param_path=param_path,
        optimizer_steps_completed=step,
        consumed_samples=samples,
        epoch=0,
        metrics={},
        loss=1.0,
        val_dataloader=None,
        grad_snr_tracker=None,
        rollout=rollout,
        config=SimpleNamespace(rollout_mode="async_round" if rollout else "sync"),
    )


@pytest.mark.parametrize("prior", [{}, {"batch": None}], ids=["absent", "existing"])
def test_callback_context_restores_payload_on_error(prior):
    context = TrainContext(kwargs=dict(prior, experiment="math"))

    def fail(ctx):
        assert ctx.kwargs["batch"] == {"rewards": [0, 1]}
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        _invoke_callback(fail, context, "batch", {"rewards": [0, 1]})
    assert context.kwargs == dict(prior, experiment="math")


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("resume", [True, False])
def test_context_restores_only_selected_resume_history(tmp_path, resume):
    previous = tmp_path / "previous"
    previous.mkdir()
    old_record = {"type": "step", "step": 4, "consumed_samples": 8}
    (previous / "metric.jsonl").write_text(json.dumps(old_record) + "\n")
    config = create_train_config(RandomTokenDataset(), str(tmp_path), "cpu")
    builder = TrainContextBuilder(config).with_param_path(str(previous), resume=resume)
    builder._topology = ParallelTopology(1)
    state = PreloadedState(checkpoint=object() if resume else None)
    context = builder._create_context(state, executor=None)
    context.metric_history.stage({"type": "step"})
    context.metric_history.commit(5, 10)
    destination = tmp_path / "metric.jsonl"
    _write_metric_snapshot(context.metric_history.snapshot(), destination)

    assert [record["step"] for record in _records(destination)] == (
        [4, 5] if resume else [5]
    )
    assert _records(previous / "metric.jsonl") == [old_record]


def test_each_checkpoint_contains_full_committed_history(tmp_path):
    callback = _callback()
    context = _context()
    history = context.metric_history
    for step in range(1, 4):
        callback.before_optimizer_step(context)
        history.commit(step, step * 2)
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        snapshot = history.snapshot()
        _write_metric_snapshot(snapshot, checkpoint / "metric.jsonl")
        history.mark_saved(checkpoint)
        records = _records(checkpoint / "metric.jsonl")
        assert [(r["step"], r["consumed_samples"]) for r in records] == [
            (i, i * 2) for i in range(1, step + 1)
        ]
        assert not history.snapshot().records


def test_snapshot_excludes_pending_records(tmp_path):
    history = MetricHistory()
    history.stage({"type": "step"})
    history.commit(1, 2)
    history.stage({"type": "step"})
    snapshot = history.snapshot()
    _write_metric_snapshot(snapshot, tmp_path / "metric.jsonl")
    history.mark_saved(tmp_path)
    assert not history.snapshot().records

    history.discard_pending()
    history.commit(2, 4)
    _write_metric_snapshot(history.snapshot(), tmp_path / "next.jsonl")
    assert [r["step"] for r in _records(tmp_path / "metric.jsonl")] == [1]
    assert [r["step"] for r in _records(tmp_path / "next.jsonl")] == [1]


@pytest.mark.parametrize("mode", ["offline", "online", "online_committed"])
def test_metric_commit_follows_optimizer_commit_on_failure(tmp_path, mode):
    config = create_train_config(RandomTokenDataset(), str(tmp_path), "cpu")
    trainer = Trainer(config)
    trainer.callbacks = [_callback()]
    context = _context()
    context.checkpoint_safe = False
    context.pending_samples = 2
    context.optimizer = SimpleNamespace(zero_grad=lambda: None)
    context.rollout = SimpleNamespace(policy_version=0)

    def fail_update(_context):
        if mode == "online_committed":
            context.rollout.policy_version += 1
        raise RuntimeError("update failed")

    context.strategy = SimpleNamespace(after_update=lambda: None)
    trainer._step_policy = fail_update
    with pytest.raises(RuntimeError, match="update failed"):
        if mode == "offline":
            trainer._commit_global_update(context, new_data=True)
        else:
            batch = SimpleNamespace(prompts=SimpleNamespace(shape=(2,)))
            trainer._commit_online_update(context, batch, new_data=True)
    committed = mode == "online_committed"
    assert context.optimizer_step == int(committed)
    assert context.consumed_samples == 2 * int(committed)
    assert context.checkpoint_safe == committed
    _write_metric_snapshot(context.metric_history.snapshot(), tmp_path / "metric.jsonl")
    records = _records(tmp_path / "metric.jsonl")
    assert [(r["step"], r["consumed_samples"]) for r in records] == (
        [(1, 2)] if committed else []
    )
    # No failed update may be committed by a later successful boundary.
    context.metric_history.commit(2, 4)
    assert len(context.metric_history.snapshot().records) == int(committed)


def test_validation_cadence_uses_global_step_after_resume():
    callback = _callback(val_step=10)
    callback.on_train_begin(_context(step=11, samples=22, checkpoint=object()))

    assert callback._next_val_step == 21


@pytest.mark.parametrize(("step", "expected"), [(0, 0), (1, 10), (11, 20)])
def test_async_validation_cadence_uses_pre_update_step(step, expected):
    callback = _callback(val_step=10)
    callback.on_train_begin(_context(step=step, checkpoint=object(), rollout=object()))

    assert callback._next_val_step == expected


@pytest.mark.parametrize("main_grad", [False, True], ids=["native", "fp32-main"])
def test_grad_norm_logs_current_pre_clipping_gradient(tmp_path, monkeypatch, main_grad):
    config = create_train_config(
        RandomTokenDataset(), str(tmp_path), "cpu", metrics=["grad_norm", "grad_snr"]
    )
    trainer = Trainer(config)
    trainer.callbacks = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, (MetricCallback, GradientClippingCallback))
    ]
    context = _context()
    context.model = torch.nn.Linear(
        2, 1, bias=False, dtype=torch.bfloat16 if main_grad else torch.float32
    )
    context.model.weight.data.zero_()
    context.executor = NoneExecutor()
    context.optimizer = PrecisionOptimizer(
        torch.optim.SGD(context.model.parameters(), lr=0.1)
    )
    context.strategy = SimpleNamespace(after_update=lambda: None)
    context.grad_snr_tracker = GradSNRTracker(beta=0.5)
    expected = torch.zeros(1, 2)

    for index, gradient in enumerate(((3.0, 4.0), (5.0, 12.0), (7.0, 24.0)), 1):
        raw = torch.tensor([gradient])
        parameter = context.model.weight
        if main_grad:
            parameter.main_grad = raw.clone()
            parameter.grad = torch.full_like(parameter, 100)
        else:
            parameter.grad = raw.clone()
        context.pending_samples = 2
        if index == 2:

            def fail_update():
                raise RuntimeError("optimizer failed")

            with monkeypatch.context() as patch:
                patch.setattr(context.optimizer, "step", fail_update)
                with pytest.raises(RuntimeError, match="optimizer failed"):
                    trainer._commit_global_update(context, new_data=True)
            assert context.grad_snr_tracker.observations == 1
            assert context.grad_snr_tracker.last_step == 1
            assert context.grad_snr_tracker._pending is None
            assert context.optimizer_step == 1
            assert context.consumed_samples == 2
            assert len(context.metric_history.snapshot().records) == 1
            # Clipping changed the failed attempt's buffer; retry the raw gradient.
            if main_grad:
                parameter.main_grad.copy_(raw)
            else:
                parameter.grad.copy_(raw)
        trainer._commit_global_update(context, new_data=True)
        expected.add_(raw * (1 / (raw.norm() + 1e-6)).clamp(max=1), alpha=-0.1)
        torch.testing.assert_close(
            context.optimizer.master_state_dict(context.model.named_parameters())[
                "weight"
            ],
            expected,
        )

    records = [
        json.loads(record) for record in context.metric_history.snapshot().records
    ]
    assert [(record["step"], record["consumed_samples"]) for record in records] == [
        (1, 2),
        (2, 4),
        (3, 6),
    ]
    assert [record["grad_norm"] for record in records] == pytest.approx([5, 13, 25])
    assert context.grad_snr_tracker.observations == 3
    assert context.grad_snr_tracker.last_step == 3
    assert "grad_snr" not in records[0]
    assert records[-1]["grad_snr"] == context.grad_snr_tracker.snr


@pytest.mark.parametrize("mismatch", ["missing", "step", "interval"])
def test_grad_snr_resume_rejects_inconsistent_estimator(mismatch):
    context = _context(step=4, checkpoint=SimpleNamespace(extra={"grad_snr": {}}))
    context.grad_snr_tracker = GradSNRTracker()
    context.grad_snr_tracker.last_step = 4
    context.grad_snr_tracker.interval = 2
    if mismatch == "missing":
        context.checkpoint.extra.clear()
    elif mismatch == "step":
        context.grad_snr_tracker.last_step = 3
    else:
        context.grad_snr_tracker.interval = 1
    callback = MetricCallback(metrics=["grad_snr"], grad_snr_interval=2)
    with pytest.raises(ValueError, match="grad_snr requires|SNR state|SNR sampling"):
        callback.on_train_begin(context)
    assert context.checkpoint_safe is False
