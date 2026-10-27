import json
from types import SimpleNamespace

import pytest

from astrai.parallel.topology import ParallelTopology
from astrai.trainer.callbacks import MetricCallback
from astrai.trainer.callbacks.checkpoint import _write_metric_snapshot
from astrai.trainer.context import PreloadedState, TrainContext, TrainContextBuilder
from astrai.trainer.context.metric import MetricHistory
from astrai.trainer.trainer import Trainer
from tests.support.rollout import RandomTokenDataset
from tests.support.trainer import create_train_config


def _callback(val_step=0):
    return MetricCallback(metrics=["loss"], val_step=val_step)


def _context(step=0, samples=0, checkpoint=None, param_path=None, async_rollout=None):
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
        async_rollout=async_rollout,
    )


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
    context.async_rollout = SimpleNamespace(policy_version=0)

    def fail_update(_optimizer):
        if mode == "online_committed":
            context.async_rollout.policy_version += 1
        raise RuntimeError("update failed")

    context.strategy = SimpleNamespace(optimizer_step=fail_update)
    with pytest.raises(RuntimeError, match="update failed"):
        if mode == "offline":
            trainer._commit_offline_update(context, new_data=True)
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
    callback.on_train_begin(
        _context(step=step, checkpoint=object(), async_rollout=object())
    )

    assert callback._next_val_step == expected
