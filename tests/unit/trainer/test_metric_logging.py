import json
from types import SimpleNamespace

import pytest

from astrai.trainer.callbacks import MetricCallback


def _callback(val_step=0):
    return MetricCallback(metrics=["loss"], val_step=val_step)


def _context(step=0, samples=0, checkpoint=None, param_path=None, async_rollout=None):
    return SimpleNamespace(
        checkpoint=checkpoint,
        param_path=param_path,
        optimizer_step=step,
        consumed_samples=samples,
        epoch=0,
        metrics={},
        loss=1.0,
        val_dataloader=None,
        dp_size=1,
        async_rollout=async_rollout,
    )


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_resume_reads_only_selected_checkpoint_metric_history(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    records = [{"type": "step", "step": 4, "consumed_samples": 8}]
    (previous / "metric.jsonl").write_text(json.dumps(records[0]) + "\n")

    callback = _callback()
    callback.on_train_begin(
        _context(step=4, samples=8, checkpoint=object(), param_path=previous)
    )
    context = _context(step=5, samples=10)
    callback.before_optimizer_step(context)
    callback.after_optimizer_step(context)
    destination = tmp_path / "new-metric.jsonl"
    callback.write_snapshot(destination)

    assert _records(destination)[:1] == records
    assert [record["step"] for record in _records(destination)] == [4, 5]
    assert _records(previous / "metric.jsonl") == records


def test_failed_optimizer_step_does_not_enter_checkpoint_metrics(tmp_path):
    callback = _callback()
    context = _context(step=1, samples=2)
    callback.on_train_begin(_context())

    callback.before_optimizer_step(context)
    callback.on_shutdown(context)
    destination = tmp_path / "metric.jsonl"
    callback.write_snapshot(destination)

    assert destination.read_text() == ""


def test_each_checkpoint_contains_full_committed_history(tmp_path):
    callback = _callback()
    context = _context()
    callback.on_train_begin(context)

    for step in range(1, 4):
        context.optimizer_step = step
        context.consumed_samples = step * 2
        callback.before_optimizer_step(context)
        callback.after_optimizer_step(context)
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        log_path = checkpoint / "metric.jsonl"
        callback.write_snapshot(log_path)
        callback.checkpoint_saved(checkpoint, published=True)
        assert [record["step"] for record in _records(log_path)] == list(
            range(1, step + 1)
        )
        assert not callback.log_cache


def test_existing_checkpoint_cannot_discard_new_metric_records(tmp_path):
    callback = _callback()
    callback.on_train_begin(_context())
    context = _context(step=1, samples=2)
    callback.before_optimizer_step(context)
    callback.after_optimizer_step(context)

    with pytest.raises(RuntimeError, match="unsaved metrics"):
        callback.checkpoint_saved(tmp_path / "existing", published=False)


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
