"""Optimizer commits, periodic checkpoints and interrupted accumulation."""

from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from astrai.parallel.executor import NoneExecutor
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks import TrainCallback
from astrai.trainer.callbacks.checkpoint import CheckpointCallback
from astrai.trainer.callbacks.progress import ProgressBarCallback
from astrai.trainer.components import build_scheduler_component
from astrai.trainer.strategy.base import BaseStrategy
from astrai.trainer.train_context import TrainContext, restore_training_progress
from astrai.trainer.trainer import Trainer
from tests.support.rollout import RandomTokenDataset
from tests.support.trainer import create_train_config


class FixedTokenDataset(RandomTokenDataset):
    def __getitem__(self, index):
        tokens = (torch.arange(self.max_length + 1) + index) % self.vocab_size
        return {"input_ids": tokens[:-1], "target_ids": tokens[1:]}


def _config(path, length=16, **kwargs):
    return create_train_config(
        FixedTokenDataset(length=length, max_length=8),
        str(path),
        "cpu",
        batch_per_device=2,
        grad_accum_steps=2,
        ckpt_interval=2,
        **kwargs,
    )


class StopAfterMicrobatch(TrainCallback):
    def __init__(self, count):
        self.count = count
        self.seen = 0
        self.state = None

    def on_batch_end(self, context):
        self.seen += 1
        if self.seen == self.count:
            self.state = (
                context.optimizer_step,
                context.consumed_samples,
                context.pending_samples,
            )
            context.request_stop()


@pytest.mark.parametrize(
    "length,epochs,stop_after",
    [(16, 1, 1), (16, 1, 5), (6, 2, 3), (6, 2, 4)],
)
def test_interrupted_accumulation_resumes_like_uninterrupted(
    tmp_path, monkeypatch, length, epochs, stop_after
):
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    baseline = tmp_path / "baseline"
    torch.manual_seed(123)
    Trainer(_config(baseline, length=length, n_epoch=epochs)).train()
    final_step = length * epochs // 4
    final_name = f"epoch_{epochs - 1}_step_{final_step}"
    expected = Checkpoint.load(baseline / final_name)

    interrupted = tmp_path / "interrupted"
    stop = StopAfterMicrobatch(stop_after)
    torch.manual_seed(123)
    Trainer(
        _config(interrupted, length=length, n_epoch=epochs), callbacks=[stop]
    ).train()
    step = stop_after // 2
    assert stop.state == (step, step * 4, (stop_after % 2) * 2)
    stopped_epoch = (stop_after - 1) // (length // 2)
    checkpoint = interrupted / f"epoch_{stopped_epoch}_step_{step}"
    saved = Checkpoint.load(checkpoint)
    assert saved.meta["optimizer_step"] == step
    assert saved.consumed_samples == step * 4
    assert saved.extra["scheduler"]["last_epoch"] == step
    assert "optimizer_steps_completed" not in saved.meta

    Trainer(_config(interrupted, length=length, n_epoch=epochs)).train(
        param_path=str(checkpoint), resume=True
    )
    actual = Checkpoint.load(interrupted / final_name)
    assert actual.consumed_samples == expected.consumed_samples == length * epochs
    assert actual.extra["scheduler"]["last_epoch"] == final_step
    for name, value in actual.state_dict.items():
        torch.testing.assert_close(value, expected.state_dict[name], rtol=0, atol=0)


def test_periodic_checkpoint_matches_update_and_cursor(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    Trainer(_config(tmp_path)).train()
    for step in (2, 4):
        saved = Checkpoint.load(tmp_path / f"epoch_0_step_{step}")
        assert saved.meta["optimizer_step"] == step
        assert saved.consumed_samples == step * 4
        assert saved.extra["scheduler"]["last_epoch"] == step
        for state in saved.extra["optimizer"]["state"].values():
            assert state["step"].item() == step
        metrics = (tmp_path / f"epoch_0_step_{step}" / "metric.jsonl").read_text()
        assert f'"step": {step},' in metrics


def _toy_context(accum=1):
    executor = NoneExecutor(grad_accum_steps=accum)
    model, optimizer, scheduler = executor.prepare(
        lambda: torch.nn.Linear(1, 1),
        lambda model: torch.optim.SGD(model.parameters(), lr=0.1),
        lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, 1),
    )

    class Strategy(BaseStrategy):
        def training_updates(self, batch, _microbatch):
            yield [{"loss": model(batch["input_ids"]).square().mean(), "metrics": {}}]

        def optimizer_step(self, optimizer):
            optimizer.step()

    return TrainContext(
        config=SimpleNamespace(
            batch_per_device=2, grad_accum_steps=accum, rollout_mode="sync"
        ),
        executor=executor,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        strategy=Strategy(model, "cpu"),
    )


def _trainer():
    trainer = Trainer.__new__(Trainer)
    trainer.callbacks = []
    return trainer


def test_short_batch_and_multiple_updates_count_independently():
    context = _toy_context()
    original = context.strategy.training_updates

    def two_updates(batch, microbatch):
        yield from original(batch, microbatch)
        yield from original(batch, microbatch)

    context.strategy.training_updates = two_updates
    _trainer()._train_batch(context, {"input_ids": torch.ones(1, 1)})
    assert context.optimizer_step == 2
    assert context.consumed_samples == 1
    assert context.scheduler.state_dict()["last_epoch"] == 2


def test_failed_scheduler_prevents_inconsistent_checkpoint():
    context = _toy_context()

    def fail():
        raise RuntimeError("scheduler failed")

    context.scheduler.step = fail
    with pytest.raises(RuntimeError, match="scheduler failed"):
        _trainer()._train_batch(context, {"input_ids": torch.ones(2, 1)})
    assert not context.checkpoint_safe
    assert context.optimizer_step == 0
    assert context.consumed_samples == 0


def test_progress_resumes_with_cumulative_step():
    context = _toy_context(accum=2)
    context.optimizer_step = 3
    context.dataloader = range(8)
    callback = ProgressBarCallback(1, file=StringIO())
    callback.on_epoch_begin(context)
    assert callback.progress_bar.n == 3
    assert callback.progress_bar.total == 7
    callback.on_epoch_end(context)


def test_accumulation_across_epochs_matches_scheduler_budget():
    context = _toy_context(accum=2)
    config = _config(Path("unused"), n_epoch=2)
    config = replace(
        config, scheduler=replace(config.scheduler, params={"warmup_ratio": 0.0})
    )
    scheduler = build_scheduler_component(
        config.scheduler, context.optimizer.optimizer, 6, config
    )
    trainer = _trainer()
    for _ in range(2):
        for _ in range(3):
            trainer._train_batch(context, {"input_ids": torch.ones(2, 1)})
    assert context.optimizer_step == 3
    assert scheduler.lr_decay_steps == 3


@pytest.mark.parametrize("published", [False, True])
def test_publication_failure_keeps_checkpoint_unsafe(published):
    context = _toy_context()
    runner = SimpleNamespace(policy_version=0)
    context.strategy._rollout_runner = runner

    def fail(optimizer):
        if published:
            optimizer.step()
            runner.policy_version += 1
        raise RuntimeError("publication failed")

    context.strategy.optimizer_step = fail
    with pytest.raises(RuntimeError, match="publication failed"):
        _trainer()._train_batch(context, {"input_ids": torch.ones(2, 1)})
    assert context.optimizer_step == 0
    assert context.consumed_samples == 0
    assert not context.checkpoint_safe
    assert context.scheduler.state_dict()["last_epoch"] == 0


def test_multi_update_batch_checkpoints_only_after_all_updates(tmp_path):
    context = _toy_context()
    original = context.strategy.training_updates
    observed = []

    def updates(batch, microbatch):
        for _ in range(2):
            yield from original(batch, microbatch)
            observed.append(
                (
                    context.optimizer_step,
                    context.consumed_samples,
                    context.checkpoint_safe,
                )
            )

    context.strategy.training_updates = updates
    checkpoint = CheckpointCallback(str(tmp_path), interval=1)
    checkpoint.on_train_begin(context)
    saved = []
    checkpoint._save_checkpoint = lambda current: saved.append(
        (current.optimizer_step, current.consumed_samples)
    )
    trainer = _trainer()
    trainer.callbacks = [checkpoint]
    trainer._train_batch(context, {"input_ids": torch.ones(2, 1)})
    assert observed == [(1, 0, False), (2, 0, False)]
    assert saved == [(2, 2)]
    assert context.checkpoint_safe


@pytest.mark.parametrize("failure", ["callback", "iterator"])
def test_interrupted_multi_update_batch_keeps_last_complete_checkpoint(
    tmp_path, failure
):
    context = _toy_context()
    original = context.strategy.training_updates

    def updates(batch, microbatch):
        yield from original(batch, microbatch)
        if failure == "iterator":
            raise RuntimeError("interrupted learner updates")
        yield from original(batch, microbatch)

    class Interrupt(TrainCallback):
        def after_optimizer_step(self, current):
            if failure == "callback":
                raise RuntimeError("interrupted learner updates")

    context.strategy.training_updates = updates
    checkpoint = CheckpointCallback(str(tmp_path), interval=1)
    checkpoint.on_train_begin(context)
    saved = []
    checkpoint._save_checkpoint = lambda current: saved.append(current.optimizer_step)
    trainer = _trainer()
    trainer.callbacks = [checkpoint, Interrupt()]
    with pytest.raises(RuntimeError, match="interrupted learner updates"):
        trainer._train_batch(context, {"input_ids": torch.ones(2, 1)})
    checkpoint.on_error(context)
    checkpoint.on_train_end(context)
    assert saved == []
    assert context.optimizer_step == 1
    assert context.consumed_samples == 0
    assert context.pending_samples == 2
    assert not context.checkpoint_safe


@pytest.mark.parametrize(
    "batch",
    [
        {"metadata": "abcd", "input_ids": torch.ones(2, 1)},
        {"input_ids": torch.ones(2, 1), "metadata": "abcd"},
        {"instruction": ["a", "b"], "metadata": "ignored"},
    ],
)
def test_batch_size_ignores_metadata_order(batch):
    assert BaseStrategy.batch_size(batch) == 2


@pytest.mark.parametrize(
    "batch",
    [
        {"metadata": ["a", "b"]},
        {"input_ids": torch.ones(2, 1), "target_ids": torch.ones(3, 1)},
        {"input_ids": torch.tensor(1)},
        {"input_ids": []},
    ],
)
def test_batch_size_rejects_missing_or_inconsistent_sample_dimensions(batch):
    with pytest.raises(ValueError):
        BaseStrategy.batch_size(batch)


@pytest.mark.parametrize("failure", ["callback", "iterator"])
def test_multi_update_checkpoint_resume_matches_complete_run(tmp_path, failure):
    def setup(path, interrupt=False):
        context = _toy_context()
        context.config.to_dict = lambda: {
            "batch_per_device": 2,
            "dp_size": 1,
            "grad_accum_steps": 1,
        }
        original = context.strategy.training_updates

        def updates(batch, microbatch):
            yield from original(batch, microbatch)
            if interrupt and failure == "iterator" and context.optimizer_step == 3:
                raise RuntimeError("interrupted learner updates")
            yield from original(batch, microbatch)

        class Interrupt(TrainCallback):
            def after_optimizer_step(self, current):
                if interrupt and failure == "callback" and current.optimizer_step == 3:
                    raise RuntimeError("interrupted learner updates")

        context.strategy.training_updates = updates
        checkpoint = CheckpointCallback(str(path), interval=1)
        trainer = _trainer()
        trainer.callbacks = [checkpoint, Interrupt()]
        return context, trainer, checkpoint

    batches = [
        {"input_ids": torch.tensor([[1.0], [2.0]])},
        {"input_ids": torch.tensor([[3.0], [4.0]])},
    ]
    torch.manual_seed(17)
    expected, baseline, callback = setup(tmp_path / "baseline")
    callback.on_train_begin(expected)
    for batch in batches:
        baseline._train_batch(expected, batch)

    torch.manual_seed(17)
    stopped, trainer, callback = setup(tmp_path / "interrupted", interrupt=True)
    callback.on_train_begin(stopped)
    trainer._train_batch(stopped, batches[0])
    with pytest.raises(RuntimeError, match="interrupted learner updates"):
        trainer._train_batch(stopped, batches[1])
    callback.on_error(stopped)
    callback.on_train_end(stopped)
    path = tmp_path / "interrupted"
    assert not (path / "epoch_0_step_1" / "manifest.json").exists()
    assert not (path / "epoch_0_step_3" / "manifest.json").exists()
    saved = Checkpoint.load(path / "epoch_0_step_2")

    resumed, trainer, callback = setup(tmp_path / "resumed")
    resumed.model.load_state_dict(saved.state_dict)
    resumed.optimizer.load_state_dict(saved.extra["optimizer"])
    resumed.scheduler.load_state_dict(saved.extra["scheduler"])
    resumed.optimizer_step, resumed.consumed_samples = restore_training_progress(
        saved, resumed.config, 1
    )
    callback.on_train_begin(resumed)
    for batch in batches[resumed.consumed_samples // 2 :]:
        trainer._train_batch(resumed, batch)
    assert resumed.optimizer_step == expected.optimizer_step == 4
    assert resumed.consumed_samples == expected.consumed_samples == 4
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(
            value, expected.model.state_dict()[name], rtol=0, atol=0
        )
