"""Fixed-weight noise oracles and checkpoint/gradient invariants on CPU DDP."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from astrai.optimization.scheduler import SchedulerFactory
from astrai.parallel.setup import find_free_port, spawn_parallel_fn
from astrai.parallel.topology import ParallelTopology
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks import TrainCallback
from astrai.trainer.callbacks.checkpoint import CheckpointCallback
from astrai.trainer.callbacks.gradient_noise import (
    GradientNoiseProbe,
    GradientNoiseTracker,
)
from astrai.trainer.callbacks.metrics import MetricCallback
from astrai.trainer.context import TrainContext
from astrai.trainer.trainer import Trainer
from tests.integration.trainer.test_elastic_resume import _TokenDataset
from tests.support.trainer import create_train_config


def _noise_oracle(directory, bucket_view, dtype):
    rank = dist.get_rank()
    dtype = getattr(torch, dtype)
    finals = []
    for observed in (False, True):
        torch.manual_seed(51)
        model = DistributedDataParallel(
            torch.nn.Linear(2, 1, bias=False).to(dtype=dtype),
            gradient_as_bucket_view=bucket_view,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        tracker = GradientNoiseTracker(interval=1, beta=0.5)
        probe = GradientNoiseProbe()
        context = SimpleNamespace(
            model=model, topology=ParallelTopology(2), optimizer_step=0
        )
        for step in (1, 2, 3):
            optimizer.zero_grad(set_to_none=True)
            first = (
                torch.tensor([[1.0, 2.0]])
                if rank == 0
                else torch.tensor([[3.0, 4.0], [3.0, 4.0]])
            )
            last = (
                torch.tensor([[1.0, 2.0]]) if rank == 0 else torch.tensor([[3.0, 4.0]])
            )
            first = first.to(dtype=dtype)
            last = last.to(dtype=dtype)
            with model.no_sync():
                model(first).sum().backward()
            if observed:
                probe.start(model)
            (model(last).sum() * (0 if rank == 0 else 1)).backward()
            if observed:
                # These must be local accumulated numerators, before DDP averaging.
                expected = 5.0 if rank == 0 else 225.0
                assert sum(float(x) for x in probe.norms.values()) == expected
            for parameter in model.parameters():
                parameter.grad.mul_(2 / 4)
            if observed:
                probe.finish(context, 1 if rank == 0 else 3, 4, tracker)
                values = tracker.commit(step)
                assert tracker.signal == pytest.approx(17)
                assert tracker.covariance == pytest.approx(6)
                if step >= 2:
                    assert values["grad_noise_scale"] == pytest.approx(6 / 17)
                assert not probe.handles and not probe.norms
            optimizer.step()
            context.optimizer_step = step
        finals.append(model.module.weight.detach().clone())
    torch.testing.assert_close(finals[0], finals[1], rtol=0, atol=0)
    Path(directory, f"rank-{rank}.json").write_text(json.dumps({"unchanged": True}))


@pytest.mark.parametrize("bucket_view", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_passive_probe_preserves_native_ddp_updates(tmp_path, bucket_view, dtype):
    spawn_parallel_fn(
        _noise_oracle,
        directory=str(tmp_path),
        bucket_view=bucket_view,
        dtype=dtype,
        world_size=2,
        backend="gloo",
        master_port=find_free_port(),
    )
    assert len(list(tmp_path.glob("rank-*.json"))) == 2


class _StopAtTwo(TrainCallback):
    def __init__(self, step=2):
        self.step = step

    def after_optimizer_step(self, context):
        if context.optimizer_step == self.step:
            context.request_stop()


def _config(directory, length=16, **overrides):
    values = dict(
        device="cpu",
        dp_mode="ddp",
        dp_size=2,
        backend="gloo",
        master_port=find_free_port(),
        batch_per_device=1,
        grad_accum_steps=2,
        ckpt_interval=100,
        random_seed=47,
        metrics=["loss", "lr", "grad_norm", "grad_noise_scale"],
        grad_noise_interval=1,
        grad_noise_beta=0.5,
    )
    values.update(overrides)
    return create_train_config(_TokenDataset(length=length), str(directory), **values)


@pytest.mark.parametrize("interval,stop", [(1, 2), (2, 3)])
def test_noise_state_and_metric_history_resume_together(tmp_path, interval, stop):
    baseline_dir, split_dir, resumed_dir = [
        tmp_path / name for name in ("baseline", "split", "resumed")
    ]
    Trainer(_config(baseline_dir, grad_noise_interval=interval)).train()
    Trainer(
        _config(split_dir, grad_noise_interval=interval), callbacks=[_StopAtTwo(stop)]
    ).train()
    split = split_dir / f"epoch_0_step_{stop}"
    state = json.loads((split / "metric_state.json").read_text())
    assert state["optimizer_step"] == stop
    assert state["gradient_noise"]["samples"] == stop // interval
    Trainer(_config(resumed_dir, grad_noise_interval=interval)).train(
        param_path=str(split), resume=True
    )
    baseline = baseline_dir / "epoch_0_step_4"
    resumed = resumed_dir / "epoch_0_step_4"
    assert json.loads((baseline / "metric_state.json").read_text()) == json.loads(
        (resumed / "metric_state.json").read_text()
    )
    original = Checkpoint.load(str(baseline))
    loaded = Checkpoint.load(str(resumed))
    for key, value in original.state_dict.items():
        torch.testing.assert_close(value, loaded.state_dict[key], rtol=0, atol=0)

    def records(path):
        return [
            {k: v for k, v in json.loads(line).items() if k != "timestamp"}
            for line in path.read_text().splitlines()
        ]

    assert records(baseline / "metric.jsonl") == records(resumed / "metric.jsonl")
    rows = records(resumed / "metric.jsonl")
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    assert [row["step"] for row in rows if "grad_noise_sample_step" in row] == list(
        range(interval, 5, interval)
    )


class _NormObserver(TrainCallback):
    def __init__(self):
        self.expected = {}

    def after_backward(self, context, local_count, global_count):
        self.expected[context.optimizer_step + 1] = float(
            torch.nn.utils.get_total_norm(
                [p.grad for p in context.model.parameters() if p.grad is not None]
            )
        )

    def after_optimizer_step(self, context):
        assert context.grad_norm == pytest.approx(self.expected[context.optimizer_step])
        if context.optimizer_step == 2:
            context.request_stop()


def test_gradient_norm_is_recorded_for_current_update(tmp_path):
    observer = _NormObserver()
    config = create_train_config(
        _TokenDataset(length=8),
        str(tmp_path),
        device="cpu",
        dp_mode="none",
        ckpt_interval=100,
    )
    Trainer(config, callbacks=[observer]).train()
    rows = [
        json.loads(line)
        for line in (tmp_path / "epoch_0_step_2/metric.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 2]
    for row in rows:
        assert row["grad_norm"] == pytest.approx(observer.expected[row["step"]])


@pytest.mark.parametrize(
    "overrides",
    [
        {"dp_size": 1, "dp_mode": "none"},
        {"dp_mode": "fsdp"},
        {"cp_size": 2},
        {"tp_size": 2},
        {"compile_mode": "default"},
    ],
)
def test_unsupported_noise_layout_is_rejected(tmp_path, overrides):
    with pytest.raises(ValueError, match="grad_noise_scale requires"):
        _config(tmp_path, **overrides)


def test_single_active_tail_rank_does_not_publish_fake_noise(tmp_path):
    Trainer(_config(tmp_path, length=9, epoch_tail_policy="flush")).train()
    path = tmp_path / "epoch_0_step_3"
    state = json.loads((path / "metric_state.json").read_text())
    assert state["optimizer_step"] == 3
    assert (
        state["gradient_noise"]["samples"]
        == state["gradient_noise"]["last_probe_step"]
        == 2
    )
    rows = [
        json.loads(line) for line in (path / "metric.jsonl").read_text().splitlines()
    ]
    assert "grad_noise_sample_step" not in rows[-1]
    assert "grad_noise_scale" not in rows[-1]


def test_async_publication_error_commits_metrics_after_policy_advance(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = SchedulerFactory.create(
        "cosine", optimizer, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
    )
    coordinator = SimpleNamespace(policy_version=0)

    class Strategy:
        @property
        def policy_version(self):
            return coordinator.policy_version

        def training_updates(self, batch, _microbatch_prompts):
            yield [{"loss": model(batch.prompts).square().mean(), "metrics": {}}]

        def optimizer_step(self, _optimizer):
            _optimizer.step()
            coordinator.policy_version += 1
            raise RuntimeError("publication failed after version advance")

    class Executor:
        grad_accum_steps = 1
        sync_gradients = True

        def accumulate(self, _model):
            return nullcontext()

        def backward(self, loss):
            loss.backward()

        def checkpoint_context(self, _model):
            return nullcontext(_model.state_dict())

    config = create_train_config(
        _TokenDataset(length=4), str(tmp_path), device="cpu", dp_mode="none"
    )
    context = TrainContext(
        model=model,
        strategy=Strategy(),
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=range(4),
        config=config,
        executor=Executor(),
        async_rollout=coordinator,
    )
    metric = MetricCallback(metrics=["loss"])
    checkpoint = CheckpointCallback(str(tmp_path), 100, metric_callback=metric)
    metric.on_train_begin(context)
    checkpoint.on_train_begin(context)
    trainer = Trainer.__new__(Trainer)
    trainer.callbacks = [metric, checkpoint]
    with pytest.raises(RuntimeError, match="publication failed"):
        trainer._train_batch(context, SimpleNamespace(prompts=torch.ones(1, 1)))
    metric.on_shutdown(context)
    checkpoint.on_shutdown(context)
    saved = tmp_path / "epoch_0_step_1"
    assert (
        context.optimizer_step
        == context.epoch_sample_offset
        == context.consumed_samples
        == 1
    )
    assert json.loads((saved / "metric_state.json").read_text())["optimizer_step"] == 1
    records = [
        json.loads(line) for line in (saved / "metric.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in records] == [1]
