"""One synchronous resume contract and one checkpoint-safe update boundary."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from astrai.config import ComponentSpec
from astrai.dataset import GlobalBatchSampler, RDSampler
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks import TrainCallback
from astrai.trainer.context import TrainContext, TrainContextBuilder
from astrai.trainer.trainer import Trainer
from tests.integration.trainer.test_elastic_resume import _take_contract, _TokenDataset
from tests.support.rollout_cases import ConstantRewardModel
from tests.support.trainer import create_train_config, make_resume_metadata


def _config(path, **overrides):
    values = dict(
        device="cpu",
        dp_mode="none",
        batch_per_device=2,
        grad_accum_steps=2,
        ckpt_interval=1,
        random_seed=47,
    )
    values.update(overrides)
    scheduler = values.pop("scheduler", None)
    config = create_train_config(_TokenDataset(length=16), str(path), **values)
    return replace(config, scheduler=scheduler) if scheduler is not None else config


class _StopAfterOne(TrainCallback):
    def after_optimizer_step(self, context):
        if context.optimizer_step == 1:
            context.request_stop()


def _write_metadata(path, metadata):
    target = path / "meta.json"
    target.write_text(json.dumps(metadata, indent=2))
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["meta.json"]["size"] = target.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2))


def test_metadata_migration_preserves_resume_and_tensor_files(tmp_path):
    baseline_dir = tmp_path / "baseline"
    split_dir = tmp_path / "split"
    resumed_dir = tmp_path / "resumed"
    Trainer(_config(baseline_dir)).train()
    Trainer(_config(split_dir), callbacks=[_StopAfterOne()]).train()
    path = split_dir / "epoch_0_step_1"
    original = json.loads((path / "meta.json").read_text())
    persisted = {
        name: (path / name).read_bytes()
        for name in (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "metric.jsonl",
        )
    }
    old = {
        key: original[key]
        for key in (
            "epoch",
            "consumed_samples",
            "optimizer_step",
            "strategy",
            "dp_size",
            "cp_size",
            "batch_per_device",
            "grad_accum_steps",
            "optimizer",
            "scheduler",
        )
    }
    _write_metadata(path, old)
    config = _config(resumed_dir)
    with pytest.raises(ValueError, match="migrate checkpoint metadata"):
        TrainContextBuilder(config).with_param_path(str(path), resume=True).build()
    old.update(make_resume_metadata(config, 16, offset=4, step=1))
    _write_metadata(path, old)
    for name, contents in persisted.items():
        assert (path / name).read_bytes() == contents
    Trainer(config).train(param_path=str(path), resume=True)
    baseline = Checkpoint.load(str(baseline_dir / "epoch_0_step_4"))
    resumed = Checkpoint.load(str(resumed_dir / "epoch_0_step_4"))
    assert resumed.consumed_samples == baseline.consumed_samples == 16
    assert resumed.meta["epoch_sample_offset"] == baseline.meta["epoch_sample_offset"]
    assert resumed.extra["scheduler"] == baseline.extra["scheduler"]
    for name, tensor in baseline.state_dict.items():
        torch.testing.assert_close(tensor, resumed.state_dict[name], rtol=0, atol=0)
    for index, saved in baseline.extra["optimizer"]["state"].items():
        for key, tensor in saved.items():
            torch.testing.assert_close(
                tensor, resumed.extra["optimizer"]["state"][index][key], rtol=0, atol=0
            )


@pytest.mark.parametrize("failure", ["optimizer", "scheduler"])
def test_partial_update_cannot_publish_checkpoint(tmp_path, monkeypatch, failure):
    source_dir = tmp_path / "source"
    resumed_dir = tmp_path / "resumed"
    schedule = ComponentSpec(
        type="torch.optim.lr_scheduler:StepLR", params={"step_size": 1, "gamma": 0.5}
    )
    Trainer(
        _config(source_dir, scheduler=schedule), callbacks=[_StopAfterOne()]
    ).train()
    source = source_dir / "epoch_0_step_1"
    original_files = {path.name: path.read_bytes() for path in source.iterdir()}

    target = (
        torch.optim.AdamW if failure == "optimizer" else torch.optim.lr_scheduler.StepLR
    )
    real_step = target.step

    def fail_after_mutation(component, *args, **kwargs):
        result = real_step(component, *args, **kwargs)
        # StepLR also steps during construction, before optimizer state is restored.
        if failure == "optimizer" or component.optimizer.state:
            raise RuntimeError("injected failure after state mutation")
        return result

    monkeypatch.setattr(target, "step", fail_after_mutation)
    with pytest.raises(RuntimeError, match="injected failure"):
        Trainer(_config(resumed_dir, scheduler=schedule, ckpt_interval=1000)).train(
            param_path=str(source), resume=True
        )
    # Shutdown must not label mutated weights with the resumed step and cursor,
    # even when that step has not yet been published in the destination directory.
    assert not list(resumed_dir.glob("epoch_*_step_*"))
    assert original_files == {path.name: path.read_bytes() for path in source.iterdir()}
    saved = Checkpoint.load(str(source))
    assert saved.consumed_samples == saved.meta["epoch_sample_offset"] == 4
    assert saved.meta["optimizer_step"] == 1
    assert all(
        state["step"].item() == 1
        for state in saved.extra["optimizer"]["state"].values()
    )


def test_start_samples_uses_global_committed_boundary(tmp_path):
    context = TrainContextBuilder(_config(tmp_path, start_samples=4)).build()
    try:
        assert context.epoch_sample_offset == context.consumed_samples == 4
        assert isinstance(context.dataloader.batch_sampler, GlobalBatchSampler)
        assert context.dataloader.batch_sampler.start_offset == 4
    finally:
        context.run_shutdown()
    with pytest.raises(ValueError, match="committed window boundary"):
        TrainContextBuilder(_config(tmp_path, start_samples=3)).build()


def test_async_contract_records_padded_extent_and_locks_layout(tmp_path):
    config = create_train_config(
        _TokenDataset(length=5),
        str(tmp_path),
        device="cuda",
        strategy="online_grpo",
        dp_size=2,
        dp_mode="ddp",
        batch_per_device=2,
        grad_accum_steps=1,
        reward_model_fn=ConstantRewardModel,
        rollout_mode="async_round",
        rollout_interval=1,
        rollout_devices=["cuda:2", "cuda:3"],
    )
    metadata = make_resume_metadata(config, 5, offset=6, step=2)
    checkpoint = Checkpoint(epoch=0, consumed_samples=6, meta=metadata)
    assert _take_contract(config, checkpoint).epoch_sample_offset == 6
    changed = replace(config, dp_size=1, dp_mode="none", batch_per_device=4)
    with pytest.raises(ValueError, match="saved learner layout"):
        _take_contract(changed, checkpoint)


def test_async_epoch_cursor_ignores_prefetch_and_resets_at_epoch_end():
    dataset = range(5)
    seed = 47
    seen = []

    def prefetch_rounds(batches, stop_requested):
        batches = iter(batches)
        current = next(batches, None)
        while current is not None and not stop_requested():
            following = next(batches, None)
            yield current
            current = following

    context = TrainContext(
        dataset=dataset,
        dataloader=torch.utils.data.DataLoader(
            dataset, batch_size=2, sampler=RDSampler(dataset, seed=seed)
        ),
        async_rollout=SimpleNamespace(iter_rounds=prefetch_rounds),
    )
    trainer = Trainer.__new__(Trainer)
    trainer.callbacks = []

    def commit_round(context, batch):
        seen.extend(batch.tolist())
        context.consumed_samples += len(batch)
        context.epoch_sample_offset += len(batch)
        if context.consumed_samples == 2:
            context.request_stop()

    trainer._train_batch = commit_round
    trainer._train_epoch(context, 0)
    assert context.epoch == 0
    assert context.epoch_sample_offset == 2
    assert context.dataloader.sampler.iter > context.epoch_sample_offset
    context.run_shutdown()

    context._stop_event.clear()
    # Rebuilding the loader discards prefetched rounds. Only the committed
    # context cursor determines which samples the resumed epoch will visit.
    context.dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, sampler=RDSampler(dataset, seed=seed)
    )
    trainer._train_epoch(context, 0)
    assert context.consumed_samples == 5
    assert context.epoch == 1
    assert context.epoch_sample_offset == 0
    trainer._train_epoch(context, 1)
    assert context.consumed_samples == 10
    assert context.epoch == 2
    assert context.epoch_sample_offset == 0
    expected = [
        index
        for epoch in range(2)
        for index in torch.randperm(
            5, generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
    ]
    assert seen == expected
    context.run_shutdown()
