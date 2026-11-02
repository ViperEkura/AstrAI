import copy
import json
from pathlib import Path

import pytest
import torch

import astrai.trainer.callbacks.checkpoint as checkpoint_module
from astrai.model.components.decoder_block import DecoderBlock
from astrai.model.components.mlp import MLP
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks import (
    GradientCheckpointingCallback,
    TrainCallback,
)
from astrai.trainer.callbacks.checkpoint import (
    _copy_tokenizer_files,
    _write_metric_snapshot,
)
from astrai.trainer.trainer import Trainer
from tests.support.checkpoint import load_checkpoint_meta
from tests.support.rollout import RandomTokenDataset
from tests.support.tokenizers import build_test_tokenizer


@pytest.mark.parametrize(
    "modules", [[], [DecoderBlock], [MLP]], ids=["disabled", "decoder", "mlp"]
)
def test_gradient_checkpointing_preserves_forward_gradients_and_restores(
    test_model, modules
):
    model = test_model["model"]
    reference = copy.deepcopy(model)
    inputs = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    callback = GradientCheckpointingCallback(modules=modules)
    originals = {name: module.forward for name, module in model.named_modules()}
    model.apply(callback._enable)
    assert {
        name
        for name, module in model.named_modules()
        if hasattr(module, "_original_forward")
    } == {
        name
        for name, module in model.named_modules()
        if isinstance(module, tuple(modules))
    }

    with torch.no_grad():
        torch.testing.assert_close(
            model(inputs)["logits"], reference(inputs)["logits"], rtol=0, atol=0
        )
    expected = reference(inputs)["logits"]
    actual = model(inputs)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for logits in (expected, actual):
        torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        ).backward()
    for (name, parameter), (ref_name, ref_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert name == ref_name
        if parameter.requires_grad:
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, ref_parameter.grad, rtol=0, atol=0
            )

    model.apply(callback._disable)
    assert all(
        not hasattr(module, "_original_forward") and module.forward == originals[name]
        for name, module in model.named_modules()
    )


def test_checkpoint_captures_completed_optimizer_step(
    base_test_env, train_config_factory, cpu_device
):
    """Checkpoint state must include the update represented by its step number."""
    model = base_test_env["model"]
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    train_config = train_config_factory(
        dataset=RandomTokenDataset(length=2),
        test_dir=base_test_env["test_dir"],
        device=cpu_device,
        batch_per_device=2,
        ckpt_interval=1,
    )

    Trainer(train_config).train()

    checkpoint = Checkpoint.load(
        str(Path(base_test_env["test_dir"]) / "epoch_0_step_1")
    )
    assert any(
        not torch.equal(checkpoint.state_dict[name].cpu(), initial_tensor)
        for name, initial_tensor in initial_state.items()
    )
    assert checkpoint.extra["optimizer"]["optimizer"]["state"]
    assert checkpoint.extra["scheduler"]["last_epoch"] == 1
    assert checkpoint.meta["optimizer_step"] == 1
    log_path = Path(base_test_env["test_dir"]) / "epoch_0_step_1" / "metric.jsonl"
    assert [
        record["step"] for record in map(json.loads, log_path.read_text().splitlines())
    ] == [1]
    assert (
        "metric.jsonl"
        in json.loads((log_path.parent / "manifest.json").read_text())["files"]
    )
    assert not (Path(base_test_env["test_dir"]) / "metric.jsonl").exists()


@pytest.mark.parametrize("reverse_callbacks", [False, True])
def test_metrics_are_published_with_each_checkpoint(
    base_test_env, train_config_factory, cpu_device, reverse_callbacks
):
    config = train_config_factory(
        dataset=RandomTokenDataset(length=4),
        test_dir=base_test_env["test_dir"],
        device=cpu_device,
        batch_per_device=2,
        ckpt_interval=1,
    )

    trainer = Trainer(config)
    if reverse_callbacks:
        trainer.callbacks.reverse()
    trainer.train()

    directory = Path(base_test_env["test_dir"])
    for step in (1, 2):
        checkpoint_dir = directory / f"epoch_0_step_{step}"
        records = [
            json.loads(line)
            for line in (checkpoint_dir / "metric.jsonl").read_text().splitlines()
        ]
        assert [
            record["step"] for record in records if record["type"] == "step"
        ] == list(range(1, step + 1))
        assert (
            "metric.jsonl"
            in json.loads((checkpoint_dir / "manifest.json").read_text())["files"]
        )


def test_checkpoint_snapshots_tokenizer_files(
    base_test_env, train_config_factory, cpu_device, tmp_path
):
    """Checkpoints copy tokenizer files from param_path so resume works."""
    param_dir = tmp_path / "model"
    param_dir.mkdir()
    build_test_tokenizer().save_pretrained(param_dir)
    (param_dir / "generation_config.json").write_text('{"eos_token_id": [0, 1]}')

    train_config = train_config_factory(
        dataset=RandomTokenDataset(length=2),
        test_dir=base_test_env["test_dir"],
        device=cpu_device,
        batch_per_device=2,
        ckpt_interval=1,
    )

    Trainer(train_config).train(param_path=str(param_dir))

    ckpt_dir = Path(base_test_env["test_dir"]) / "epoch_0_step_1"
    assert (ckpt_dir / "tokenizer.json").is_file()
    assert (ckpt_dir / "tokenizer_config.json").is_file()
    files = json.loads((ckpt_dir / "manifest.json").read_text())["files"]
    assert {"tokenizer.json", "tokenizer_config.json", "generation_config.json"} <= set(
        files
    )
    assert json.loads((ckpt_dir / "generation_config.json").read_text())[
        "eos_token_id"
    ] == [0, 1]

    # Resuming with param_path == checkpoint dir must not raise
    # (samefile guard).
    _copy_tokenizer_files(str(ckpt_dir), str(ckpt_dir))


def test_interrupt_before_first_step_still_checkpoints(
    base_test_env, train_config_factory, cpu_device
):
    """A graceful interrupt handled before the first optimizer step must
    still leave a checkpoint (regression: optimizer_step == last_ckpt_step
    skipped the emergency save entirely)."""

    class _ImmediateStop(TrainCallback):
        def on_epoch_begin(self, context):
            context.request_stop()

    train_config = train_config_factory(
        dataset=RandomTokenDataset(length=2),
        test_dir=base_test_env["test_dir"],
        device=cpu_device,
        batch_per_device=2,
        ckpt_interval=1000,
    )

    trainer = Trainer(train_config)
    trainer.callbacks.insert(0, _ImmediateStop())
    trainer.train()

    meta = load_checkpoint_meta(base_test_env["test_dir"])
    assert meta["optimizer_step"] == 0


def test_failed_metric_write_retries_without_losing_committed_history(
    train_config_factory, tmp_path, monkeypatch
):
    config = train_config_factory(
        RandomTokenDataset(length=4), str(tmp_path), device="cpu", ckpt_interval=1
    )
    attempts = []

    def fail_first_write(snapshot, destination):
        attempts.append(destination)
        _write_metric_snapshot(snapshot, destination)
        if len(attempts) == 1:
            raise OSError("metric write interrupted")

    monkeypatch.setattr(checkpoint_module, "_write_metric_snapshot", fail_first_write)
    with pytest.raises(OSError, match="metric write interrupted"):
        Trainer(config).train()

    # Shutdown retries the committed step after the failed staging directory
    # was removed. The failed write must neither acknowledge nor duplicate it.
    checkpoint = tmp_path / "epoch_0_step_1"
    records = list(
        map(json.loads, (checkpoint / "metric.jsonl").read_text().splitlines())
    )
    assert [(r["step"], r["consumed_samples"]) for r in records] == [(1, 2)]
    assert len(attempts) == 2
    assert not list(tmp_path.glob(".epoch_*.tmp-*"))
    Checkpoint.verify(str(checkpoint))


def test_existing_checkpoint_cannot_discard_new_metric_records(
    train_config_factory, tmp_path
):
    config = train_config_factory(
        RandomTokenDataset(length=4), str(tmp_path), device="cpu", ckpt_interval=1
    )
    Trainer(config).train()
    checkpoint = tmp_path / "epoch_0_step_1" / "metric.jsonl"
    original = checkpoint.read_bytes()
    with pytest.raises(RuntimeError, match="unsaved metrics"):
        Trainer(config).train()
    assert checkpoint.read_bytes() == original
