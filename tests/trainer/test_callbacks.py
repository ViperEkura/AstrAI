from pathlib import Path

import torch

from astrai.model.components.decoder_block import DecoderBlock
from astrai.serialization import Checkpoint
from astrai.trainer.train_callback import (
    GradientCheckpointingCallback,
    TrainCallback,
    _copy_tokenizer_files,
)
from astrai.trainer.trainer import Trainer
from tests.helpers import RandomTokenDataset, load_checkpoint_meta


def test_gradient_checkpointing_enable_disable(test_model):
    """Enable wraps forward, _disable restores it."""
    model = test_model["model"]
    callback = GradientCheckpointingCallback(modules=[DecoderBlock])

    originals = [layer.forward for layer in model.layers]

    for layer in model.layers:
        callback._enable(layer)

    for i, layer in enumerate(model.layers):
        assert hasattr(layer, "_original_forward")
        assert layer.forward is not originals[i]

    for layer in model.layers:
        callback._disable(layer)

    for layer in model.layers:
        assert not hasattr(layer, "_original_forward")


def test_gradient_checkpointing_empty_modules_noop(test_model):
    """modules=None should leave forwards untouched."""
    model = test_model["model"]
    callback = GradientCheckpointingCallback()

    for layer in model.layers:
        callback._enable(layer)

    for layer in model.layers:
        assert not hasattr(layer, "_original_forward")


def test_gradient_checkpointing_forward_unchanged(test_model):
    """Forward output unchanged after patching (no_grad)."""
    model = test_model["model"]
    device = test_model["device"]
    callback = GradientCheckpointingCallback(modules=[DecoderBlock])

    input_ids = torch.randint(0, 1000, (2, 32)).to(device)

    with torch.no_grad():
        ref = model(input_ids)["logits"].clone()

    for layer in model.layers:
        callback._enable(layer)

    with torch.no_grad():
        out = model(input_ids)["logits"]

    assert torch.equal(ref, out)


def test_gradient_checkpointing_backward(test_model):
    """backward passes gradients through checkpointed layers."""
    model = test_model["model"]
    device = test_model["device"]
    callback = GradientCheckpointingCallback(modules=[DecoderBlock])

    for layer in model.layers:
        callback._enable(layer)

    input_ids = torch.randint(0, 1000, (2, 32)).to(device)
    target_ids = torch.randint(0, 1000, (2, 32)).to(device)

    logits = model(input_ids)["logits"]
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(), target_ids.flatten()
    )
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} gradient is None"

    for layer in model.layers:
        callback._disable(layer)

    model.zero_grad()
    for name, p in model.named_parameters():
        assert p.grad is None or p.grad.sum().item() == 0, f"{name} grad not zeroed"


def test_gradient_checkpointing_trainer_integration(
    base_test_env, random_dataset, train_config_factory, device
):
    """Gradient checkpointing runs end-to-end via Trainer."""
    train_config = train_config_factory(
        model_fn=lambda: base_test_env["model"],
        dataset=random_dataset,
        test_dir=base_test_env["test_dir"],
        device=device,
        ckpt_interval=3,
        gradient_checkpointing_modules=[DecoderBlock],
    )

    trainer = Trainer(train_config)
    gc_callbacks = [
        c for c in trainer.callbacks if isinstance(c, GradientCheckpointingCallback)
    ]
    assert gc_callbacks and gc_callbacks[0].modules == (DecoderBlock,)
    trainer.train()


def test_callback_integration(
    base_test_env, random_dataset, train_config_factory, device
):
    """Test that all callbacks are properly integrated"""
    train_config = train_config_factory(
        model_fn=lambda: base_test_env["model"],
        dataset=random_dataset,
        test_dir=base_test_env["test_dir"],
        device=device,
        ckpt_interval=3,
    )

    callback_calls = []

    class TrackingCallback(TrainCallback):
        def on_train_begin(self, context):
            callback_calls.append("on_train_begin")

        def on_batch_end(self, context):
            callback_calls.append("on_batch_end")

        def on_epoch_end(self, context):
            callback_calls.append("on_epoch_end")

    trainer = Trainer(train_config, callbacks=[TrackingCallback()])
    trainer.train()

    assert "on_train_begin" in callback_calls
    assert "on_batch_end" in callback_calls
    assert "on_epoch_end" in callback_calls


def test_checkpoint_captures_completed_optimizer_step(
    base_test_env, train_config_factory, device
):
    """Checkpoint state must include the update represented by its step number."""
    model = base_test_env["model"]
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    train_config = train_config_factory(
        model_fn=lambda: model,
        dataset=RandomTokenDataset(length=2),
        test_dir=base_test_env["test_dir"],
        device=device,
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
    assert checkpoint.extra["optimizer"]["state"]
    assert checkpoint.extra["scheduler"]["last_epoch"] == 1
    assert checkpoint.meta["optimizer_step"] == 1
    assert (
        Path(base_test_env["test_dir"]) / "epoch_0_step_1" / "metric.jsonl"
    ).is_file()


def test_checkpoint_snapshots_tokenizer_files(
    base_test_env, train_config_factory, device, tmp_path
):
    """Checkpoints copy tokenizer files from param_path so resume works."""
    param_dir = tmp_path / "model"
    param_dir.mkdir()
    (param_dir / "tokenizer.json").write_text("{}")
    (param_dir / "tokenizer_config.json").write_text("{}")

    train_config = train_config_factory(
        model_fn=lambda: base_test_env["model"],
        dataset=RandomTokenDataset(length=2),
        test_dir=base_test_env["test_dir"],
        device=device,
        batch_per_device=2,
        ckpt_interval=1,
    )

    Trainer(train_config).train(param_path=str(param_dir))

    ckpt_dir = Path(base_test_env["test_dir"]) / "epoch_0_step_1"
    assert (ckpt_dir / "tokenizer.json").is_file()
    assert (ckpt_dir / "tokenizer_config.json").is_file()

    # Resuming with param_path == checkpoint dir must not raise
    # (samefile guard).
    _copy_tokenizer_files(str(ckpt_dir), str(ckpt_dir))


def test_copy_tokenizer_files_skips_missing_and_none(tmp_path):
    _copy_tokenizer_files(None, str(tmp_path))
    _copy_tokenizer_files(str(tmp_path), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_interrupt_before_first_step_still_checkpoints(
    base_test_env, train_config_factory, device
):
    """A graceful interrupt handled before the first optimizer step must
    still leave a checkpoint (regression: optimizer_step == last_ckpt_step
    skipped the emergency save entirely)."""

    class _ImmediateStop(TrainCallback):
        def on_epoch_begin(self, context):
            context.request_stop()

    train_config = train_config_factory(
        model_fn=lambda: base_test_env["model"],
        dataset=RandomTokenDataset(length=2),
        test_dir=base_test_env["test_dir"],
        device=device,
        batch_per_device=2,
        ckpt_interval=1000,
    )

    trainer = Trainer(train_config)
    trainer.callbacks.insert(0, _ImmediateStop())
    trainer.train()

    meta = load_checkpoint_meta(base_test_env["test_dir"])
    assert meta["optimizer_step"] == 0
