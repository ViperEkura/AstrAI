"""Checkpoint optimizer loading for new stages and complete resumes."""

from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from astrai.config import ComponentSpec, TrainConfig
from astrai.optimization.optimizer import OptimizerFactory
from astrai.serialization import Checkpoint
from astrai.serialization import checkpoint as checkpoint_module
from astrai.trainer.components import load_optimizer_state_dict
from astrai.trainer.train_context import (
    TrainContextBuilder,
    repair_legacy_training_progress,
)
from scripts.tools.checkpoint import repair_progress_command
from tests.support.models import make_model


def _config(**overrides):
    values = dict(
        strategy="seq",
        model=ComponentSpec(type="autoregressive_lm"),
        dataset=ComponentSpec(
            type="tests.support.rollout:RandomTokenDataset",
            params={"length": 16, "max_length": 8, "vocab_size": 200},
        ),
        optimizer=ComponentSpec(
            type="torch.optim:AdamW",
            params={"lr": 0.02, "weight_decay": 0.2, "betas": (0.8, 0.9)},
        ),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR",
            params={"step_size": 1, "gamma": 0.8},
        ),
        device_type="cpu",
        dp_mode="none",
        batch_per_device=2,
        ckpt_dir="unused",
    )
    values.update(overrides)
    return TrainConfig(**values)


def _save_checkpoint(path, *, with_optimizer=True):
    model, config = make_model("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(3):
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    extras = (
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        if with_optimizer
        else {}
    )
    checkpoint = Checkpoint(
        state_dict=model.state_dict(),
        config=config.to_dict(),
        epoch=0,
        consumed_samples=6,
        meta={"optimizer": {"type": "torch.optim:AdamW"}},
        extra=extras,
    )
    checkpoint.save(path)
    return checkpoint


@pytest.mark.parametrize(
    "flags,resume",
    [
        (None, False),
        ({"optimizer": False}, False),
        ({"scheduler": True}, False),
        ({"model": False, "optimizer": False}, False),
        ({"optimizer": False, "scheduler": True}, False),
        (None, True),
        ({"optimizer": False}, True),
        ({"scheduler": False}, True),
    ],
)
def test_builder_loads_selected_state_dicts(tmp_path, monkeypatch, flags, resume):
    checkpoint = _save_checkpoint(tmp_path)
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    loaded_extras = []
    loaded_weights = []
    original_load = checkpoint_module.load_torch
    original_weights = checkpoint_module.load_state_dict

    def record_load(path, **kwargs):
        loaded_extras.append(Path(path).name)
        return original_load(path, **kwargs)

    def record_weights(path, **kwargs):
        loaded_weights.append(Path(path).name)
        return original_weights(path, **kwargs)

    monkeypatch.setattr(checkpoint_module, "load_torch", record_load)
    monkeypatch.setattr(checkpoint_module, "load_state_dict", record_weights)
    kwargs = {} if flags is None else {"load_state_dict": flags}
    config = _config(**kwargs)
    options = config.state_dict_load_options(resume)
    context = TrainContextBuilder(config).with_param_path(str(tmp_path), resume).build()

    assert set(loaded_extras) == {
        name + ".pt" for name in ("optimizer", "scheduler") if options[name]
    }
    assert bool(loaded_weights) is options["model"]
    if options["model"]:
        for name, value in context.model.state_dict().items():
            torch.testing.assert_close(value, checkpoint.state_dict[name])
    else:
        assert any(
            not torch.equal(value, checkpoint.state_dict[name])
            for name, value in context.model.state_dict().items()
        )
    states = context.optimizer.state_dict()["state"]
    if options["optimizer"]:
        saved_states = list(checkpoint.extra["optimizer"]["state"].values())
        current_states = list(states.values())
        assert len(current_states) == len(saved_states)
        for current, saved in zip(current_states, saved_states, strict=True):
            for name in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(current[name], saved[name])
    else:
        assert not states

    assert context.consumed_samples == (6 if resume else 0)
    assert (context.checkpoint is not None) is resume
    group = context.optimizer.param_groups[0]
    scheduler = context.scheduler.state_dict()
    assert scheduler["last_epoch"] == (3 if options["scheduler"] else 0)
    assert scheduler["gamma"] == (0.5 if options["scheduler"] else 0.8)
    assert group["lr"] == pytest.approx(0.0125 if options["scheduler"] else 0.02)
    full_resume = resume and options["optimizer"] and options["scheduler"]
    assert group["weight_decay"] == (0.01 if full_resume else 0.2)
    assert group["betas"] == ((0.9, 0.999) if full_resume else (0.8, 0.9))
    for param in context.model.parameters():
        param.grad = torch.ones_like(param)
    context.optimizer.step()
    context.scheduler.step()
    assert group["lr"] == pytest.approx(0.00625 if options["scheduler"] else 0.016)


def test_weights_only_checkpoint_starts_with_fresh_optimizer(tmp_path, monkeypatch):
    _save_checkpoint(tmp_path, with_optimizer=False)
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    context = TrainContextBuilder(_config()).with_param_path(str(tmp_path)).build()
    assert not context.optimizer.state_dict()["state"]
    assert context.consumed_samples == 0


def test_changing_optimizer_requires_explicit_reset(tmp_path, monkeypatch):
    _save_checkpoint(tmp_path)
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    optimizer = ComponentSpec(type="torch.optim:SGD", params={"lr": 0.02})
    with pytest.raises(ValueError, match="load_state_dict.optimizer=False"):
        TrainContextBuilder(_config(optimizer=optimizer)).with_param_path(
            str(tmp_path)
        ).build()
    context = (
        TrainContextBuilder(
            _config(optimizer=optimizer, load_state_dict={"optimizer": False})
        )
        .with_param_path(str(tmp_path))
        .build()
    )
    assert isinstance(context.optimizer.optimizer, torch.optim.SGD)
    assert not context.optimizer.state_dict()["state"]


def test_unknown_state_dict_component_is_rejected():
    with pytest.raises(ValueError, match="Unknown load_state_dict components"):
        _config(load_state_dict={"optimzer": True})


def test_muon_adamw_reuses_both_histories_with_new_stage_lr():
    model = torch.nn.Linear(4, 4)
    old = OptimizerFactory.create_checked("muon_adamw", model, lr=0.1, ns_steps=1)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    old.step()
    new = OptimizerFactory.create_checked("muon_adamw", model, lr=0.02, ns_steps=1)
    scheduler = torch.optim.lr_scheduler.StepLR(new, step_size=1, gamma=0.8)
    saved = old.state_dict()

    load_optimizer_state_dict(new, saved)

    for name in ("muon", "adamw"):
        restored = getattr(new, name).state_dict()
        assert restored["state"]
        for param_id, state in saved[name]["state"].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(restored["state"][param_id][key], value)
        assert all(group["lr"] == 0.02 for group in getattr(new, name).param_groups)
    assert new.param_groups[0] is new.muon.param_groups[0]
    assert new.param_groups[-1] is new.adamw.param_groups[-1]
    new.step()
    scheduler.step()
    assert all(group["lr"] == pytest.approx(0.016) for group in new.param_groups)


@pytest.mark.parametrize(
    "legacy,step,cursor,batch,accum,expected_step,expected_cursor",
    [
        (True, 2, 5, 1, 2, 3, 6),
        (True, 3, 7, 1, 2, 3, 6),
        (False, 3, 5, 2, 2, 3, 5),
    ],
)
def test_resume_progress_formats(
    tmp_path,
    monkeypatch,
    legacy,
    step,
    cursor,
    batch,
    accum,
    expected_step,
    expected_cursor,
):
    checkpoint = _save_checkpoint(tmp_path)
    checkpoint.consumed_samples = cursor
    checkpoint.meta.update(
        optimizer_step=step,
        batch_per_device=batch,
        dp_size=1,
        grad_accum_steps=accum,
    )
    if legacy:
        repaired_step, repaired_cursor = repair_legacy_training_progress(checkpoint)
        checkpoint.meta["optimizer_step"] = repaired_step
        checkpoint.consumed_samples = repaired_cursor
    checkpoint.save(tmp_path)
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    context = (
        TrainContextBuilder(_config(batch_per_device=batch, grad_accum_steps=accum))
        .with_param_path(str(tmp_path), resume=True)
        .build()
    )
    assert context.optimizer_step == expected_step
    assert context.consumed_samples == expected_cursor
    assert context.pending_samples == 0
    assert context.scheduler.state_dict()["last_epoch"] == 3


@pytest.mark.parametrize(
    "step,cursor,batch,accum", [(2, 5, 1, 2), (3, 7, 1, 2), (2, 4, 2, 1)]
)
def test_raw_legacy_checkpoint_is_rejected_before_resume(
    tmp_path, monkeypatch, step, cursor, batch, accum
):
    checkpoint = _save_checkpoint(tmp_path)
    checkpoint.consumed_samples = cursor
    checkpoint.meta.update(
        optimizer_step=step, batch_per_device=batch, dp_size=1, grad_accum_steps=accum
    )
    checkpoint.save(tmp_path)
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    with pytest.raises(ValueError, match="scripts.tools.checkpoint"):
        (
            TrainContextBuilder(_config(batch_per_device=batch, grad_accum_steps=accum))
            .with_param_path(str(tmp_path), resume=True)
            .build()
        )


def test_checkpoint_repair_command_writes_resumable_copy(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "repaired"
    checkpoint = _save_checkpoint(source)
    checkpoint.consumed_samples = 5
    checkpoint.meta.update(
        optimizer_step=2, batch_per_device=1, dp_size=1, grad_accum_steps=2
    )
    checkpoint.save(source)
    (source / "tokenizer.json").write_text("{}")
    before = (source / "meta.json").read_bytes()
    result = CliRunner().invoke(repair_progress_command, [str(source), str(output)])
    assert result.exit_code == 0, result.output
    assert (source / "meta.json").read_bytes() == before
    assert (output / "tokenizer.json").read_text() == "{}"
    repaired = Checkpoint.load(output, verify_checksums=True)
    assert repaired.meta["optimizer_step"] == 3
    assert repaired.consumed_samples == 6
    for name, value in checkpoint.state_dict.items():
        torch.testing.assert_close(repaired.state_dict[name], value, rtol=0, atol=0)
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    context = (
        TrainContextBuilder(_config(batch_per_device=1, grad_accum_steps=2))
        .with_param_path(str(output), resume=True)
        .build()
    )
    assert context.optimizer_step == 3
    assert context.consumed_samples == 6
    assert context.scheduler.state_dict()["last_epoch"] == 3
    result = CliRunner().invoke(repair_progress_command, [str(source), str(output)])
    assert result.exit_code != 0
