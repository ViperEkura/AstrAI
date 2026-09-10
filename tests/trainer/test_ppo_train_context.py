"""Wiring tests for online PPO: config validation, critic assembly, and
checkpoint round-trip of critic state."""

import os
import subprocess
from pathlib import Path

import pytest
import torch

from astrai.config import TrainConfig
from astrai.model.transformer import AutoRegressiveLM
from astrai.model.value import ValueModel
from astrai.serialization import Checkpoint
from astrai.trainer.rollout import BaseRewardModel
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.train_callback import CheckpointCallback
from astrai.trainer.train_context import TrainContext, TrainContextBuilder
from astrai.trainer.trainer import Trainer
from tests.helpers import (
    FakeExecutor,
    build_test_tokenizer,
    make_model,
    make_rollout_config,
)


class _StubRewardModel(BaseRewardModel):
    def score(self, prompts, responses):
        return torch.zeros(len(prompts), len(responses[0]) if prompts else 0)


class _StubDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {"instruction": "hello", "input": ""}


def _stub_collate(batch):
    return {
        "instruction": [b["instruction"] for b in batch],
        "input": [b.get("input", "") for b in batch],
    }


def _ppo_config(device, **overrides):
    defaults = dict(
        strategy="online_ppo",
        model_fn=lambda: AutoRegressiveLM(make_rollout_config()),
        dataset=_StubDataset(),
        optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.0),
        scheduler_fn=lambda o: SchedulerFactory.create(
            "cosine", o, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
        ),
        reward_model_fn=_StubRewardModel,
        critic_model_fn=lambda: ValueModel(make_rollout_config()),
        collate_fn=_stub_collate,
        device_type=device,
        dp_mode="none",
        strategy_kwargs={"clip_eps": 0.2, "group_size": 2},
        rollout_interval=1,
        rollout_max_policy_lag=0,
        rollout_max_tokens=4,
        rollout_temperature=1.0,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_online_ppo_config_requires_critic_model_fn(device):
    with pytest.raises(ValueError, match="critic_model_fn is required"):
        _ppo_config(device, critic_model_fn=None)


def test_online_ppo_config_accepts_critic(device):
    config = _ppo_config(device)
    assert config.strategy == "online_ppo"


def test_create_critic_warm_starts_backbone_from_policy(device, monkeypatch):
    monkeypatch.setenv("LOCAL_DEVICE", device)
    model, config = make_model(device)
    cfg = _ppo_config(device)
    builder = TrainContextBuilder(cfg)
    context = TrainContext(model=model)

    critic, _ = builder._create_critic(context, FakeExecutor())

    policy_sd = model.state_dict()
    critic_sd = critic.state_dict()
    for key in policy_sd:
        assert torch.equal(critic_sd[key], policy_sd[key])
    assert torch.count_nonzero(critic_sd["value_head.weight"]) == 0
    assert torch.count_nonzero(critic_sd["value_head.bias"]) == 0


def test_create_critic_restores_checkpoint_extras(device, monkeypatch):
    monkeypatch.setenv("LOCAL_DEVICE", device)
    model, config = make_model(device)
    cfg = _ppo_config(device)
    builder = TrainContextBuilder(cfg)

    saved_critic = ValueModel(config).to(device)
    with torch.no_grad():
        saved_critic.value_head.weight.fill_(1.0)
    saved_optimizer = torch.optim.SGD(saved_critic.parameters(), lr=0.1)
    checkpoint = Checkpoint(
        state_dict=model.state_dict(),
        config=config.to_dict(),
        extra={
            "optimizer": {},
            "scheduler": {},
            "value_model": saved_critic.state_dict(),
            "value_optimizer": saved_optimizer.state_dict(),
        },
    )
    context = TrainContext(model=model, checkpoint=checkpoint)

    critic, critic_optimizer = builder._create_critic(context, FakeExecutor())

    assert torch.equal(
        critic.state_dict()["value_head.weight"],
        saved_critic.state_dict()["value_head.weight"],
    )
    assert (
        critic_optimizer.state_dict()["param_groups"]
        == saved_optimizer.state_dict()["param_groups"]
    )


def test_create_critic_resume_without_extras_fails_loudly(device, monkeypatch):
    monkeypatch.setenv("LOCAL_DEVICE", device)
    model, _ = make_model(device)
    cfg = _ppo_config(device)
    builder = TrainContextBuilder(cfg)
    checkpoint = Checkpoint(
        state_dict=model.state_dict(),
        extra={"optimizer": {}, "scheduler": {}},
    )
    context = TrainContext(model=model, checkpoint=checkpoint)

    with pytest.raises(
        ValueError, match="missing extras: value_model, value_optimizer"
    ):
        builder._create_critic(context, FakeExecutor())


def test_builder_resumes_critic_from_checkpoint(device, temp_dir, monkeypatch):
    """A full TrainContextBuilder resume restores the persisted critic."""
    monkeypatch.setenv("LOCAL_DEVICE", device)
    model, config = make_model(device)
    saved_critic = ValueModel(config).to(device)
    with torch.no_grad():
        saved_critic.value_head.weight.fill_(2.0)
    saved_optimizer = torch.optim.SGD(saved_critic.parameters(), lr=0.1)
    policy_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    policy_scheduler = SchedulerFactory.create(
        "cosine", policy_optimizer, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
    )
    checkpoint = Checkpoint(
        state_dict=model.state_dict(),
        epoch=0,
        consumed_samples=2,
        config=config.to_dict(),
        extra={
            "optimizer": policy_optimizer.state_dict(),
            "scheduler": policy_scheduler.state_dict(),
            "value_model": saved_critic.state_dict(),
            "value_optimizer": saved_optimizer.state_dict(),
        },
        meta={"policy_version": 3},
    )
    checkpoint.save(temp_dir)
    build_test_tokenizer(vocab_size=200).save_pretrained(temp_dir)

    cfg = _ppo_config(
        device,
        model_fn=lambda: AutoRegressiveLM(config),
        critic_model_fn=lambda: ValueModel(config),
        ckpt_dir=os.path.join(temp_dir, "ckpt"),
    )
    context = TrainContextBuilder(cfg).with_param_path(temp_dir, resume=True).build()

    assert isinstance(context.strategy.critic, ValueModel)
    assert torch.equal(
        context.strategy.critic.state_dict()["value_head.weight"],
        saved_critic.state_dict()["value_head.weight"],
    )
    assert context.strategy.policy_version == 3


def test_save_extra_persists_critic_state(device):
    model, _ = make_model(device)
    critic = ValueModel(make_rollout_config()).to(device)
    from astrai.trainer.strategy import PPOStrategy

    strategy = PPOStrategy(
        model=model,
        device=device,
        critic=critic,
        critic_optimizer=torch.optim.SGD(critic.parameters(), lr=0.0),
        executor=FakeExecutor(),
    )
    context = TrainContext(strategy=strategy)

    extra = CheckpointCallback.save_extra(context)

    assert set(extra) == {"value_model", "value_optimizer"}
    saved = extra["value_model"]
    live = critic.state_dict()
    assert set(saved) == set(live)
    for key in saved:
        assert torch.equal(saved[key], live[key])


def test_save_extra_without_critic_has_no_value_entries(device):
    model, _ = make_model(device)
    from astrai.trainer.strategy import GRPOStrategy
    from tests.helpers import make_frozen

    strategy = GRPOStrategy(
        model=model,
        device=device,
        old_model=None,
        ref_model=make_frozen(model, device),
        executor=FakeExecutor(),
    )
    context = TrainContext(strategy=strategy)

    extra = CheckpointCallback.save_extra(context)

    assert "value_model" not in extra
    assert "value_optimizer" not in extra


def test_trainer_default_callbacks_do_not_break_ppo(device, temp_dir):
    """The Trainer's default callback set constructs fine for online_ppo."""
    cfg = _ppo_config(device, ckpt_dir=os.path.join(temp_dir, "ckpt"))
    trainer = Trainer(cfg)
    assert trainer.callbacks


def test_sh_checkpoint_extra_files_detects_online_ppo(temp_dir):
    """The shell completeness helper derives PPO's extra required files."""
    lib = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "docker"
        / "lib"
        / "train-common.sh"
    )
    ppo_yaml = Path(temp_dir) / "ppo.yaml"
    ppo_yaml.write_text("train_type: online_ppo\n")
    grpo_yaml = Path(temp_dir) / "grpo.yaml"
    grpo_yaml.write_text('train_type: "online_grpo"\n')
    quoted_yaml = Path(temp_dir) / "quoted.yaml"
    quoted_yaml.write_text('  train_type:   "online_ppo"\n')

    def extra_files(yaml_path):
        script = f'source "{lib}"; checkpoint_extra_files "{yaml_path}"'
        result = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    assert extra_files(ppo_yaml) == "value_model.pt value_optimizer.pt"
    assert extra_files(quoted_yaml) == "value_model.pt value_optimizer.pt"
    assert extra_files(grpo_yaml) == ""
