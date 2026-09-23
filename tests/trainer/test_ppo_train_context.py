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
from astrai.trainer.backend import ColocatedBackend, P2PCopyPublisher, ReplicaBackend
from astrai.trainer.rollout import BaseRewardModel, RolloutEvaluator
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


def test_builder_wires_val_evaluator_with_inherited_params(
    device, temp_dir, monkeypatch
):
    """The rollout wiring builds a val evaluator whose SamplingParams
    inherit every unset field from the training rollout."""
    monkeypatch.setenv("LOCAL_DEVICE", device)
    build_test_tokenizer(vocab_size=200).save_pretrained(temp_dir)

    cfg = _ppo_config(
        device,
        model_fn=lambda: AutoRegressiveLM(make_rollout_config()),
        ckpt_dir=os.path.join(temp_dir, "ckpt"),
        rollout_val_temperature=0.0,
        rollout_val_group_size=1,
    )
    context = TrainContextBuilder(cfg).with_param_path(temp_dir).build()

    assert isinstance(context.val_evaluator, RolloutEvaluator)
    assert context.val_evaluator.params.temperature == 0.0
    assert context.val_evaluator.params.group_size == 1
    assert context.val_evaluator.params.max_tokens == cfg.rollout_max_tokens
    assert context.val_evaluator.params.top_p == cfg.rollout_top_p

    # Training-side defaults are untouched by the val overrides.
    runner = context.strategy._rollout_runner
    assert runner.generator.params.group_size == 2  # from strategy_kwargs
    assert runner.generator.params.temperature == cfg.rollout_temperature
    # The evaluator shares the training generator (same scheduler) until
    # a dedicated val backend is configured.
    assert context.val_evaluator.generator is runner.generator


def test_builder_wires_val_replica_and_publisher(device, temp_dir, monkeypatch):
    """rollout_val_device builds a dedicated replica backend for the val
    evaluator and registers a P2PCopyPublisher on the strategy."""
    monkeypatch.setenv("LOCAL_DEVICE", device)
    build_test_tokenizer(vocab_size=200).save_pretrained(temp_dir)

    cfg = _ppo_config(
        device,
        model_fn=lambda: AutoRegressiveLM(make_rollout_config()),
        ckpt_dir=os.path.join(temp_dir, "ckpt"),
        rollout_val_device=device,
    )
    context = TrainContextBuilder(cfg).with_param_path(temp_dir).build()

    train_generator = context.strategy._rollout_runner.generator
    assert isinstance(train_generator.backend, ColocatedBackend)
    assert isinstance(context.val_evaluator.generator.backend, ReplicaBackend)
    assert context.val_evaluator.generator is not train_generator

    publishers = context.strategy._weight_publishers
    assert len(publishers) == 1
    assert isinstance(publishers[0], P2PCopyPublisher)
    # The val replica started at the trainer's live policy version.
    assert (
        context.val_evaluator.generator.backend.policy_version
        == train_generator.backend.policy_version
    )


def test_builder_right_sizes_rollout_pool(device, temp_dir, monkeypatch):
    """rollout_pool_seq_len clamps the scheduler's KV budget below the
    model's context window (min semantics; None keeps the default)."""
    monkeypatch.setenv("LOCAL_DEVICE", device)
    build_test_tokenizer(vocab_size=200).save_pretrained(temp_dir)

    cfg = _ppo_config(
        device,
        model_fn=lambda: AutoRegressiveLM(make_rollout_config()),
        ckpt_dir=os.path.join(temp_dir, "ckpt"),
        rollout_pool_seq_len=32,
    )
    context = TrainContextBuilder(cfg).with_param_path(temp_dir).build()
    scheduler = context.strategy._rollout_runner.generator.backend.scheduler
    assert scheduler.max_seq_len == 32

    cfg = _ppo_config(
        device,
        model_fn=lambda: AutoRegressiveLM(make_rollout_config()),
        ckpt_dir=os.path.join(temp_dir, "ckpt2"),
    )
    context = TrainContextBuilder(cfg).with_param_path(temp_dir).build()
    scheduler = context.strategy._rollout_runner.generator.backend.scheduler
    assert scheduler.max_seq_len == make_rollout_config().max_position_embeddings


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
