import subprocess
from pathlib import Path

import pytest
import torch

from astrai.config import ComponentSpec, TrainConfig
from astrai.model.value import ValueModel
from astrai.serialization import Checkpoint
from astrai.trainer.context import TrainContext, TrainContextBuilder
from tests.support.models import make_model, make_rollout_config
from tests.support.rollout import FakeExecutor, LengthRewardModel


def _ppo_config(device, **overrides):
    defaults = dict(
        strategy="online_ppo",
        model=ComponentSpec(
            type="autoregressive_lm", params={"config": make_rollout_config().to_dict()}
        ),
        dataset=ComponentSpec(
            type="tests.support.rollout:InstructionDataset",
            params={
                "length": 2,
                "collate": "tests.support.rollout:instruction_collate_fn",
            },
        ),
        optimizer=ComponentSpec(type="torch.optim:SGD", params={"lr": 0.0}),
        scheduler=ComponentSpec(
            type="cosine",
            params={"warmup_steps": 1, "lr_decay_steps": 4, "min_rate": 0.05},
        ),
        reward_model_fn=LengthRewardModel,
        critic_model_fn=lambda: ValueModel(make_rollout_config()),
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


def test_create_critic_warm_starts_backbone_from_policy(cpu_device, monkeypatch):
    monkeypatch.setenv("LOCAL_DEVICE", cpu_device)
    model, config = make_model(cpu_device)
    cfg = _ppo_config(cpu_device)
    builder = TrainContextBuilder(cfg)
    context = TrainContext(model=model)

    critic, _ = builder._create_critic(context, FakeExecutor())

    policy_sd = model.state_dict()
    critic_sd = critic.state_dict()
    # The critic is trunk + value_head: every trunk key must transfer, and
    # the policy's lm_head is the one key it deliberately does not carry.
    for key in policy_sd:
        if key.startswith("lm_head."):
            assert key not in critic_sd
        else:
            assert torch.equal(critic_sd[key], policy_sd[key])
    assert torch.count_nonzero(critic_sd["value_head.weight"]) == 0
    assert torch.count_nonzero(critic_sd["value_head.bias"]) == 0


def test_create_critic_resume_without_extras_fails_loudly(cpu_device, monkeypatch):
    monkeypatch.setenv("LOCAL_DEVICE", cpu_device)
    model, _ = make_model(cpu_device)
    cfg = _ppo_config(cpu_device)
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
