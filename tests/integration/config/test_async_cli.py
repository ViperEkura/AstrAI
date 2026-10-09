"""Exercise CLI/YAML parameters all the way into the real TrainConfig."""

import pytest
import torch
from click.testing import CliRunner
from torch.utils.data import TensorDataset

from scripts.tools import train as train_module
from tests.support.models import make_rollout_config


def test_async_cli_yaml_and_explicit_flags_reach_training_config(tmp_path, monkeypatch):
    make_rollout_config().to_file(tmp_path / "config.json")
    yaml = tmp_path / "train.yaml"
    yaml.write_text(
        f"training:\n  train_type: online_grpo\n  param_path: {tmp_path}\n  data_root_path: {tmp_path}\n  dp_mode: none\n  rollout_mode: async_round\n  rollout_devices: [cuda:1, cuda:2]\n  rollout_interval: 1\n  rollout_worker_timeout_s: 91\n  rollout_startup_timeout_s: 73\n  rollout_weight_timeout_s: 123\n  async_train_microbatch_prompts: 2\n  reward_model: tests.integration.trainer.test_async_round:_Reward\n"
    )
    captured = []

    class Trainer:
        def __init__(self, config):
            captured.append(config)

        def train(self, **kwargs):
            assert kwargs == {"param_path": str(tmp_path), "resume": False}

    monkeypatch.setattr(train_module, "Trainer", Trainer)
    monkeypatch.setattr(
        train_module.DatasetFactory,
        "load",
        lambda **kwargs: TensorDataset(torch.ones(5)),
    )
    result = CliRunner().invoke(
        train_module.train_command,
        [
            "--config",
            str(yaml),
            "--rollout_devices",
            "cuda:3",
            "--rollout_temperature",
            "0.8",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    config = captured[0]
    assert config.rollout_mode == "async_round"
    assert config.rollout_devices == ["cuda:3"]
    assert config.rollout_temperature == 0.8
    assert (
        config.rollout_startup_timeout_s,
        config.rollout_worker_timeout_s,
        config.rollout_weight_timeout_s,
    ) == (73, 91, 123)
    assert config.async_train_microbatch_prompts == 2
    assert callable(config.reward_model_fn)
    assert config.strategy_kwargs["group_size"] == 4


def test_cli_rejects_unknown_yaml_parameters(tmp_path):
    yaml = tmp_path / "train.yaml"
    yaml.write_text("training:\n  learner_device: cuda:3\n")
    result = CliRunner().invoke(
        train_module.train_command, ["--config", str(yaml), "--dry-run"]
    )
    assert result.exit_code == 2
    assert "Unknown config keys: learner_device" in result.output


@pytest.mark.parametrize("reference", ["missing_separator", "math:pi"])
def test_reward_factory_must_be_importable_and_callable(reference):
    with pytest.raises(ValueError, match="reward_model"):
        train_module.load_reward_factory(reference)
