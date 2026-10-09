"""Parameter ownership and asynchronous startup regressions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import TensorDataset

from astrai.config import TrainConfig
from astrai.trainer.rollout.configuration import resolve_async_rollout
from astrai.trainer.rollout.setup import configure_rollout
from astrai.trainer.rollout.types import SamplingParams
from astrai.trainer.train_context import TrainContextBuilder


def _config(**overrides):
    values = dict(
        model_fn=Mock(),
        dataset=TensorDataset(torch.ones(5)),
        optimizer_fn=Mock(),
        scheduler_fn=Mock(),
        strategy="online_grpo",
        reward_model_fn=Mock(),
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2"],
        rollout_interval=1,
    )
    values.update(overrides)
    return TrainConfig(**values)


@pytest.mark.parametrize("name", ["learner_device", "group_size", "rollout_devies"])
def test_unknown_training_fields_fail(name):
    with pytest.raises(ValueError, match=name):
        _config(**{name: 2})


@pytest.mark.parametrize(
    "name,value",
    [
        ("rl_update_epochs", 3),
        ("rl_minibatch_prompts", 2),
        ("gradient_chunked_logprobs", True),
        ("moe_aux_loss_coef", 0.5),
    ],
)
def test_strategy_kwargs_cannot_override_owned_fields(name, value):
    with pytest.raises(ValueError, match="not strategy_kwargs"):
        _config(strategy_kwargs={name: value})


@pytest.mark.parametrize("value", [0, 1, True, 1.5, "4"])
def test_invalid_grpo_group_size_fails_before_startup(value):
    with pytest.raises(ValueError, match="group_size"):
        _config(strategy_kwargs={"group_size": value})


@pytest.mark.parametrize(
    "name",
    [
        "rollout_temperature",
        "rollout_val_temperature",
        "rollout_worker_timeout_s",
        "rollout_startup_timeout_s",
        "rollout_weight_timeout_s",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_sampling_and_timeout_fields_fail(name, value):
    with pytest.raises(ValueError):
        _config(**{name: value})


def test_runtime_world_size_rejected_before_loading_models(monkeypatch):
    builder = TrainContextBuilder(_config())
    load = Mock()
    builder._load_preloaded_state = load
    monkeypatch.setattr("astrai.trainer.train_context.get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="actual world_size=1"):
        builder.build()
    load.assert_not_called()


def test_effective_grpo_defaults_and_all_worker_parameters(monkeypatch):
    cfg = _config(
        batch_per_device=5,
        rollout_startup_timeout_s=73,
        rollout_worker_timeout_s=91,
        rollout_weight_timeout_s=123,
        random_seed=42,
        rollout_temperature=0.8,
        rollout_top_k=7,
        rollout_top_p=0.75,
        rollout_max_tokens=16,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(ffn_type="mlp", max_position_embeddings=256),
        parameters=lambda: iter(
            [SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.bfloat16)]
        ),
    )
    context = SimpleNamespace(
        model=model,
        executor=SimpleNamespace(model_for_inference=lambda m: m),
        strategy=SimpleNamespace(
            group_size=4, supports_online=lambda: True, set_rollout_runner=Mock()
        ),
        consumed_samples=10,
        checkpoint=None,
        optimizer_step=2,
    )
    coordinator = Mock()
    monkeypatch.setattr(
        "astrai.trainer.rollout.setup.AsyncRoundCoordinator", coordinator
    )
    monkeypatch.setattr("torch.cuda.device_count", lambda: 3)
    configure_rollout(
        context, cfg, "tokenizer", {}, Mock(), lambda _: None, Mock(), Mock()
    )
    values = coordinator.call_args.kwargs
    assert values["params"] == SamplingParams(
        group_size=4, max_tokens=16, temperature=0.8, top_k=7, top_p=0.75
    )
    assert values["max_batch_size"] == 12
    assert values["max_prompts_per_worker"] == 3
    assert values["startup_timeout_s"] == 73
    assert values["worker_timeout_s"] == 91
    assert values["weight_timeout_s"] == 123
    assert values["random_seed"] == 42
    assert values["sample_cursor"] == 10
    assert cfg.rollout_max_policy_lag == values["max_policy_lag"] == 1
    assert values["model_dtype"] == torch.bfloat16
    assert values["devices"] == ["cuda:1", "cuda:2"]


@pytest.mark.parametrize("devices", [["cuda:0"], ["cuda:3"]])
def test_device_conflicts_fail_before_startup(monkeypatch, devices):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 3)
    with pytest.raises(ValueError, match="exclude|exceed"):
        resolve_async_rollout(
            _config(rollout_devices=devices),
            torch.device("cuda:0"),
            SamplingParams(group_size=4),
            256,
            0,
        )
