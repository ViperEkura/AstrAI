from dataclasses import fields, replace

import pytest

from astrai.config import ComponentSpec, TrainConfig


def _settings():
    return {
        "strategy": "seq",
        "dataset": ComponentSpec(type="tests.support.rollout:RandomTokenDataset"),
        "model": ComponentSpec(
            type="torch.nn:Linear", params={"in_features": 2, "out_features": 2}
        ),
        "scheduler": ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR", params={"step_size": 1}
        ),
    }


@pytest.mark.parametrize(
    "name, legacy, expected",
    [
        (
            "nora_nadamw",
            {"lr": 0.003, "nora_lr": 0.02, "nadamw_eps": 1e-8, "nora_eps": 1e-10},
            {"lr": 0.003, "nora_lr": 0.02},
        ),
        (
            "mano_adamw",
            {"mano_momentum": 0.8, "mano_nesterov": False, "adamw_betas": [0.9, 0.95]},
            {"momentum": 0.8, "nesterov": False},
        ),
        (
            "torch.optim:SGD",
            {"lr": 0.01, "momentum": 0.9},
            {"lr": 0.01, "momentum": 0.9},
        ),
    ],
)
def test_legacy_optimizer_inputs_normalize_once(name, legacy, expected):
    config = TrainConfig(
        **_settings(), optimizer_name=name, optimizer_hyperparameters=legacy
    )
    assert config.optimizer == ComponentSpec(type=name, params=expected)
    assert config.to_dict()["optimizer_hyperparameters"] == expected
    assert "optimizer_name" not in {entry.name for entry in fields(config)}
    assert "optimizer_hyperparameters" not in config.__dict__


def test_explicit_spec_wins_over_stale_legacy_metadata_and_round_trips():
    spec = ComponentSpec(type="torch.optim:SGD", params={"lr": 0.04})
    config = TrainConfig(
        **_settings(),
        optimizer=spec,
        optimizer_name="nora_nadamw",
        optimizer_hyperparameters={"lr": 99.0, "nora_lr": 88.0},
    )
    metadata = config.to_dict()
    assert config.optimizer == spec
    assert metadata["optimizer_name"] == spec.type
    assert metadata["optimizer_hyperparameters"] == spec.params
    assert TrainConfig.from_dict(metadata).optimizer == spec


def test_optimizer_metadata_follows_replaced_spec():
    config = TrainConfig(
        **_settings(),
        optimizer=ComponentSpec(type="torch.optim:SGD", params={"lr": 0.1}),
    )
    changed = replace(
        config, optimizer=ComponentSpec(type="torch.optim:AdamW", params={"lr": 0.01})
    )
    assert changed.optimizer_name == "torch.optim:AdamW"
    assert changed.optimizer_hyperparameters == {"lr": 0.01}
    assert config.optimizer_name == "torch.optim:SGD"


def test_positional_optimizer_spec_wins_over_legacy_metadata():
    settings = _settings()
    spec = ComponentSpec(type="torch.optim:SGD", params={"lr": 0.04})
    config = TrainConfig(
        settings["strategy"],
        settings["dataset"],
        settings["model"],
        spec,
        scheduler=settings["scheduler"],
        optimizer_name="nora_nadamw",
        optimizer_hyperparameters={"lr": 99.0},
    )
    assert config.optimizer == spec
    assert config.optimizer_name == spec.type
    assert config.optimizer_hyperparameters == spec.params
