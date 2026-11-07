import dataclasses
import multiprocessing as mp
import pickle
from types import SimpleNamespace
from typing import List, Optional

import click
import pytest
import torch
from click.testing import CliRunner

from astrai.config import ComponentSpec, merge_yaml_into_kwargs
from astrai.config.cli import (
    OptSpec,
    apply_specs,
)
from astrai.dataset import DatasetFactory
from astrai.serialization import save_bin
from astrai.trainer.components import build_dataset_component
from scripts.tools import train as train_module
from tests.support.models import make_rollout_config


def _read_dataset_from_spawned_config(config, queue):
    dataset, _ = build_dataset_component(config.dataset, None, config.random_seed)
    queue.put((len(dataset), dataset[0]["input_ids"].tolist()))


@pytest.fixture
def training_configs(monkeypatch):
    configs = []

    def capture(config):
        configs.append(config)
        return SimpleNamespace(train=lambda **kwargs: None)

    monkeypatch.setattr(train_module, "Trainer", capture)
    return configs


def test_merge_yaml_overrides_defaults_but_not_explicit_cli(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "training:\n"
        "  optimizer: nora_nadamw\n"
        "  max_lr: 2e-4\n"
        "  nora_lr: 0.004\n"
        "  batch_per_device: 8\n",
        encoding="utf-8",
    )
    click_values = {
        "optimizer": "nora_nadamw",
        "max_lr": 3e-4,
        "nora_lr": 5e-3,
        "batch_per_device": 16,
    }

    merged = merge_yaml_into_kwargs(
        str(config_path), click_values, explicit_keys={"batch_per_device"}
    )

    assert merged["max_lr"] == 2e-4
    assert merged["nora_lr"] == 4e-3
    assert merged["batch_per_device"] == 16


def test_merge_yaml_parses_scientific_notation_as_float(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text("training:\n  max_lr: 2e-5\n", encoding="utf-8")

    merged = merge_yaml_into_kwargs(
        str(config_path), {"max_lr": 3e-4}, explicit_keys=set()
    )

    assert merged["max_lr"] == 2e-5
    assert isinstance(merged["max_lr"], float)


def test_merge_yaml_ignores_unknown_sections(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text("unknown:\n  max_lr: 1.0\n", encoding="utf-8")

    merged = merge_yaml_into_kwargs(str(config_path), {"max_lr": 3e-4})

    assert merged["max_lr"] == 3e-4


@pytest.mark.parametrize(
    "annotation,default,args,expected",
    [
        (Optional[int], None, ["--value", "4"], 4),
        (int | None, None, ["--value", "4"], 4),
        ("Optional[int]", None, ["--value", "4"], 4),
        (List[str], (), ["--value", "loss", "--value", "lr"], ("loss", "lr")),
        (list[str], (), ["--value", "loss"], ("loss",)),
        (bool, True, ["--no-value"], False),
    ],
)
def test_inferred_cli_options_parse_values(annotation, default, args, expected):
    config = dataclasses.make_dataclass(
        "Options", [("value", annotation, default)], namespace={"__module__": __name__}
    )

    @click.command()
    @apply_specs([OptSpec("value", "G")], config)
    def command(value):
        click.echo(repr(value))

    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert result.output.strip() == repr(expected)


def test_cli_rejects_unknown_yaml_parameters(tmp_path):
    yaml = tmp_path / "train.yaml"
    yaml.write_text("training:\n  learner_device: cuda:3\n")
    result = CliRunner().invoke(
        train_module.train_command, ["--config", str(yaml), "--dry-run"]
    )
    assert result.exit_code == 2
    assert "Unknown config keys: learner_device" in result.output


@pytest.mark.parametrize(
    "optimizer, flags, expected",
    [
        ("muon_adamw", ["--muon_ns_steps", "7"], {"ns_steps": 7}),
        ("nora_nadamw", ["--nora_lr", "0.02"], {"nora_lr": 0.02}),
        (
            "mano_adamw",
            ["--mano_momentum", "0.8", "--no-mano_nesterov"],
            {"momentum": 0.8, "nesterov": False},
        ),
    ],
)
def test_seq_cli_defers_dataset_construction(
    training_configs, tmp_path, monkeypatch, optimizer, flags, expected
):
    make_rollout_config(max_position_embeddings=8).to_file(tmp_path / "config.json")
    data_dir = tmp_path / "packed"
    save_bin(str(data_dir), {"sequence": [torch.arange(33)]})
    yaml = tmp_path / "train.yaml"
    yaml.write_text(
        f"training:\n  train_type: seq\n  param_path: {tmp_path}\n"
        f"  data_root_path: {data_dir}\n  window_size: 8\n  stride: 8\n"
        "  dp_mode: none\n  batch_per_device: 1\n  grad_accum_steps: 1\n"
    )
    captured = training_configs

    def reject_parent_load(**kwargs):
        raise AssertionError("DatasetFactory.load ran in the parent")

    monkeypatch.setattr(DatasetFactory, "load", reject_parent_load)
    result = CliRunner().invoke(
        train_module.train_command,
        ["--config", str(yaml), "--optimizer", optimizer, "--max_lr", "0.003", *flags],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(captured) == 1
    assert captured[0].dataset.type == "packed"
    assert captured[0].dataset.params["path"] == str(data_dir)
    assert isinstance(captured[0].dataset, ComponentSpec)
    assert captured[0].scheduler.type == "cosine"
    config = captured[0]
    assert config.optimizer.type == optimizer
    assert config.optimizer.params["lr"] == 0.003
    for key, value in expected.items():
        assert config.optimizer.params[key] == value
    assert config.to_dict()["optimizer_hyperparameters"] == config.optimizer.params


def test_cli_accepts_declarative_components(training_configs, tmp_path, monkeypatch):
    make_rollout_config(max_position_embeddings=8).to_file(tmp_path / "config.json")
    data_dir = tmp_path / "packed"
    save_bin(str(data_dir), {"sequence": [torch.arange(33)]})
    config_path = tmp_path / "components.yaml"
    config_path.write_text(
        "precision:\n  compute_dtype: float32\nmodel:\n  type: torch.nn:Linear\n"
        "  params: {in_features: 2, out_features: 2}\n"
        "optimizer:\n  type: torch.optim:SGD\n  params: {lr: 0.01}\n"
        "scheduler:\n  type: torch.optim.lr_scheduler:StepLR\n"
        '  params: {step_size: "$total_steps", gamma: 0.9}\n'
        f"dataset:\n  type: packed\n  params:\n    path: {data_dir}\n"
        "    train_type: seq\n    window_size: 8\n    stride: 8\n"
        f"training:\n  train_type: seq\n  param_path: {tmp_path}\n"
    )
    captured = training_configs
    result = CliRunner().invoke(
        train_module.train_command, ["--config", str(config_path)]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    config = captured[0]
    assert (config.model.type, config.optimizer.type, config.scheduler.type) == (
        "torch.nn:Linear",
        "torch.optim:SGD",
        "torch.optim.lr_scheduler:StepLR",
    )
    assert config.precision.compute_dtype == "float32"
    assert config.dataset.params["path"] == str(data_dir)
    dataset, _ = build_dataset_component(config.dataset, None, config.random_seed)
    assert len(dataset) > 0


def test_seq_config_factory_survives_spawn(training_configs, tmp_path, monkeypatch):
    make_rollout_config(max_position_embeddings=8).to_file(tmp_path / "config.json")
    data_dir = tmp_path / "packed"
    save_bin(str(data_dir), {"sequence": [torch.arange(33)]})
    yaml = tmp_path / "train.yaml"
    yaml.write_text(
        f"training:\n  train_type: seq\n  param_path: {tmp_path}\n"
        f"  data_root_path: {data_dir}\n  window_size: 8\n  stride: 8\n"
        "  dp_mode: none\n  batch_per_device: 1\n  grad_accum_steps: 1\n"
    )
    captured = training_configs
    result = CliRunner().invoke(train_module.train_command, ["--config", str(yaml)])
    assert result.exit_code == 0, (result.output, result.exception)
    config = captured[0]
    assert len(pickle.dumps(config)) < 100_000

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_read_dataset_from_spawned_config, args=(config, queue)
    )
    process.start()
    assert queue.get(timeout=30) == (4, list(range(8)))
    process.join(timeout=30)
    assert process.exitcode == 0


@pytest.mark.parametrize("reference", ["missing_separator", "math:pi"])
def test_reward_factory_must_be_importable_and_callable(reference):
    with pytest.raises(ValueError, match="reward_model"):
        train_module.load_reward_factory(reference)


def test_dry_run_uses_resolved_components_without_starting_trainer(
    tmp_path, monkeypatch
):
    make_rollout_config(max_position_embeddings=8).to_file(tmp_path / "config.json")
    path = tmp_path / "train.yaml"
    path.write_text(
        f"model:\n  param_path: {tmp_path}\n"
        f"data:\n  data_root_path: {tmp_path}\n"
        "training:\n  train_type: seq\n  batch_per_device: 2\n"
        "  grad_accum_steps: 3\n"
        "parallel:\n  dp_size: 2\n  dp_mode: ddp\n"
        "optimizer:\n  type: torch.optim:SGD\n  params: {lr: 2e-5}\n"
    )

    def reject_start(*args, **kwargs):
        raise AssertionError("dry-run started a Trainer")

    monkeypatch.setattr(train_module, "Trainer", reject_start)
    result = CliRunner().invoke(
        train_module.train_command, ["--config", str(path), "--dry-run"]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert "torch.optim:SGD" in result.output
    assert "'lr': 2e-05" in result.output
    rows = dict(
        [part.strip() for part in line.split(":", 1)]
        for line in result.output.splitlines()
        if " : " in line
    )
    assert rows["Global batch"].strip() == "12"
    assert "'window_size': 8" in result.output


@pytest.mark.parametrize("dry_run", [False, True])
def test_dry_run_and_execution_share_config_validation(tmp_path, monkeypatch, dry_run):
    make_rollout_config().to_file(tmp_path / "config.json")
    path = tmp_path / "train.yaml"
    path.write_text(
        f"training:\n  train_type: seq\n  param_path: {tmp_path}\n"
        f"  data_root_path: {tmp_path}\n  grad_accum_steps: 0\n"
    )

    def reject_start(*args, **kwargs):
        raise AssertionError("invalid configuration started a Trainer")

    monkeypatch.setattr(train_module, "Trainer", reject_start)
    result = CliRunner().invoke(
        train_module.train_command,
        ["--config", str(path)] + (["--dry-run"] if dry_run else []),
    )
    assert result.exit_code == 2
    assert "grad_accum_steps" in result.output
