import dataclasses
import multiprocessing as mp
import pickle
import typing as t
from typing import List, Optional

import click
import pytest
import torch
from click.testing import CliRunner
from torch.utils.data import TensorDataset

from astrai.config import ComponentSpec, TrainConfig, merge_yaml_into_kwargs
from astrai.config.cli import (
    GroupedCommand,
    OptSpec,
    apply_specs,
    option_from_spec,
)
from astrai.dataset import DatasetFactory
from astrai.model.components.mlp import MLP
from astrai.serialization import save_bin
from astrai.trainer.components import build_dataset_component
from scripts.tools import train as train_module
from tests.support.models import make_rollout_config


def _read_dataset_from_spawned_config(config, queue):
    dataset, _ = build_dataset_component(config.dataset, None, config.random_seed)
    queue.put((len(dataset), dataset[0]["input_ids"].tolist()))


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


def test_type_inference_from_config_fields():
    @click.command()
    @apply_specs(
        [
            OptSpec("n_epoch", "G"),
            OptSpec("rollout_max_policy_lag", "G"),
            OptSpec("pin_memory", "G"),
            OptSpec("metrics", "G"),
        ],
        TrainConfig,
    )
    def cmd(**kwargs):
        pass

    params = {p.name: p for p in cmd.params}
    assert params["n_epoch"].default == 1
    assert params["n_epoch"].type.name == "integer"
    assert params["rollout_max_policy_lag"].default is None
    assert params["rollout_max_policy_lag"].type.name == "integer"
    assert params["pin_memory"].secondary_opts == ["--no-pin_memory"]
    assert params["pin_memory"].default is False  # config default, no override
    assert params["metrics"].multiple
    assert params["metrics"].default == ("loss", "lr", "grad_norm")


def test_spec_overrides_beat_config_defaults():
    @click.command()
    @apply_specs(
        [
            OptSpec("num_workers", "G", default=4),
            OptSpec("dp_mode", "G", default="fsdp"),
        ],
        TrainConfig,
    )
    def cmd(**kwargs):
        pass

    params = {p.name: p for p in cmd.params}
    assert params["num_workers"].default == 4
    assert params["dp_mode"].default == "fsdp"


def test_cli_only_flag_pair_and_one_way_flag():
    @click.command()
    @apply_specs(
        [
            OptSpec("muon_nesterov", "G", type=bool, default=True),
            OptSpec("dry_run", "G", is_flag=True, default=False),
        ]
    )
    def cmd(**kwargs):
        pass

    params = {p.name: p for p in cmd.params}
    assert params["muon_nesterov"].secondary_opts == ["--no-muon_nesterov"]
    assert params["muon_nesterov"].default is True
    assert params["dry_run"].is_flag
    assert not params["dry_run"].secondary_opts


def test_uninferrable_type_raises_without_spec_type():
    with pytest.raises(TypeError, match="cannot infer"):
        option_from_spec(OptSpec("mystery", "G"), {})


def test_annotation_styles_old_and_new():
    """Optional[X], Union[X, None], X | None, List[str], and list[str] all
    infer the same click types."""

    @dataclasses.dataclass
    class OldStyle:
        opt_int: t.Optional[int] = None
        opt_float: t.Optional[float] = None
        union_int: t.Union[int, None] = None
        names: t.List[str] = dataclasses.field(default_factory=lambda: ["loss", "lr"])
        flag: bool = True

    @dataclasses.dataclass
    class NewStyle:
        opt_int: Optional[int] = None
        names: List[str] = dataclasses.field(default_factory=lambda: ["loss"])
        flag: bool = False

    @click.command()
    @apply_specs(
        [
            OptSpec("opt_int", "G"),
            OptSpec("opt_float", "G"),
            OptSpec("union_int", "G"),
            OptSpec("names", "G"),
            OptSpec("flag", "G"),
        ],
        OldStyle,
    )
    def old_cmd(**kwargs):
        pass

    params = {p.name: p for p in old_cmd.params}
    assert params["opt_int"].type.name == "integer"
    assert params["opt_int"].default is None
    assert params["opt_float"].type.name == "float"
    assert params["union_int"].type.name == "integer"
    assert params["names"].multiple
    assert params["names"].type.name == "text"
    assert params["flag"].secondary_opts == ["--no-flag"]
    assert params["flag"].default is True

    @click.command()
    @apply_specs(
        [
            OptSpec("opt_int", "G"),
            OptSpec("names", "G"),
            OptSpec("flag", "G"),
        ],
        NewStyle,
    )
    def new_cmd(**kwargs):
        pass

    params = {p.name: p for p in new_cmd.params}
    assert params["opt_int"].type.name == "integer"
    assert params["names"].multiple
    assert params["names"].default == ("loss",)
    assert params["flag"].secondary_opts == ["--no-flag"]
    assert params["flag"].default is False


def test_stringified_annotations_resolve_via_get_type_hints():
    """PEP 563 modules (``from __future__ import annotations``) leave
    ``Field.type`` as a string; hints resolution still infers types."""

    @dataclasses.dataclass
    class FutureStyle:
        opt_int: "t.Optional[int]" = None
        count: "int" = 3

    @click.command()
    @apply_specs(
        [
            OptSpec("opt_int", "G"),
            OptSpec("count", "G"),
        ],
        FutureStyle,
    )
    def cmd(**kwargs):
        pass

    params = {p.name: p for p in cmd.params}
    assert params["opt_int"].type.name == "integer"
    assert params["count"].type.name == "integer"
    assert params["count"].default == 3


def test_help_order_follows_spec_table():
    @click.command(cls=GroupedCommand)
    @apply_specs(
        [
            OptSpec("first", "G", type=int, default=1),
            OptSpec("second", "G", type=int, default=2),
        ]
    )
    def cmd(**kwargs):
        pass

    result = CliRunner().invoke(cmd, ["--help"])
    assert result.exit_code == 0
    assert result.output.index("--first") < result.output.index("--second")


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
        DatasetFactory,
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
            "--no-rollout_enable_overlap",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    config = captured[0]
    assert config.rollout_mode == "async_round"
    assert config.rollout_devices == ["cuda:3"]
    assert config.rollout_temperature == 0.8
    assert config.rollout_enable_overlap is False
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


def test_seq_cli_defers_dataset_construction(tmp_path, monkeypatch):
    make_rollout_config(max_position_embeddings=8).to_file(tmp_path / "config.json")
    data_dir = tmp_path / "packed"
    save_bin(str(data_dir), {"sequence": [torch.arange(33)]})
    yaml = tmp_path / "train.yaml"
    yaml.write_text(
        f"training:\n  train_type: seq\n  param_path: {tmp_path}\n"
        f"  data_root_path: {data_dir}\n  window_size: 8\n  stride: 8\n"
        "  dp_mode: none\n  batch_per_device: 1\n  grad_accum_steps: 1\n"
    )
    captured = []

    class Trainer:
        def __init__(self, config):
            captured.append(config)

        def train(self, **kwargs):
            pass

    def reject_parent_load(**kwargs):
        raise AssertionError("DatasetFactory.load ran in the parent")

    monkeypatch.setattr(train_module, "Trainer", Trainer)
    monkeypatch.setattr(DatasetFactory, "load", reject_parent_load)
    result = CliRunner().invoke(train_module.train_command, ["--config", str(yaml)])
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(captured) == 1
    assert captured[0].dataset.type == "packed"
    assert captured[0].dataset.params["path"] == str(data_dir)
    assert isinstance(captured[0].dataset, ComponentSpec)
    assert captured[0].scheduler.type == "cosine"


def test_cli_accepts_declarative_components(tmp_path, monkeypatch):
    make_rollout_config(max_position_embeddings=8).to_file(tmp_path / "config.json")
    data_dir = tmp_path / "packed"
    save_bin(str(data_dir), {"sequence": [torch.arange(33)]})
    config_path = tmp_path / "components.yaml"
    config_path.write_text(
        "model:\n  type: torch.nn:Linear\n  dtype: float32\n"
        "  params: {in_features: 2, out_features: 2}\n"
        "optimizer:\n  type: torch.optim:SGD\n  params: {lr: 0.01}\n"
        "scheduler:\n  type: torch.optim.lr_scheduler:StepLR\n"
        '  params: {step_size: "$total_steps", gamma: 0.9}\n'
        f"dataset:\n  type: packed\n  params:\n    path: {data_dir}\n"
        "    train_type: seq\n    window_size: 8\n    stride: 8\n"
        f"training:\n  train_type: seq\n  param_path: {tmp_path}\n"
    )
    captured = []

    class Trainer:
        def __init__(self, config):
            captured.append(config)

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(train_module, "Trainer", Trainer)
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
    assert config.model_dtype == "float32"
    assert config.dataset.params["path"] == str(data_dir)
    dataset, _ = build_dataset_component(config.dataset, None, config.random_seed)
    assert len(dataset) > 0


def test_seq_config_factory_survives_spawn(tmp_path, monkeypatch):
    make_rollout_config(max_position_embeddings=8).to_file(tmp_path / "config.json")
    data_dir = tmp_path / "packed"
    save_bin(str(data_dir), {"sequence": [torch.arange(33)]})
    yaml = tmp_path / "train.yaml"
    yaml.write_text(
        f"training:\n  train_type: seq\n  param_path: {tmp_path}\n"
        f"  data_root_path: {data_dir}\n  window_size: 8\n  stride: 8\n"
        "  dp_mode: none\n  batch_per_device: 1\n  grad_accum_steps: 1\n"
    )
    captured = []

    class Trainer:
        def __init__(self, config):
            captured.append(config)

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(train_module, "Trainer", Trainer)
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


def test_sft_cli_routes_linear_ce_options_from_yaml(tmp_path, monkeypatch):
    make_rollout_config().to_file(tmp_path / "config.json")
    yaml = tmp_path / "train.yaml"
    yaml.write_text(
        f"training:\n  train_type: sft\n  param_path: {tmp_path}\n"
        f"  data_root_path: {tmp_path}\n  dp_mode: none\n"
        "  loss_backend: cuda_linear_ce\n  loss_chunk_size: 128\n"
        "  gradient_checkpointing: true\n"
    )
    captured = []

    class Trainer:
        def __init__(self, config):
            captured.append(config)

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(train_module, "Trainer", Trainer)
    monkeypatch.setattr(
        DatasetFactory,
        "load",
        lambda **kwargs: TensorDataset(torch.ones(5)),
    )
    result = CliRunner().invoke(
        train_module.train_command,
        ["--config", str(yaml), "--loss_chunk_size", "256"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured[0].strategy_kwargs["loss_backend"] == "cuda_linear_ce"
    assert captured[0].strategy_kwargs["loss_chunk_size"] == 256
    assert captured[0].gradient_checkpointing_modules == [MLP]


@pytest.mark.parametrize("reference", ["missing_separator", "math:pi"])
def test_reward_factory_must_be_importable_and_callable(reference):
    with pytest.raises(ValueError, match="reward_model"):
        train_module.load_reward_factory(reference)


@pytest.mark.parametrize(
    "yaml_value,cli_flags,expected",
    [
        ("", [], {}),
        ("  load_state_dict: {optimizer: false}\n", [], {"optimizer": False}),
        (
            "  load_state_dict: {optimizer: false}\n",
            ["--load_state_dict", '{"model": true, "optimizer": true}'],
            {"model": True, "optimizer": True},
        ),
        (
            "  load_state_dict: {model: false, scheduler: true}\n",
            [],
            {"model": False, "scheduler": True},
        ),
    ],
)
def test_state_dict_loading_config_reaches_trainer(
    tmp_path, monkeypatch, yaml_value, cli_flags, expected
):
    make_rollout_config().to_file(tmp_path / "config.json")
    path = tmp_path / "train.yaml"
    path.write_text(
        f"training:\n  train_type: seq\n  param_path: {tmp_path}\n"
        f"  data_root_path: {tmp_path}\n  dp_mode: none\n" + yaml_value
    )
    captured = []

    class Trainer:
        def __init__(self, config):
            captured.append(config)

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(train_module, "Trainer", Trainer)
    result = CliRunner().invoke(
        train_module.train_command, ["--config", str(path), *cli_flags]
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured[0].load_state_dict == expected


@pytest.mark.parametrize(
    "strategy", ["seq", "sft", "dpo", "grpo", "online_grpo", "online_dpo", "online_ppo"]
)
def test_cli_options_instantiate_selected_strategy(tmp_path, monkeypatch, strategy):
    make_rollout_config().to_file(tmp_path / "config.json")
    path = tmp_path / "train.yaml"
    path.write_text(
        f"training:\n  train_type: {strategy}\n  param_path: {tmp_path}\n"
        f"  data_root_path: {tmp_path}\n  dp_mode: none\n"
        + (
            "  reward_model: tests.integration.trainer.test_async_round:_Reward\n"
            if strategy.startswith("online_")
            else ""
        )
    )
    captured = []

    class Trainer:
        def __init__(self, config):
            captured.append(config)

        def train(self, **kwargs):
            pass

    monkeypatch.setattr(train_module, "Trainer", Trainer)
    result = CliRunner().invoke(train_module.train_command, ["--config", str(path)])
    assert result.exit_code == 0, (result.output, result.exception)
    config = captured[0]
    model = torch.nn.Linear(2, 2)
    options = dict(config.strategy_kwargs)
    capabilities = train_module.StrategyFactory.capabilities(strategy)
    if capabilities.reference_model:
        options["ref_model"] = model
    if capabilities.old_model:
        options["old_model"] = model
    if capabilities.critic:
        options["critic"] = model
        options["critic_optimizer"] = torch.optim.SGD(model.parameters(), lr=0.1)
    instance = train_module.StrategyFactory.create_checked(
        strategy, model=model, device="cpu", **options
    )
    assert instance.model is model
