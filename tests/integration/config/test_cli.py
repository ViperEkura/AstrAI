import dataclasses
import typing as t

import click
import pytest
from click.testing import CliRunner

from astrai.config import TrainConfig, merge_yaml_into_kwargs
from astrai.config.cli import (
    GroupedCommand,
    OptSpec,
    apply_specs,
    option_from_spec,
)


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
        opt_int: int | None = None
        names: list[str] = dataclasses.field(default_factory=lambda: ["loss"])
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
