import os
import re
from collections import OrderedDict
from collections.abc import Callable
from functools import partial

import click
import torch
import yaml
from click.core import ParameterSource
from torch import optim

from astrai.config import AutoRegressiveLMConfig, TrainConfig
from astrai.config.train_config import (
    BACKENDS,
    PARALLEL_MODES,
    START_METHODS,
    TRAIN_TYPES,
)
from astrai.dataset import DatasetFactory, dpo_collate_fn, grpo_collate_fn
from astrai.model import AutoRegressiveLM
from astrai.model.components.decoder_block import DecoderBlock
from astrai.optim import OptimizerFactory
from astrai.trainer import SchedulerFactory, Trainer
from astrai.trainer.rollout import BaseRewardModel


class GroupedOption(click.Option):
    """A ``click.Option`` that carries a ``group`` label for help output."""

    def __init__(self, *args, group: str = "Options", **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group


class GroupedCommand(click.Command):
    """A ``click.Command`` that renders options grouped by their ``group``."""

    def format_options(self, ctx, formatter):
        groups: OrderedDict[str, list] = OrderedDict()
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is None:
                continue
            group = getattr(param, "group", "Options")
            groups.setdefault(group, []).append(record)
        for group_name, records in groups.items():
            with formatter.section(group_name):
                formatter.write_dl(records)


def opt(*param_decls, group: str, **kwargs):
    """Shorthand for ``click.option`` that tags the option with a group."""
    kwargs.setdefault("cls", GroupedOption)
    kwargs["group"] = group
    return click.option(*param_decls, **kwargs)


_YAML_FLOAT_PATTERN = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)


def _enable_yaml12_floats() -> None:
    """PyYAML implements YAML 1.1, where ``2e-5`` parses as a string; switch its
    float resolver to the YAML 1.2 core schema so scientific notation works."""
    yaml.SafeLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float", _YAML_FLOAT_PATTERN, list("-+0123456789.")
    )


def _merge_yaml_into_kwargs(
    config_path: str,
    passed_kwargs: dict,
    explicit_keys: set[str] | None = None,
) -> dict:
    """Merge Click defaults, YAML values, then explicit CLI values."""
    _enable_yaml12_floats()

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    merged = dict(passed_kwargs)
    for section in ("model", "data", "parallel", "training", "ckpt", "log"):
        if section in cfg:
            merged.update(cfg[section])

    if explicit_keys is None:
        explicit_keys = set(passed_kwargs)
    for key in explicit_keys:
        if key in passed_kwargs:
            merged[key] = passed_kwargs[key]

    return merged


_TRAIN_TYPE = sorted(TRAIN_TYPES)
_PARALLEL = sorted(PARALLEL_MODES)
_SCHEDULES = ["cosine", "sgdr", "wsd"]
_OPTIMIZERS = OptimizerFactory.list_registered()
_BACKENDS = sorted(BACKENDS)
_START_METHODS = sorted(START_METHODS)


@click.command(
    name="train",
    cls=GroupedCommand,
    help="Start model training (pretrain / SFT / DPO / GRPO).",
    context_settings={"show_default": True},
)
@opt(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True),
    group="Paths & Setup",
    help="YAML config file. CLI flags override YAML values.",
)
@opt(
    "--train_type",
    type=click.Choice(_TRAIN_TYPE),
    required=False,
    group="Paths & Setup",
    help="Training type.",
)
@opt(
    "--data_root_path",
    type=click.Path(exists=True),
    group="Paths & Setup",
    help="Root directory of the dataset.",
)
@opt(
    "--param_path",
    type=click.Path(exists=True),
    group="Paths & Setup",
    help="Path to model parameters or resume checkpoint.",
)
@opt(
    "--resume",
    is_flag=True,
    default=False,
    group="Paths & Setup",
    help="Resume from checkpoint.",
)
@opt("--n_epoch", type=int, default=1, group="Training", help="Number of epochs.")
@opt(
    "--batch_per_device",
    type=int,
    default=1,
    group="Training",
    help="Batch size per GPU.",
)
@opt(
    "--grad_accum_steps",
    type=int,
    default=1,
    group="Training",
    help="Gradient accumulation steps.",
)
@opt(
    "--warmup_ratio",
    type=float,
    default=0.05,
    group="LR Schedule",
    help="Fraction of total steps for LR warmup.",
)
@opt(
    "--max_lr",
    type=float,
    default=3e-4,
    group="Optimizer",
    help="Max learning rate.",
)
@opt(
    "--optimizer",
    type=click.Choice(_OPTIMIZERS),
    default="muon_adamw",
    group="Optimizer",
    help="Built-in optimizer.",
)
@opt(
    "--max_grad_norm",
    type=float,
    default=1.0,
    group="Training",
    help="Max gradient norm for clipping.",
)
@opt(
    "--weight_decay",
    type=float,
    default=0.1,
    group="Optimizer",
    help="Weight decay for eligible optimizer parameters.",
)
@opt(
    "--nora_lr", type=float, default=5e-3, group="Optimizer", help="Nora learning rate."
)
@opt(
    "--nora_beta", type=float, default=0.95, group="Optimizer", help="Nora EMA factor."
)
@opt(
    "--nora_momentum",
    type=float,
    default=0.95,
    group="Optimizer",
    help="Nora update momentum.",
)
@opt(
    "--nora_weight_decay",
    type=float,
    default=0.0,
    group="Optimizer",
    help="Nora weight decay.",
)
@opt(
    "--muon_momentum",
    type=float,
    default=0.95,
    group="Optimizer",
    help="Muon momentum factor.",
)
@opt(
    "--muon_nesterov/--no-muon_nesterov",
    default=True,
    group="Optimizer",
    help="Muon Nesterov.",
)
@opt(
    "--muon_ns_steps",
    type=int,
    default=5,
    group="Optimizer",
    help="Muon Newton-Schulz steps.",
)
@opt(
    "--muon_adjust_lr",
    type=click.Choice(["original", "match_rms_adamw"]),
    default="match_rms_adamw",
    group="Optimizer",
    help="Muon LR adjustment strategy.",
)
@opt(
    "--mano_momentum",
    type=float,
    default=0.95,
    group="Optimizer",
    help="Mano momentum factor.",
)
@opt(
    "--mano_nesterov/--no-mano_nesterov",
    default=True,
    group="Optimizer",
    help="Mano Nesterov momentum.",
)
@opt(
    "--random_seed",
    type=int,
    default=3407,
    group="Data Loading",
    help="Random seed.",
)
@opt(
    "--num_workers",
    type=int,
    default=4,
    group="Data Loading",
    help="DataLoader workers.",
)
@opt(
    "--pin_memory/--no-pin_memory",
    default=True,
    group="Data Loading",
    help="Pin memory.",
)
@opt(
    "--persistent_workers/--no-persistent_workers",
    default=True,
    group="Data Loading",
    help="Keep DataLoader workers alive between epochs.",
)
@opt(
    "--window_size",
    type=int,
    default=None,
    group="Data Loading",
    help="Max input sequence length.",
)
@opt(
    "--stride",
    type=int,
    default=None,
    group="Data Loading",
    help="Step size for sliding window.",
)
@opt("--dpo_beta", type=float, default=0.1, group="Algorithm", help="DPO beta.")
@opt("--group_size", type=int, default=4, group="Algorithm", help="GRPO group size.")
@opt(
    "--grpo_clip_eps",
    type=float,
    default=0.2,
    group="Algorithm",
    help="GRPO clip epsilon.",
)
@opt(
    "--grpo_clip_eps_low",
    type=float,
    default=None,
    group="Algorithm",
    help="Optional lower GRPO clip epsilon; defaults to --grpo_clip_eps.",
)
@opt(
    "--grpo_clip_eps_high",
    type=float,
    default=None,
    group="Algorithm",
    help="Optional upper GRPO clip epsilon for DAPO Clip-Higher.",
)
@opt(
    "--grpo_loss_aggregation",
    type=click.Choice(["token", "sequence"]),
    default="token",
    group="Algorithm",
    help="Aggregate GRPO loss by token (DAPO) or equally by sequence.",
)
@opt(
    "--grpo_overlong_max_len",
    type=int,
    default=None,
    group="Algorithm",
    help="Optional response length limit for DAPO soft overlong shaping.",
)
@opt(
    "--grpo_overlong_buffer_len",
    type=int,
    default=0,
    group="Algorithm",
    help="Length of the linear DAPO overlong penalty window.",
)
@opt(
    "--grpo_overlong_penalty_scale",
    type=float,
    default=1.0,
    group="Algorithm",
    help="Scale applied to the DAPO soft overlong penalty.",
)
@opt(
    "--grpo_kl_coef",
    type=float,
    default=0.01,
    group="Algorithm",
    help="GRPO KL penalty coefficient.",
)
@opt(
    "--label_smoothing",
    type=float,
    default=0.0,
    group="Data Loading",
    help="Label smoothing.",
)
@opt(
    "--moe_aux_loss_coef",
    type=float,
    default=0.01,
    group="Algorithm",
    help="MoE load balancing auxiliary loss coefficient (0=disable).",
)
@opt(
    "--rollout_interval",
    type=int,
    default=512,
    group="Algorithm",
    help="Steps between rollouts.",
)
@opt(
    "--rollout_max_policy_lag",
    type=int,
    default=None,
    group="Algorithm",
    help="Maximum accepted rollout/live policy-version gap.",
)
@opt(
    "--rollout_temperature",
    type=float,
    default=0.7,
    group="Algorithm",
    help="Rollout temperature.",
)
@opt(
    "--rollout_top_k",
    type=int,
    default=0,
    group="Algorithm",
    help="Rollout top-k (0=disable).",
)
@opt(
    "--rollout_top_p",
    type=float,
    default=0.9,
    group="Algorithm",
    help="Rollout top-p.",
)
@opt(
    "--rollout_max_tokens",
    type=int,
    default=1024,
    group="Algorithm",
    help="Max tokens per rollout response.",
)
@opt(
    "--gradient_checkpointing/--no-gradient_checkpointing",
    default=False,
    group="Misc",
    help="Enable activation checkpointing.",
)
@opt(
    "--compile",
    "compile_mode",
    type=click.Choice(["default", "reduce-overhead", "max-autotune"]),
    default=None,
    group="Misc",
    help="torch.compile mode. Omit to disable.",
)
@opt(
    "--ckpt_interval",
    type=int,
    default=5000,
    group="Checkpoint",
    help="Steps between checkpoints.",
)
@opt(
    "--ckpt_dir",
    type=click.Path(),
    default="checkpoint",
    group="Checkpoint",
    help="Checkpoint directory.",
)
@opt(
    "--val_split",
    type=float,
    default=None,
    group="Validation",
    help="Validation split ratio.",
)
@opt(
    "--val_step",
    type=int,
    default=1000,
    group="Validation",
    help="Steps between validation runs.",
)
@opt(
    "--metrics",
    multiple=True,
    default=("loss", "lr", "grad_norm", "grad_snr"),
    group="Validation",
    help="Metrics to log (repeatable).",
)
@opt("--start_epoch", type=int, default=0, group="Checkpoint", help="Start epoch.")
@opt(
    "--start_samples",
    type=int,
    default=0,
    group="Checkpoint",
    help="Start samples (per rank).",
)
@opt(
    "--master_addr",
    type=str,
    default="localhost",
    group="Distributed",
    help="Master node address.",
)
@opt(
    "--master_port",
    type=str,
    default="29500",
    group="Distributed",
    help="Master node port.",
)
@opt(
    "--backend",
    type=click.Choice(_BACKENDS),
    default="nccl",
    group="Distributed",
    help="Distributed backend.",
)
@opt("--nprocs", type=int, default=1, group="Distributed", help="Number of GPUs.")
@opt(
    "--parallel_mode",
    type=click.Choice(_PARALLEL),
    default="fsdp",
    group="Distributed",
    help="Parallel strategy.",
)
@opt(
    "--device_type",
    type=str,
    default="cuda",
    group="Distributed",
    help="Device type.",
)
@opt(
    "--start_method",
    type=click.Choice(_START_METHODS),
    default="spawn",
    group="Distributed",
    help="Multiprocessing start method.",
)
@opt(
    "--neftune_alpha",
    type=float,
    default=0.0,
    group="Algorithm",
    help="NEFTune noise alpha.",
)
@opt(
    "--schedule_type",
    type=click.Choice(_SCHEDULES),
    default="cosine",
    group="LR Schedule",
    help="LR scheduler.",
)
@opt(
    "--min_rate",
    type=float,
    default=None,
    group="LR Schedule",
    help="Minimum LR as fraction of base LR.",
)
@opt(
    "--cycle_length",
    type=int,
    default=None,
    group="LR Schedule",
    help="SGDR first cycle length.",
)
@opt(
    "--t_mult",
    type=int,
    default=2,
    group="LR Schedule",
    help="SGDR cycle length multiplier.",
)
@opt(
    "--stable_steps",
    type=int,
    default=None,
    group="LR Schedule",
    help="WSD stable plateau steps.",
)
@opt(
    "--decay_steps",
    type=int,
    default=None,
    group="LR Schedule",
    help="WSD decay steps.",
)
@opt(
    "--tp_size",
    type=int,
    default=None,
    group="Distributed",
    help="Tensor parallelism (future).",
)
@opt(
    "--dry-run",
    is_flag=True,
    default=False,
    group="Misc",
    help="Validate config and print plan, do not train.",
)
@click.pass_context
def train_command(ctx, config_path, dry_run, metrics, **kwargs):
    """Start model training (pretrain / SFT / DPO / GRPO)."""
    kwargs["metrics"] = metrics
    if config_path:
        explicit_keys = {
            key
            for key in kwargs
            if ctx.get_parameter_source(key) is ParameterSource.COMMANDLINE
        }
        kwargs = _merge_yaml_into_kwargs(config_path, kwargs, explicit_keys)

    required = ["train_type", "data_root_path", "param_path"]
    missing = [k for k in required if kwargs.get(k) is None]
    if missing:
        raise click.UsageError(
            f"Missing required options: {', '.join(missing)}. "
            f"Use --config YAML or provide them directly."
        )

    # Convert tuple back to list
    kwargs["metrics"] = list(kwargs["metrics"])
    # Remove tp_size (not yet wired)
    kwargs.pop("tp_size", None)

    if dry_run:
        _print_dry_run(kwargs)
        return

    train(**kwargs)


def _print_dry_run(kwargs: dict) -> None:
    """Print training plan summary."""
    rows = [
        ("Train type", kwargs.get("train_type")),
        ("Model path", kwargs.get("param_path")),
        ("Data path", kwargs.get("data_root_path")),
        ("Parallel mode", kwargs.get("parallel_mode", "none")),
        ("GPUs", str(kwargs.get("nprocs", 1))),
        ("Epochs", str(kwargs.get("n_epoch", 1))),
        ("Batch/device", str(kwargs.get("batch_per_device", 1))),
        ("Grad accum", str(kwargs.get("grad_accum_steps", 1))),
        ("Optimizer", str(kwargs.get("optimizer", "muon_adamw"))),
        ("Max LR", str(kwargs.get("max_lr", "?"))),
        ("Schedule", str(kwargs.get("schedule_type", "cosine"))),
        ("Warmup ratio", str(kwargs.get("warmup_ratio", 0.05))),
        ("Window size", str(kwargs.get("window_size", "config default"))),
        ("Checkpoint dir", str(kwargs.get("ckpt_dir", "checkpoint"))),
        ("Checkpoint interval", str(kwargs.get("ckpt_interval", 5000))),
        ("Resume", str(kwargs.get("resume", False))),
    ]
    max_len = max(len(k) for k, _ in rows)
    click.secho("\n=== Training Plan (dry-run) ===", fg="cyan", bold=True)
    for key, val in rows:
        click.echo(f"  {key:<{max_len}s} : {val}")
    click.secho("=" * 40, fg="cyan")


def create_model(config):
    return AutoRegressiveLM(config).to(dtype=torch.bfloat16)


def create_optimizer(
    model, optimizer_name: str = "muon_adamw", **kwargs
) -> optim.Optimizer:
    return OptimizerFactory.create(optimizer_name, model, **kwargs)


def create_scheduler(
    optimizer: optim.Optimizer, **kwargs
) -> optim.lr_scheduler.LRScheduler:
    schedule_type = kwargs.pop("schedule_type")
    return SchedulerFactory.create(schedule_type, optimizer, **kwargs)


def compute_total_steps(
    dataset_len: int,
    n_epoch: int,
    batch_per_device: int,
    nprocs: int,
    grad_accum_steps: int,
) -> int:

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    samples_per_replica = ceil_div(dataset_len, nprocs)
    batches_per_replica = ceil_div(samples_per_replica, batch_per_device)
    total_steps = (batches_per_replica // grad_accum_steps) * n_epoch
    return total_steps


def train(
    train_type: str,
    param_path: str,
    data_root_path: str,
    resume: bool,
    n_epoch: int,
    batch_per_device: int,
    start_epoch: int,
    start_samples: int,
    grad_accum_steps: int,
    warmup_ratio: float,
    ckpt_interval: int,
    ckpt_dir: str,
    val_split: float,
    val_step: int,
    metrics: list[str],
    max_grad_norm: float,
    random_seed: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    gradient_checkpointing: bool,
    window_size: int,
    stride: int,
    nprocs: int,
    parallel_mode: str,
    device_type: str,
    backend: str,
    master_addr: str,
    master_port: str,
    start_method: str,
    neftune_alpha: float,
    schedule_type: str,
    min_rate: float,
    cycle_length: int,
    t_mult: int,
    stable_steps: int,
    decay_steps: int,
    **kwargs,
):
    if train_type not in _TRAIN_TYPE:
        raise ValueError(
            f"Invalid train_type '{train_type}'. "
            f"Must be one of: {', '.join(_TRAIN_TYPE)}"
        )
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Model directory not found: {param_path}")
    if nprocs > 1 and parallel_mode == "none":
        raise ValueError("--nprocs > 1 requires --parallel_mode to be 'ddp' or 'fsdp'")

    # Load config
    config_path = os.path.join(param_path, "config.json")
    config = AutoRegressiveLMConfig.from_file(config_path)
    config.neftune_alpha = neftune_alpha

    if window_size is None:
        window_size = config.max_position_embeddings

    strategy_kwargs = {
        "beta": kwargs.pop("dpo_beta"),
        "label_smoothing": kwargs.pop("label_smoothing"),
        "clip_eps": kwargs.pop("grpo_clip_eps"),
        "clip_eps_low": kwargs.pop("grpo_clip_eps_low"),
        "clip_eps_high": kwargs.pop("grpo_clip_eps_high"),
        "loss_aggregation": kwargs.pop("grpo_loss_aggregation"),
        "overlong_max_len": kwargs.pop("grpo_overlong_max_len"),
        "overlong_buffer_len": kwargs.pop("grpo_overlong_buffer_len"),
        "overlong_penalty_scale": kwargs.pop("grpo_overlong_penalty_scale"),
        "kl_coef": kwargs.pop("grpo_kl_coef"),
        "group_size": kwargs.pop("group_size"),
    }

    rollout_interval = kwargs.pop("rollout_interval", 512)
    rollout_max_policy_lag = kwargs.pop("rollout_max_policy_lag", None)
    rollout_temperature = kwargs.pop("rollout_temperature", 0.7)
    rollout_top_k = kwargs.pop("rollout_top_k", 0)
    rollout_top_p = kwargs.pop("rollout_top_p", 0.9)
    rollout_max_tokens = kwargs.pop("rollout_max_tokens", 1024)
    reward_model_fn: Callable[[], BaseRewardModel] | None = None

    executor_kwargs = {}
    if parallel_mode == "ddp":
        executor_kwargs.update(
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    model_fn = partial(create_model, config)
    dataset = DatasetFactory.load(
        train_type=train_type,
        load_path=data_root_path,
        window_size=window_size,
        stride=stride,
        tokenizer_path=param_path,
    )

    optimizer_name = kwargs.pop("optimizer", "muon_adamw")
    optimizer_kwargs = {
        "lr": kwargs.pop("max_lr"),
        "weight_decay": kwargs.pop("weight_decay"),
        "nora_lr": kwargs.pop("nora_lr", 5e-3),
        "nora_beta": kwargs.pop("nora_beta", 0.95),
        "nora_momentum": kwargs.pop("nora_momentum", 0.95),
        "nora_weight_decay": kwargs.pop("nora_weight_decay", 0.0),
        "momentum": kwargs.pop("muon_momentum", 0.95),
        "nesterov": kwargs.pop("muon_nesterov", True),
        "ns_steps": kwargs.pop("muon_ns_steps", 5),
        "adjust_lr_fn": kwargs.pop("muon_adjust_lr", "match_rms_adamw"),
        "mano_momentum": kwargs.pop("mano_momentum", 0.95),
        "mano_nesterov": kwargs.pop("mano_nesterov", True),
    }
    optimizer_fn = partial(
        create_optimizer,
        optimizer_name=optimizer_name,
        **optimizer_kwargs,
    )
    if optimizer_name == "nora_nadamw":
        optimizer_hyperparameters = {
            key: optimizer_kwargs[key]
            for key in (
                "lr",
                "weight_decay",
                "nora_lr",
                "nora_beta",
                "nora_momentum",
                "nora_weight_decay",
            )
        }
        optimizer_hyperparameters.update(
            {"nadamw_betas": [0.9, 0.999], "nadamw_eps": 1e-8, "nora_eps": 1e-10}
        )
    elif optimizer_name == "mano_adamw":
        optimizer_hyperparameters = {
            key: optimizer_kwargs[key]
            for key in ("lr", "weight_decay", "mano_momentum", "mano_nesterov")
        }
        optimizer_hyperparameters.update(
            {"adamw_betas": [0.9, 0.95], "adamw_eps": 1e-8}
        )
    else:
        optimizer_hyperparameters = {
            key: optimizer_kwargs[key]
            for key in (
                "lr",
                "weight_decay",
                "momentum",
                "nesterov",
                "ns_steps",
                "adjust_lr_fn",
            )
        }

    total_steps = compute_total_steps(
        len(dataset), n_epoch, batch_per_device, nprocs, grad_accum_steps
    )
    warmup_steps = int(warmup_ratio * total_steps)
    warmup_steps = min(warmup_steps, total_steps)

    scheduler_kwargs = {"warmup_steps": warmup_steps}

    if schedule_type == "cosine":
        scheduler_kwargs["lr_decay_steps"] = total_steps - warmup_steps
    elif schedule_type == "sgdr":
        scheduler_kwargs["cycle_length"] = cycle_length or (total_steps - warmup_steps)
        scheduler_kwargs["t_mult"] = t_mult
    elif schedule_type == "wsd":
        remaining = total_steps - warmup_steps
        stable_steps_ = stable_steps or max(1, int(remaining * 0.8))
        scheduler_kwargs["stable_steps"] = stable_steps_
        scheduler_kwargs["decay_steps"] = max(
            1, decay_steps or (remaining - stable_steps_)
        )

    if min_rate is not None:
        scheduler_kwargs["min_rate"] = min_rate

    scheduler_fn = partial(
        create_scheduler,
        schedule_type=schedule_type,
        **scheduler_kwargs,
    )

    grad_ckpt_modules = [DecoderBlock] if gradient_checkpointing else []
    compile_mode = kwargs.pop("compile_mode", None)

    collate_fn = None
    if train_type == "dpo":
        collate_fn = dpo_collate_fn
    elif train_type == "grpo":
        collate_fn = grpo_collate_fn
    elif train_type in ("online_grpo", "online_dpo"):
        collate_fn = None

    train_config = TrainConfig(
        model_fn=model_fn,
        strategy=train_type,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        optimizer_name=optimizer_name,
        optimizer_hyperparameters=optimizer_hyperparameters,
        ckpt_dir=ckpt_dir,
        n_epoch=n_epoch,
        batch_per_device=batch_per_device,
        start_epoch=start_epoch,
        start_samples=start_samples,
        ckpt_interval=ckpt_interval,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        random_seed=random_seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        nprocs=nprocs,
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
        parallel_mode=parallel_mode,
        device_type=device_type,
        start_method=start_method,
        val_split=val_split,
        val_step=val_step,
        metrics=metrics,
        gradient_checkpointing_modules=grad_ckpt_modules,
        compile_mode=compile_mode,
        executor_kwargs=executor_kwargs,
        strategy_kwargs=strategy_kwargs,
        neftune_alpha=neftune_alpha,
        collate_fn=collate_fn,
        rollout_interval=rollout_interval,
        rollout_max_policy_lag=rollout_max_policy_lag,
        rollout_temperature=rollout_temperature,
        rollout_top_k=rollout_top_k,
        rollout_top_p=rollout_top_p,
        rollout_max_tokens=rollout_max_tokens,
        reward_model_fn=reward_model_fn,
        moe_aux_loss_coef=kwargs.pop("moe_aux_loss_coef", 0.01),
    )

    trainer = Trainer(train_config)
    trainer.train(param_path=param_path, resume=resume)


if __name__ == "__main__":
    train_command()
