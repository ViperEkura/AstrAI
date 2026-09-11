import os
from collections.abc import Callable
from functools import partial

import click
import torch
from click.core import ParameterSource
from torch import optim

from astrai.config import AutoRegressiveLMConfig, TrainConfig
from astrai.config.cli import (
    GroupedCommand,
    OptSpec,
    apply_specs,
    merge_yaml_into_kwargs,
)
from astrai.config.train_config import (
    BACKENDS,
    DP_MODES,
    START_METHODS,
    TRAIN_TYPES,
)
from astrai.dataset import DatasetFactory, dpo_collate_fn, grpo_collate_fn
from astrai.model import AutoRegressiveLM, ValueModel
from astrai.model.components.decoder_block import DecoderBlock
from astrai.optim import OptimizerFactory
from astrai.trainer import SchedulerFactory, Trainer
from astrai.trainer.rollout import BaseRewardModel

# Re-exported under its historical name for tests importing it from here.
_merge_yaml_into_kwargs = merge_yaml_into_kwargs

_TRAIN_TYPE = sorted(TRAIN_TYPES)
_DP = sorted(DP_MODES)
_SCHEDULES = ["cosine", "sgdr", "wsd"]
_OPTIMIZERS = OptimizerFactory.list_registered()
_BACKENDS = sorted(BACKENDS)
_START_METHODS = sorted(START_METHODS)

# Option table: types/defaults marked AUTO are inferred from TrainConfig
# fields; everything else (CLI-only options and default overrides) is
# declared inline. Table order is the --help order.
_SPECS = [
    OptSpec(
        "config_path",
        "Paths & Setup",
        type=click.Path(exists=True),
        param_decls=("--config", "-c", "config_path"),
        help="YAML config file. CLI flags override YAML values.",
    ),
    OptSpec("train_type", "Paths & Setup", choices=_TRAIN_TYPE, help="Training type."),
    OptSpec(
        "data_root_path",
        "Paths & Setup",
        type=click.Path(exists=True),
        help="Root directory of the dataset.",
    ),
    OptSpec(
        "param_path",
        "Paths & Setup",
        type=click.Path(exists=True),
        help="Path to model parameters or resume checkpoint.",
    ),
    OptSpec(
        "resume",
        "Paths & Setup",
        is_flag=True,
        default=False,
        help="Resume from checkpoint.",
    ),
    OptSpec("n_epoch", "Training", help="Number of epochs."),
    OptSpec("batch_per_device", "Training", default=1, help="Batch size per GPU."),
    OptSpec(
        "grad_accum_steps",
        "Training",
        help="Gradient accumulation steps.",
    ),
    OptSpec("max_grad_norm", "Training", help="Max gradient norm for clipping."),
    OptSpec(
        "warmup_ratio",
        "LR Schedule",
        type=float,
        default=0.05,
        help="Fraction of total steps for LR warmup.",
    ),
    OptSpec(
        "max_lr",
        "Optimizer",
        type=float,
        default=3e-4,
        help="Max learning rate.",
    ),
    OptSpec(
        "optimizer",
        "Optimizer",
        choices=_OPTIMIZERS,
        default="muon_adamw",
        help="Built-in optimizer.",
    ),
    OptSpec(
        "weight_decay",
        "Optimizer",
        type=float,
        default=0.1,
        help="Weight decay for eligible optimizer parameters.",
    ),
    OptSpec(
        "nora_lr", "Optimizer", type=float, default=5e-3, help="Nora learning rate."
    ),
    OptSpec(
        "nora_beta", "Optimizer", type=float, default=0.95, help="Nora EMA factor."
    ),
    OptSpec(
        "nora_momentum",
        "Optimizer",
        type=float,
        default=0.95,
        help="Nora update momentum.",
    ),
    OptSpec(
        "nora_weight_decay",
        "Optimizer",
        type=float,
        default=0.0,
        help="Nora weight decay.",
    ),
    OptSpec(
        "muon_momentum",
        "Optimizer",
        type=float,
        default=0.95,
        help="Muon momentum factor.",
    ),
    OptSpec(
        "muon_nesterov",
        "Optimizer",
        type=bool,
        default=True,
        help="Muon Nesterov.",
    ),
    OptSpec(
        "muon_ns_steps",
        "Optimizer",
        type=int,
        default=5,
        help="Muon Newton-Schulz steps.",
    ),
    OptSpec(
        "muon_adjust_lr",
        "Optimizer",
        choices=["original", "match_rms_adamw"],
        default="match_rms_adamw",
        help="Muon LR adjustment strategy.",
    ),
    OptSpec(
        "mano_momentum",
        "Optimizer",
        type=float,
        default=0.95,
        help="Mano momentum factor.",
    ),
    OptSpec(
        "mano_nesterov",
        "Optimizer",
        type=bool,
        default=True,
        help="Mano Nesterov momentum.",
    ),
    OptSpec("random_seed", "Data Loading", help="Random seed."),
    OptSpec("num_workers", "Data Loading", default=4, help="DataLoader workers."),
    OptSpec("pin_memory", "Data Loading", default=True, help="Pin memory."),
    OptSpec(
        "persistent_workers",
        "Data Loading",
        default=True,
        help="Keep DataLoader workers alive between epochs.",
    ),
    OptSpec(
        "window_size",
        "Data Loading",
        type=int,
        default=None,
        help="Max input sequence length.",
    ),
    OptSpec(
        "stride",
        "Data Loading",
        type=int,
        default=None,
        help="Step size for sliding window.",
    ),
    OptSpec(
        "label_smoothing",
        "Data Loading",
        type=float,
        default=0.0,
        help="Label smoothing.",
    ),
    OptSpec("dpo_beta", "Algorithm", type=float, default=0.1, help="DPO beta."),
    OptSpec("group_size", "Algorithm", type=int, default=4, help="GRPO group size."),
    OptSpec(
        "grpo_clip_eps",
        "Algorithm",
        type=float,
        default=0.2,
        help="GRPO clip epsilon.",
    ),
    OptSpec(
        "grpo_clip_eps_low",
        "Algorithm",
        type=float,
        default=None,
        help="Optional lower GRPO clip epsilon; defaults to --grpo_clip_eps.",
    ),
    OptSpec(
        "grpo_clip_eps_high",
        "Algorithm",
        type=float,
        default=None,
        help="Optional upper GRPO clip epsilon for DAPO Clip-Higher.",
    ),
    OptSpec(
        "grpo_loss_aggregation",
        "Algorithm",
        choices=["token", "sequence"],
        default="token",
        help="Aggregate GRPO loss by token (DAPO) or equally by sequence.",
    ),
    OptSpec(
        "grpo_overlong_max_len",
        "Algorithm",
        type=int,
        default=None,
        help="Optional response length limit for DAPO soft overlong shaping.",
    ),
    OptSpec(
        "grpo_overlong_buffer_len",
        "Algorithm",
        type=int,
        default=0,
        help="Length of the linear DAPO overlong penalty window.",
    ),
    OptSpec(
        "grpo_overlong_penalty_scale",
        "Algorithm",
        type=float,
        default=1.0,
        help="Scale applied to the DAPO soft overlong penalty.",
    ),
    OptSpec(
        "grpo_kl_coef",
        "Algorithm",
        type=float,
        default=0.01,
        help="GRPO KL penalty coefficient.",
    ),
    OptSpec(
        "ppo_gamma",
        "Algorithm",
        type=float,
        default=1.0,
        help="PPO reward discount factor.",
    ),
    OptSpec(
        "ppo_gae_lambda",
        "Algorithm",
        type=float,
        default=0.95,
        help="PPO GAE bias/variance trade-off.",
    ),
    OptSpec(
        "ppo_vf_coef",
        "Algorithm",
        type=float,
        default=0.5,
        help="PPO value-loss coefficient.",
    ),
    OptSpec(
        "moe_aux_loss_coef",
        "Algorithm",
        help="MoE load balancing auxiliary loss coefficient (0=disable).",
    ),
    OptSpec("rollout_interval", "Algorithm", help="Steps between rollouts."),
    OptSpec(
        "rollout_max_policy_lag",
        "Algorithm",
        help="Maximum accepted rollout/live policy-version gap.",
    ),
    OptSpec("rollout_temperature", "Algorithm", help="Rollout temperature."),
    OptSpec("rollout_top_k", "Algorithm", help="Rollout top-k (0=disable)."),
    OptSpec("rollout_top_p", "Algorithm", help="Rollout top-p."),
    OptSpec("rollout_max_tokens", "Algorithm", help="Max tokens per rollout response."),
    OptSpec("neftune_alpha", "Algorithm", help="NEFTune noise alpha."),
    OptSpec("val_split", "Validation", help="Validation split ratio."),
    OptSpec("val_step", "Validation", help="Steps between validation runs."),
    OptSpec(
        "metrics",
        "Validation",
        default=("loss", "lr", "grad_norm", "grad_snr"),
        help="Metrics to log (repeatable).",
    ),
    OptSpec("ckpt_interval", "Checkpoint", help="Steps between checkpoints."),
    OptSpec(
        "ckpt_dir",
        "Checkpoint",
        type=click.Path(),
        default="checkpoint",
        help="Checkpoint directory.",
    ),
    OptSpec("start_epoch", "Checkpoint", help="Start epoch."),
    OptSpec("start_samples", "Checkpoint", help="Start samples (per rank)."),
    OptSpec(
        "dp_size",
        "Distributed",
        help="Data-parallel replicas; total GPUs/processes = dp_size x cp_size.",
    ),
    OptSpec(
        "cp_size",
        "Distributed",
        type=int,
        default=None,
        help="Context parallelism: shard sequences across contiguous ranks (seq pretraining).",
    ),
    OptSpec(
        "tp_size",
        "Distributed",
        type=int,
        default=None,
        help="Tensor parallelism: shard Linear projections over features "
        "(attention heads / ffn channels).",
    ),
    OptSpec(
        "dp_mode",
        "Distributed",
        choices=_DP,
        default="fsdp",
        help="Data-parallel gradient-sync strategy (none/ddp/fsdp).",
    ),
    OptSpec("backend", "Distributed", choices=_BACKENDS, help="Distributed backend."),
    OptSpec("master_addr", "Distributed", help="Master node address."),
    OptSpec("master_port", "Distributed", help="Master node port."),
    OptSpec("device_type", "Distributed", help="Device type."),
    OptSpec(
        "start_method",
        "Distributed",
        choices=_START_METHODS,
        help="Multiprocessing start method.",
    ),
    OptSpec(
        "gradient_checkpointing",
        "Misc",
        type=bool,
        default=False,
        help="Enable activation checkpointing.",
    ),
    OptSpec(
        "compile_mode",
        "Misc",
        choices=["default", "reduce-overhead", "max-autotune"],
        param_decls=("--compile", "compile_mode"),
        help="torch.compile mode. Omit to disable.",
    ),
    OptSpec(
        "dry_run",
        "Misc",
        is_flag=True,
        default=False,
        param_decls=("--dry-run",),
        help="Validate config and print plan, do not train.",
    ),
    OptSpec(
        "schedule_type",
        "LR Schedule",
        choices=_SCHEDULES,
        default="cosine",
        help="LR scheduler.",
    ),
    OptSpec(
        "min_rate",
        "LR Schedule",
        type=float,
        default=None,
        help="Minimum LR as fraction of base LR.",
    ),
    OptSpec(
        "cycle_length",
        "LR Schedule",
        type=int,
        default=None,
        help="SGDR first cycle length.",
    ),
    OptSpec(
        "t_mult",
        "LR Schedule",
        type=int,
        default=2,
        help="SGDR cycle length multiplier.",
    ),
    OptSpec(
        "stable_steps",
        "LR Schedule",
        type=int,
        default=None,
        help="WSD stable plateau steps.",
    ),
    OptSpec(
        "decay_steps",
        "LR Schedule",
        type=int,
        default=None,
        help="WSD decay steps.",
    ),
]


@click.command(
    name="train",
    cls=GroupedCommand,
    help="Start model training (pretrain / SFT / DPO / GRPO).",
    context_settings={"show_default": True},
)
@apply_specs(_SPECS, TrainConfig)
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
    kwargs["tp_size"] = kwargs.pop("tp_size") or 1
    kwargs["cp_size"] = kwargs.pop("cp_size") or 1
    kwargs["dp_size"] = kwargs.pop("dp_size") or 1

    if dry_run:
        _print_dry_run(kwargs)
        return

    train(**kwargs)


def _print_dry_run(kwargs: dict) -> None:
    """Print training plan summary."""
    dp_size = kwargs.get("dp_size", 1) or 1
    cp_size = kwargs.get("cp_size", 1) or 1
    tp_size = kwargs.get("tp_size", 1) or 1
    rows = [
        ("Train type", kwargs.get("train_type")),
        ("Model path", kwargs.get("param_path")),
        ("Data path", kwargs.get("data_root_path")),
        ("DP mode", kwargs.get("dp_mode", "none")),
        ("DP replicas", str(dp_size)),
        ("CP size", str(cp_size)),
        ("TP size", str(tp_size)),
        ("GPUs", str(dp_size * cp_size * tp_size)),
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


def create_value_model(config):
    return ValueModel(config).to(dtype=torch.bfloat16)


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
    dp_size: int,
    grad_accum_steps: int,
) -> int:

    def ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    samples_per_replica = ceil_div(dataset_len, dp_size)
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
    dp_size: int,
    cp_size: int,
    tp_size: int,
    dp_mode: str,
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
    if dp_size > 1 and dp_mode == "none":
        raise ValueError("--dp_size > 1 requires --dp_mode to be 'ddp' or 'fsdp'")

    if cp_size > 1:
        if tp_size > 1:
            raise ValueError(
                "--cp_size > 1 combined with --tp_size > 1 is not verified "
                "yet: the ring-attention patch and head-sharded projections "
                "interact on the SDPA inputs"
            )
        if train_type not in ("seq", "sft"):
            raise ValueError(
                "--cp_size > 1 supports seq (pretrain) and sft only; RL "
                "strategies need cross-shard logprob handling and a "
                "context-parallel rollout path"
            )
        if dp_mode not in ("ddp", "fsdp"):
            raise ValueError("--cp_size > 1 requires --dp_mode ddp or fsdp")

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
        "gamma": kwargs.pop("ppo_gamma"),
        "gae_lambda": kwargs.pop("ppo_gae_lambda"),
        "vf_coef": kwargs.pop("ppo_vf_coef"),
    }

    rollout_interval = kwargs.pop("rollout_interval", 512)
    rollout_max_policy_lag = kwargs.pop("rollout_max_policy_lag", None)
    rollout_temperature = kwargs.pop("rollout_temperature", 0.7)
    rollout_top_k = kwargs.pop("rollout_top_k", 0)
    rollout_top_p = kwargs.pop("rollout_top_p", 0.9)
    rollout_max_tokens = kwargs.pop("rollout_max_tokens", 1024)
    reward_model_fn: Callable[[], BaseRewardModel] | None = None
    critic_model_fn = None
    if train_type == "online_ppo":
        # The optimizer defaults to the policy's; critic_optimizer_fn can
        # override it in the TrainConfig.
        critic_model_fn = partial(create_value_model, config)

    executor_kwargs = {}
    if dp_mode == "ddp":
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

    # The scheduler counts optimizer steps over data-parallel replicas; cp
    # peers split each batch's sequence rather than consuming extra samples.
    total_steps = compute_total_steps(
        len(dataset), n_epoch, batch_per_device, dp_size, grad_accum_steps
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
    elif train_type in ("online_grpo", "online_dpo", "online_ppo"):
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
        dp_size=dp_size,
        cp_size=cp_size,
        tp_size=tp_size,
        dp_mode=dp_mode,
        backend=backend,
        master_addr=master_addr,
        master_port=master_port,
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
        critic_model_fn=critic_model_fn,
        moe_aux_loss_coef=kwargs.pop("moe_aux_loss_coef", 0.01),
    )

    trainer = Trainer(train_config)
    trainer.train(param_path=param_path, resume=resume)


if __name__ == "__main__":
    train_command()
