from dataclasses import field
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from pydantic import ConfigDict, field_validator, model_validator
from pydantic.dataclasses import dataclass
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from astrai.config.base import BaseConfig
from astrai.model.components.lora import LoRAConfig

TRAIN_TYPES = frozenset(
    {"seq", "sft", "dpo", "grpo", "online_grpo", "online_dpo", "online_ppo"}
)
# Data-parallel gradient-sync strategies.
DP_MODES = frozenset({"none", "ddp", "fsdp"})
BACKENDS = frozenset({"nccl", "gloo"})
START_METHODS = frozenset({"spawn", "fork", "forkserver"})
_COMPILE_MODES = frozenset({"default", "reduce-overhead", "max-autotune"})
_ROUTE_RECOMPUTE_VALIDATION_MODES = frozenset({"off", "record", "error"})


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class TrainConfig(BaseConfig):
    """Training configuration.

    Combines hyperparameters with runtime objects (model_fn, dataset, etc.).
    Only JSON-serializable fields are written to checkpoint meta via to_dict().

    Args:
        model_fn (Callable[[], nn.Module]): Model factory for training.
        strategy (str): Training strategy (seq, sft, dpo, grpo, online_*).
        dataset (Dataset): Dataset for training.
        optimizer_fn (Callable[[nn.Module], Optimizer]): Optimizer factory for training.
        optimizer_name (Optional[str]): Serializable built-in optimizer identifier. Defaults to None.
        optimizer_hyperparameters (Dict[str, Any]): Serializable optimizer settings. Defaults to {}.
        scheduler_fn (Callable[[Optimizer], LRScheduler]): Scheduler factory for training.
        n_epoch (int): Number of epochs for training. Defaults to 1.
        batch_per_device (int): Batch size per device. Defaults to 4.
        grad_accum_steps (int): Number of iterations between optimizer steps. Defaults to 1.
        max_grad_norm (Optional[float]): Maximum gradient norm. None disables clipping. Defaults to 1.0.
        gradient_checkpointing_modules (List[type]): Module types to enable activation checkpointing for. Defaults to [].
        gradient_checkpointing_route_validation (str): Optional forward/recompute MoE route diagnostic mode: off, record, or error. Defaults to off.
        compile_mode (Optional[str]): torch.compile mode: 'default', 'reduce-overhead', 'max-autotune', or None. Defaults to None.
        start_epoch (int): Start epoch for training. Defaults to 0.
        start_samples (int): Start samples count (per rank). Superseded by checkpoint consumed_samples. Defaults to 0.
        ckpt_dir (str): Checkpoint directory. Defaults to "./checkpoint".
        ckpt_interval (int): Number of optimizer steps between checkpoints. Defaults to 5000.
        lora (Optional[LoRAConfig]): LoRA config. None means full fine-tuning. Defaults to None.
        metrics (List[str]): Metrics to record during training. Defaults to ["loss", "lr", "grad_norm"].
        random_seed (int): Random seed. Defaults to 3407.
        num_workers (int): Number of workers for dataloader. Defaults to 0.
        prefetch_factor (Optional[int]): Prefetch factor for dataloader. Defaults to None.
        persistent_workers (bool): Keep DataLoader workers alive between epochs. Defaults to False.
        pin_memory (bool): Pin memory for dataloader. Defaults to False.
        collate_fn (Optional[Callable[[List[Any]], Any]]): Collate function for dataloader (e.g. dpo_collate_fn). Defaults to None.
        dp_size (int): Data-parallel replicas; total training processes (``nprocs``) are derived as ``dp_size * cp_size * tp_size``. Defaults to 1.
        cp_size (int): Context-parallel group size: shards the sequence across contiguous ranks (seq pretraining, torch-native attention only). Defaults to 1.
        tp_size (int): Tensor-parallel group size: shards Linear projections over features (attention heads / ffn channels) using the default plan over the standard module layout. Defaults to 1.
        dp_mode (str): Data-parallel gradient-sync strategy: none, ddp, fsdp. Defaults to "none".
        backend (str): Distributed training backend. Defaults to "nccl".
        master_addr (str): Master address for distributed training. Defaults to "localhost".
        master_port (str): Master port for distributed training. Defaults to "29500".
        start_method (str): Multiprocessing start method: spawn/fork/forkserver. Defaults to "spawn".
        device_type (str): Device type for distributed training. Defaults to "cuda".
        val_dataset (Optional[Dataset]): Dataset for validation. Defaults to None.
        val_split (Optional[float]): Ratio to split from training dataset for validation, e.g. 0.05. Defaults to None.
        val_step (int): Number of optimizer steps between validation runs. Defaults to 1000.
        neftune_alpha (float): NEFTune noise alpha, 0=disabled, typical: 5.0. Defaults to 0.0.
        moe_aux_loss_coef (float): Weight applied to the MoE load-balancing loss. Defaults to 0.01.
        rollout_interval (int): Number of optimizer steps between online rollouts. Defaults to 512.
        rollout_max_policy_lag (Optional[int]): Maximum accepted gap between rollout and live policy versions. None derives ``rollout_interval - 1``. Defaults to None.
        rollout_temperature (float): Sampling temperature for online rollout. Defaults to 0.7.
        rollout_top_k (int): Top-k filtering for online rollout, 0=disable. Defaults to 0.
        rollout_top_p (float): Top-p (nucleus) filtering for online rollout. Defaults to 0.9.
        rollout_max_tokens (int): Maximum generated tokens per response in rollout. Defaults to 1024.
        reward_model_fn (Optional[Callable]): Factory for reward model, required for online RL strategies. Defaults to None.
        critic_model_fn (Optional[Callable]): Factory for the value (critic) model, required for online_ppo. Defaults to None.
        critic_optimizer_fn (Optional[Callable]): Factory for the critic optimizer; None reuses optimizer_fn. Defaults to None.
        executor_kwargs (Dict[str, Any]): Extra kwargs passed to ExecutorFactory.create(). Defaults to {}.
        strategy_kwargs (Dict[str, Any]): Extra strategy arguments. Defaults to {}.
    """

    model_fn: Callable[[], nn.Module]
    strategy: str
    dataset: Dataset
    optimizer_fn: Callable[[nn.Module], Optimizer]
    scheduler_fn: Callable[[Optimizer], LRScheduler]
    optimizer_name: Optional[str] = None
    optimizer_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    n_epoch: int = 1
    batch_per_device: int = 4
    grad_accum_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    gradient_checkpointing_modules: List[type] = field(default_factory=list)
    gradient_checkpointing_route_validation: str = "off"
    compile_mode: Optional[str] = None

    start_epoch: int = 0
    start_samples: int = 0
    ckpt_dir: str = "./checkpoint"
    ckpt_interval: int = 5000

    lora: Optional[LoRAConfig] = None

    metrics: List[str] = field(default_factory=lambda: ["loss", "lr", "grad_norm"])

    random_seed: int = 3407
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory: bool = False
    collate_fn: Optional[Callable[[List[Any]], Any]] = None

    dp_size: int = 1
    cp_size: int = 1
    tp_size: int = 1
    dp_mode: str = "none"
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "29500"
    start_method: str = "spawn"

    device_type: str = "cuda"
    val_dataset: Optional[Dataset] = None
    val_split: Optional[float] = None
    val_step: int = 1000
    neftune_alpha: float = 0.0
    moe_aux_loss_coef: float = 0.01

    rollout_interval: int = 512
    rollout_max_policy_lag: Optional[int] = None
    rollout_temperature: float = 0.7
    rollout_top_k: int = 0
    rollout_top_p: float = 0.9
    rollout_max_tokens: int = 1024
    reward_model_fn: Optional[Callable] = None
    critic_model_fn: Optional[Callable] = None
    critic_optimizer_fn: Optional[Callable] = None

    executor_kwargs: Dict[str, Any] = field(default_factory=dict)
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)

    @field_validator("strategy")
    def _validate_strategy(cls, v: str) -> str:
        if v not in TRAIN_TYPES:
            raise ValueError(
                f"strategy must be one of {sorted(TRAIN_TYPES)}, got {v!r}"
            )
        return v

    @field_validator("dp_mode")
    def _validate_dp_mode(cls, v: str) -> str:
        if v not in DP_MODES:
            raise ValueError(f"dp_mode must be one of {sorted(DP_MODES)}, got {v!r}")
        return v

    @field_validator("dp_size")
    def _validate_dp_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"dp_size must be >= 1, got {v}")
        return v

    @property
    def nprocs(self) -> int:
        """Total training processes: dp_size x cp_size x tp_size.

        The world size follows from the parallelism degrees the user
        configures, not the other way around — the sampler shards data over
        dp replicas only, so every batch-scaling formula wants dp_size, and
        divisibility by cp_size holds by construction.
        """
        return self.dp_size * self.cp_size * self.tp_size

    @field_validator("cp_size")
    def _validate_cp_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"cp_size must be >= 1, got {v}")
        return v

    @field_validator("tp_size")
    def _validate_tp_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"tp_size must be >= 1, got {v}")
        return v

    @field_validator("backend")
    def _validate_backend(cls, v: str) -> str:
        if v not in BACKENDS:
            raise ValueError(f"backend must be one of {sorted(BACKENDS)}, got {v!r}")
        return v

    @field_validator("start_method")
    def _validate_start_method(cls, v: str) -> str:
        if v not in START_METHODS:
            raise ValueError(
                f"start_method must be one of {sorted(START_METHODS)}, got {v!r}"
            )
        return v

    @field_validator("compile_mode")
    def _validate_compile_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {sorted(_COMPILE_MODES)} or None, got {v!r}"
            )
        return v

    @field_validator("gradient_checkpointing_route_validation")
    def _validate_route_recompute_validation(cls, v: str) -> str:
        if v not in _ROUTE_RECOMPUTE_VALIDATION_MODES:
            raise ValueError(
                "gradient_checkpointing_route_validation must be one of "
                f"{sorted(_ROUTE_RECOMPUTE_VALIDATION_MODES)}, got {v!r}"
            )
        return v

    @field_validator(
        "n_epoch",
        "batch_per_device",
        "grad_accum_steps",
        "ckpt_interval",
        "val_step",
        "rollout_interval",
        "rollout_max_tokens",
    )
    def _validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator("rollout_temperature")
    def _validate_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator("rollout_top_p")
    def _validate_top_p(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"rollout_top_p must be in (0, 1], got {v}")
        return v

    @field_validator(
        "rollout_top_k", "num_workers", "neftune_alpha", "moe_aux_loss_coef"
    )
    def _validate_non_negative(cls, v):
        if v < 0:
            raise ValueError(f"must be non-negative, got {v}")
        return v

    @field_validator("rollout_max_policy_lag")
    def _validate_optional_non_negative_int(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError(f"rollout_max_policy_lag must be non-negative, got {v}")
        return v

    @field_validator("max_grad_norm")
    def _validate_max_grad_norm(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f"max_grad_norm must be positive or None, got {v}")
        return v

    @field_validator("val_split")
    def _validate_val_split(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0 < v < 1:
            raise ValueError(f"val_split must be in (0, 1) or None, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_route_recompute_configuration(self) -> "TrainConfig":
        if self.gradient_checkpointing_route_validation != "off":
            if not self.gradient_checkpointing_modules:
                raise ValueError(
                    "gradient_checkpointing_route_validation requires "
                    "gradient_checkpointing_modules"
                )
            if self.compile_mode is not None:
                raise ValueError(
                    "gradient checkpoint route validation is not supported with "
                    "torch.compile"
                )
        return self

    @model_validator(mode="after")
    def _validate_online_strategy(self) -> "TrainConfig":
        if self.strategy.startswith("online_"):
            if self.reward_model_fn is None:
                raise ValueError(
                    f"reward_model_fn is required for online RL strategy "
                    f"{self.strategy!r}"
                )
            if self.strategy == "online_ppo" and self.critic_model_fn is None:
                raise ValueError(
                    "critic_model_fn is required for online RL strategy 'online_ppo'"
                )
            if self.nprocs > 1:
                raise ValueError(
                    f"online RL strategy {self.strategy!r} requires single-process "
                    f"training (nprocs=1): per-rank rollouts issue different "
                    f"numbers of forward passes and desynchronize the "
                    f"ddp/fsdp collectives, deadlocking NCCL"
                )
            if (
                self.rollout_max_policy_lag is not None
                and self.rollout_max_policy_lag < self.rollout_interval - 1
            ):
                raise ValueError(
                    f"rollout_max_policy_lag={self.rollout_max_policy_lag} "
                    f"cannot be below rollout_interval - 1 = "
                    f"{self.rollout_interval - 1}: the replay cache reuses "
                    f"rollouts up to that lag, so a tighter bound guarantees "
                    f"a fatal RolloutVersionError mid-training"
                )
        return self
