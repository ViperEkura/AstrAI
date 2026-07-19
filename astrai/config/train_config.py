from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset

from astrai.config.base import BaseConfig
from astrai.model.components.lora import LoRAConfig


def required(**kw):
    return {"required": True, **kw}


@dataclass
class TrainConfig(BaseConfig):
    # basic setting
    model_fn: Callable[[], nn.Module] = field(
        default=None, metadata=required(help="Model factory for training.")
    )
    strategy: str = field(default=None, metadata=required(help="Training strategy."))
    dataset: Dataset = field(
        default=None, metadata=required(help="Dataset for training.")
    )
    optimizer_fn: Callable[[nn.Module], Optimizer] = field(
        default=None, metadata=required(help="Optimizer factory for training.")
    )
    scheduler_fn: Callable[[Optimizer], LRScheduler] = field(
        default=None, metadata=required(help="Scheduler factory for training.")
    )
    n_epoch: int = field(default=1, metadata={"help": "Number of epochs for training."})
    batch_per_device: int = field(
        default=4, metadata={"help": "Batch size per device."}
    )
    grad_accum_steps: int = field(
        default=1, metadata={"help": "Number of iterations between steps."}
    )
    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm. None disables clipping."},
    )
    gradient_checkpointing_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "Module types to enable activation checkpointing for."},
    )

    # checkpoint setting
    start_epoch: int = field(default=0, metadata={"help": "Start epoch for training."})
    start_samples: int = field(
        default=0,
        metadata={
            "help": "Start samples count (per rank). Superseded by checkpoint consumed_samples."
        },
    )
    ckpt_dir: str = field(
        default="./checkpoint", metadata={"help": "Checkpoint directory."}
    )
    ckpt_interval: int = field(
        default=5000,
        metadata={"help": "Number of optimizer steps between checkpoints."},
    )

    # lora setting
    lora: Optional[LoRAConfig] = field(
        default=None,
        metadata={"help": "LoRA config. None means full fine-tuning."},
    )

    # metric setting
    log_dir: str = field(
        default="./checkpoint/logs", metadata={"help": "Directory for metric logs."}
    )
    metrics: List[str] = field(
        default_factory=lambda: ["loss", "lr", "grad_norm"],
        metadata={"help": "Metrics to record during training."},
    )

    # dataloader setting
    random_seed: int = field(default=3407, metadata={"help": "Random seed."})
    num_workers: int = field(
        default=0, metadata={"help": "Number of workers for dataloader."}
    )
    prefetch_factor: Optional[int] = field(
        default=None, metadata={"help": "Prefetch factor for dataloader."}
    )
    pin_memory: bool = field(
        default=False, metadata={"help": "Pin memory for dataloader."}
    )
    collate_fn: Optional[Callable[[List[Any]], Any]] = field(
        default=None,
        metadata={"help": "Collate function for dataloader (e.g. dpo_collate_fn)."},
    )

    # distributed training
    nprocs: int = field(
        default=1, metadata={"help": "Number of processes for distributed training."}
    )
    backend: str = field(
        default="nccl", metadata={"help": "Distributed training backend."}
    )
    master_addr: str = field(
        default="localhost",
        metadata={"help": "Master address for distributed training."},
    )
    master_port: str = field(
        default="29500", metadata={"help": "Master port for distributed training."}
    )
    parallel_mode: str = field(
        default="none",
        metadata={"help": "Parallel strategy: none, ddp, fsdp."},
    )
    start_method: str = field(
        default="spawn",
        metadata={"help": "Multiprocessing start method (spawn/fork/forkserver)."},
    )

    # others
    device_type: str = field(
        default="cuda", metadata={"help": "Device type for distributed training."}
    )
    val_dataset: Optional[Dataset] = field(
        default=None, metadata={"help": "Dataset for validation."}
    )
    val_split: Optional[float] = field(
        default=None,
        metadata={
            "help": "Ratio to split from training dataset for validation (e.g. 0.05). Ignored if val_dataset is set."
        },
    )
    val_step: int = field(
        default=1000,
        metadata={"help": "Number of optimizer steps between validation runs."},
    )
    neftune_alpha: float = field(
        default=0.0,
        metadata={"help": "NEFTune noise alpha (0=disabled, typical: 5.0)."},
    )

    # online rollout
    rollout_interval: int = field(
        default=512,
        metadata={"help": "Number of optimizer steps between online rollouts."},
    )
    rollout_temperature: float = field(
        default=0.7, metadata={"help": "Sampling temperature for online rollout."}
    )
    rollout_top_k: int = field(
        default=0, metadata={"help": "Top-k filtering for online rollout (0=disable)."}
    )
    rollout_top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p (nucleus) filtering for online rollout."},
    )
    rollout_max_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum generated tokens per response in rollout."},
    )
    reward_model_fn: Optional[Callable] = field(
        default=None,
        metadata={
            "help": "Factory for reward model (required for online RL strategies)."
        },
    )

    executor_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Extra kwargs passed to ExecutorFactory.create()."},
    )
    extra_kwargs: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Other arguments."}
    )

    def __post_init__(self):
        self.validate()

    def validate(self):
        for fld in fields(self):
            if fld.metadata.get("required") and getattr(self, fld.name) is None:
                raise ValueError(f"TrainConfig.{fld.name} is required but got None.")
