import argparse
import os
from functools import partial
from typing import Any, Dict

import torch
import torch.optim as optim
from torch import Tensor, nn

from astrai.config import AutoRegressiveLMConfig, TrainConfig
from astrai.dataset import DatasetFactory
from astrai.model import AutoRegressiveLM
from astrai.model.components.decoder_block import DecoderBlock
from astrai.trainer import SchedulerFactory, Trainer


class MuonMix(optim.Optimizer):
    """Combined Muon (matrix) + AdamW (non-matrix) optimizer."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, defaults)

        matrix_params: list[Tensor] = []
        other_params: list[Tensor] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.dim() >= 2
                and "norm" not in name
                and "bias" not in name
                and "embed" not in name
                and "lm_head" not in name
            ):
                matrix_params.append(param)
            else:
                other_params.append(param)

        self.muon = optim.Muon(
            matrix_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        self.adamw = optim.AdamW(
            [{"params": other_params, "weight_decay": 0.0}],
            lr=lr,
            betas=(0.9, 0.95),
            fused=True,
        )

        self.param_groups = [*self.muon.param_groups, *self.adamw.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        self.muon.step(closure)
        self.adamw.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = [*self.muon.param_groups, *self.adamw.param_groups]


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Train the AutoRegressiveLM model.")

    parser.add_argument(
        "--train_type",
        type=str,
        required=True,
        choices=["seq", "sft", "dpo", "grpo"],
        help="Train type.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        required=True,
        help="Path to the root directory of the dataset.",
    )
    parser.add_argument(
        "--param_path",
        type=str,
        required=True,
        help="Path to the model parameters or resume checkpoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from checkpoint at --param_path "
        "(restore epoch, consumed_samples, optimizer & scheduler state).",
    )

    parser.add_argument(
        "--n_epoch", type=int, default=1, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_per_device", type=int, default=1, help="Batch size per GPU."
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of iterations between each optimizer step.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for LR warmup.",
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4, help="Max learning rate for training."
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay (applied to Muon matrix params; non-matrix use 0).",
    )
    parser.add_argument(
        "--muon_momentum",
        type=float,
        default=0.95,
        help="Momentum factor for Muon optimizer.",
    )
    parser.add_argument(
        "--muon_nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Nesterov momentum for Muon.",
    )
    parser.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Newton-Schulz iteration steps for Muon.",
    )
    parser.add_argument(
        "--muon_adjust_lr",
        type=str,
        default="match_rms_adamw",
        choices=["original", "match_rms_adamw"],
        help="Muon learning rate adjustment strategy.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=3407, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--no_pin_memory",
        action="store_false",
        dest="pin_memory",
        help="Disable pin memory",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Max length of the input sequence.",
    )
    parser.add_argument(
        "--stride", type=int, default=None, help="Step size of the input sequence."
    )
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta value.")
    parser.add_argument("--group_size", type=int, default=4, help="GRPO group size.")
    parser.add_argument(
        "--grpo_clip_eps", type=float, default=0.2, help="GRPO clipping epsilon."
    )
    parser.add_argument(
        "--grpo_kl_coef", type=float, default=0.01, help="GRPO KL penalty coefficient."
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="cross_entropy function label smoothing parameter",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable activation checkpointing for DecoderBlock modules.",
    )

    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=5000,
        help="Number of iters between checkpoints.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoint",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=None,
        help="Ratio to split from training dataset for validation (e.g. 0.05).",
    )
    parser.add_argument(
        "--val_step",
        type=int,
        default=1000,
        help="Number of optimizer steps between validation runs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["loss", "lr", "grad_norm"],
        help="Metrics to log (e.g. --metrics loss lr val_loss). Default: loss lr grad_norm.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="checkpoint/logs",
        help="Directory for metric logs.",
    )
    parser.add_argument(
        "--start_epoch", type=int, default=0, help="Start epoch for training."
    )
    parser.add_argument(
        "--start_samples",
        type=int,
        default=0,
        help="Start samples (per rank) for training.",
    )

    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master node address for distributed training.",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="Master node port for distributed training.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="Distributed training backend.",
    )
    parser.add_argument("--nprocs", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="none",
        choices=["none", "ddp", "fsdp"],
        help="Parallel training strategy (none, ddp, fsdp).",
    )
    parser.add_argument(
        "--device_type", type=str, default="cuda", help="Device type to use."
    )
    parser.add_argument(
        "--start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method.",
    )
    parser.add_argument(
        "--neftune_alpha",
        type=float,
        default=0.0,
        help="NEFTune noise alpha (0=disabled, typical: 5.0).",
    )

    parser.add_argument(
        "--schedule_type",
        type=str,
        default="cosine",
        choices=["cosine", "sgdr", "wsd"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--min_rate",
        type=float,
        default=None,
        help="Minimum LR as fraction of base LR. Uses scheduler default if not set (cosine/sgdr: 0.05, wsd: 0.0).",
    )
    parser.add_argument(
        "--cycle_length",
        type=int,
        default=None,
        help="SGDR first cycle length in steps. Defaults to total_steps - warmup_steps.",
    )
    parser.add_argument(
        "--t_mult",
        type=int,
        default=2,
        help="SGDR cycle length multiplier per restart.",
    )
    parser.add_argument(
        "--stable_steps",
        type=int,
        default=None,
        help="WSD stable plateau steps. Required when --schedule_type wsd.",
    )
    parser.add_argument(
        "--decay_steps",
        type=int,
        default=None,
        help="WSD decay steps. Defaults to total_steps - warmup_steps - stable_steps.",
    )

    args = parser.parse_args()

    return args


def create_model(config):
    return AutoRegressiveLM(config).to(dtype=torch.bfloat16)


def create_optimizer(model, **kwargs) -> MuonMix:
    return MuonMix(model, **kwargs)


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
    log_dir: str,
    max_grad_norm: float,
    random_seed: int,
    num_workers: int,
    pin_memory: bool,
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
    assert train_type in ["seq", "sft", "dpo", "grpo"]
    assert os.path.exists(param_path)
    if nprocs > 1 and parallel_mode == "none":
        raise ValueError("--nprocs > 1 requires --parallel_mode to be 'ddp' or 'fsdp'")

    # Load config
    config_path = os.path.join(param_path, "config.json")
    config = AutoRegressiveLMConfig.from_file(config_path)
    config.neftune_alpha = neftune_alpha

    if window_size is None:
        window_size = config.max_len

    strategy_kwargs = {
        "beta": kwargs.pop("dpo_beta"),
        "label_smoothing": kwargs.pop("label_smoothing"),
        "clip_eps": kwargs.pop("grpo_clip_eps"),
        "kl_coef": kwargs.pop("grpo_kl_coef"),
        "group_size": kwargs.pop("group_size"),
    }

    executor_kwargs = {
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
    }

    model_fn = partial(create_model, config)
    dataset = DatasetFactory.load(
        train_type=train_type,
        load_path=data_root_path,
        window_size=window_size,
        stride=stride,
    )

    optimizer_fn = partial(
        create_optimizer,
        lr=kwargs.pop("max_lr"),
        weight_decay=kwargs.pop("weight_decay"),
        momentum=kwargs.pop("muon_momentum"),
        nesterov=kwargs.pop("muon_nesterov"),
        ns_steps=kwargs.pop("muon_ns_steps"),
        adjust_lr_fn=kwargs.pop("muon_adjust_lr"),
    )

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

    train_config = TrainConfig(
        model_fn=model_fn,
        strategy=train_type,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
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
        log_dir=log_dir,
        gradient_checkpointing_modules=grad_ckpt_modules,
        executor_kwargs=executor_kwargs,
        extra_kwargs=strategy_kwargs,
        neftune_alpha=neftune_alpha,
    )

    trainer = Trainer(train_config)
    trainer.train(param_path=param_path, resume=resume)


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
