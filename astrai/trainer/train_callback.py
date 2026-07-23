import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import IO, Callable, List, Optional, Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm import tqdm

from astrai.factory import BaseFactory
from astrai.parallel import only_on_rank
from astrai.parallel.setup import get_current_device
from astrai.serialization import Checkpoint
from astrai.trainer.metric_util import (
    ctx_get_expert_load_cv,
    ctx_get_expert_load_max,
    ctx_get_expert_load_min,
    ctx_get_grad_norm,
    ctx_get_language_model_loss,
    ctx_get_loss,
    ctx_get_lr,
    ctx_get_router_aux_loss,
    ctx_get_router_entropy,
    ctx_get_router_loss,
    ctx_get_router_z_loss,
    ctx_get_val_loss,
)
from astrai.trainer.train_context import TrainContext

logger = logging.getLogger(__name__)


@runtime_checkable
class TrainCallback(Protocol):
    """
    Callback interface for trainer.
    """

    def on_train_begin(self, context: TrainContext):
        """Called at the beginning of training."""

    def on_train_end(self, context: TrainContext):
        """Called at the end of training."""

    def on_epoch_begin(self, context: TrainContext):
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, context: TrainContext):
        """Called at the end of each epoch."""

    def on_batch_begin(self, context: TrainContext):
        """Called at the beginning of each batch."""

    def on_batch_end(self, context: TrainContext):
        """Called at the end of each batch."""

    def on_optimizer_step(self, context: TrainContext):
        """Called on every optimizer step (sync step only)."""

    def on_error(self, context: TrainContext):
        """Called when an error occurs during training."""


class CallbackFactory(BaseFactory[TrainCallback]):
    """Factory for registering and creating training callbacks.

    Example:
        @CallbackFactory.register("my_callback")
        class MyCallback(TrainCallback):
            ...

        callback = CallbackFactory.create("my_callback", **kwargs)
    """


@CallbackFactory.register("gradient_clipping")
class GradientClippingCallback(TrainCallback):
    """
    Gradient clipping callback for trainer.
    """

    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = max_grad_norm

    def on_optimizer_step(self, context: TrainContext):
        context.grad_norm = context.executor.clip_grad_norm(
            context.model, self.max_grad_norm
        )


@CallbackFactory.register("gradient_checkpointing")
class GradientCheckpointingCallback(TrainCallback):
    """
    Activation checkpointing callback — trades compute for memory
    by recomputing specified module activations during the backward pass.

    Args:
        modules: Module types to apply checkpointing to.
    """

    def __init__(self, modules: Optional[List[type]] = None):
        self.modules = tuple(modules) if modules else ()

    def _enable(self, module: nn.Module):
        if self.modules and isinstance(module, self.modules):
            fn = module.forward
            module._original_forward = fn
            module.forward = lambda *a, **kw: torch_checkpoint(
                fn, *a, use_reentrant=False, **kw
            )

    @staticmethod
    def _disable(module: nn.Module):
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward

    def on_train_begin(self, context: TrainContext):
        context.model.apply(self._enable)
        logger.info("Gradient checkpointing enabled")

    def on_train_end(self, context: TrainContext):
        context.model.apply(self._disable)


@CallbackFactory.register("checkpoint")
class CheckpointCallback(TrainCallback):
    """
    Checkpoint callback for trainer.
    """

    extra_keys = ("optimizer", "scheduler")

    def __init__(
        self,
        save_dir: str,
        interval: int,
        weight_only: bool = False,
        save_extra_fn: Optional[Callable[["TrainContext"], dict]] = None,
    ):
        self.save_dir = save_dir
        self.interval = interval
        self.weight_only = weight_only
        self.save_extra_fn = save_extra_fn or CheckpointCallback.save_extra
        self.last_ckpt_step = None

    def on_train_begin(self, context: TrainContext):
        self.last_ckpt_step = context.optimizer_step

    def _save_checkpoint(self, context: TrainContext):
        self.last_ckpt_step = context.optimizer_step

        with context.executor.checkpoint_context(context.model) as state_dict:
            if state_dict is not None:
                save_path = os.path.join(
                    self.save_dir,
                    f"epoch_{context.epoch}_step_{context.optimizer_step}",
                )
                extra = self.save_extra_fn(context)
                meta = context.config.to_dict()
                context.checkpoint = Checkpoint(
                    state_dict=state_dict,
                    epoch=context.epoch,
                    consumed_samples=context.consumed_samples,
                    config=context.model_config,
                    extra=extra,
                    meta=meta,
                )
                context.checkpoint.save(save_path)

    def on_batch_end(self, context: TrainContext):
        if context.optimizer_step - self.last_ckpt_step >= self.interval:
            self._save_checkpoint(context)

    def on_train_end(self, context: TrainContext):
        if context.optimizer_step != self.last_ckpt_step:
            self._save_checkpoint(context)

    def on_error(self, context: TrainContext):
        self._save_checkpoint(context)

    @staticmethod
    def save_extra(context: TrainContext) -> dict:
        extra = {}
        for name in CheckpointCallback.extra_keys:
            obj = getattr(context, name, None)
            if obj:
                extra[name] = obj.state_dict()
        return extra


@CallbackFactory.register("progress_bar")
class ProgressBarCallback(TrainCallback):
    """
    Progress bar callback for trainer.
    """

    def __init__(
        self, num_epoch: int, log_interval: int = 100, file: Optional[IO[str]] = None
    ):
        self.num_epoch = num_epoch
        self.log_interval = log_interval
        self.file = file
        self.progress_bar: tqdm = None

    @only_on_rank(0)
    def on_epoch_begin(self, context: TrainContext):
        total_steps = len(context.dataloader) // context.executor.grad_accum_steps
        self.progress_bar = tqdm(
            total=total_steps,
            desc=f"Epoch {context.epoch + 1}/{self.num_epoch}",
            dynamic_ncols=True,
            file=self.file or sys.stdout,
        )

    @only_on_rank(0)
    def on_optimizer_step(self, context: TrainContext):
        postfix = {
            "step": f"{context.optimizer_step:d}",
            "loss": f"{context.loss:.4f}",
            "lr": f"{context.optimizer.param_groups[-1]['lr']:.2e}",
        }
        if context.grad_norm is not None:
            postfix["grad_norm"] = f"{context.grad_norm:.2f}"
        if context.val_loss is not None:
            postfix["val_loss"] = f"{context.val_loss:.4f}"
        if context.router_loss is not None:
            postfix["router"] = f"{context.router_loss:.3e}"
            postfix["load_cv"] = f"{context.expert_load_cv:.2f}"
        self.progress_bar.set_postfix(postfix)
        self.progress_bar.update(1)

    @only_on_rank(0)
    def on_epoch_end(self, context: TrainContext):
        _ = context
        if self.progress_bar:
            self.progress_bar.close()


@CallbackFactory.register("metric")
class MetricCallback(TrainCallback):
    def __init__(
        self,
        log_dir: str,
        save_interval: int,
        metrics: List[str] = None,
        val_step: int = 0,
    ):
        self.last_log_flush_step = None
        self.save_interval = save_interval
        self.metrics = metrics or ["loss", "lr"]
        self.val_step = val_step
        self._next_val_step = 0

        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_cache = []

        self._metric_funcs = {
            "loss": ctx_get_loss,
            "lr": ctx_get_lr,
            "val_loss": ctx_get_val_loss,
            "grad_norm": ctx_get_grad_norm,
            "language_model_loss": ctx_get_language_model_loss,
            "router_loss": ctx_get_router_loss,
            "router_aux_loss": ctx_get_router_aux_loss,
            "router_z_loss": ctx_get_router_z_loss,
            "router_entropy": ctx_get_router_entropy,
            "expert_load_min": ctx_get_expert_load_min,
            "expert_load_max": ctx_get_expert_load_max,
            "expert_load_cv": ctx_get_expert_load_cv,
        }

    def _metrics(self, context: TrainContext, names):
        return {
            m: self._metric_funcs[m](context)
            for m in names
            if self._metric_funcs[m](context) is not None
        }

    @only_on_rank(0)
    def _append(self, event_type: str, context: TrainContext, **extra):
        entry = {
            "type": event_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": context.epoch,
            "step": context.optimizer_step,
            "consumed_samples": context.consumed_samples,
            **extra,
        }
        self.log_cache.append(entry)

    def _run_validation(self, context: TrainContext) -> float:
        context.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in context.val_dataloader:
                loss = context.strategy(batch)
                total_loss += loss.item()
                num_batches += 1

        if context.world_size > 1 and dist.is_initialized():
            stats = torch.tensor(
                [total_loss, float(num_batches)], device=get_current_device()
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            avg_loss = (stats[0] / stats[1]).item()
        else:
            avg_loss = total_loss / max(num_batches, 1)

        context.model.train()
        return avg_loss

    def on_train_begin(self, context: TrainContext):
        self.last_log_flush_step = context.optimizer_step

    @only_on_rank(0)
    def _flush(self, epoch, step):
        log_file = self.log_dir / f"epoch_{epoch}_step_{step}_metric.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            for log in self.log_cache:
                f.write(json.dumps(log) + "\n")

    def on_optimizer_step(self, context):
        if (
            context.val_dataloader is not None
            and self.val_step > 0
            and context.optimizer_step >= self._next_val_step
        ):
            context.val_loss = self._run_validation(context)
            self._next_val_step = context.optimizer_step + self.val_step
            self._append("validation", context, val_loss=context.val_loss)

        step_metrics = [m for m in self.metrics if m != "val_loss"]
        self._append("step", context, **self._metrics(context, step_metrics))

        if context.optimizer_step - self.last_log_flush_step >= self.save_interval:
            self._flush(context.epoch, context.optimizer_step)
            self.last_log_flush_step = context.optimizer_step

    def on_epoch_end(self, context):
        self._append("epoch", context)

    def on_train_end(self, context):
        if (
            self.last_log_flush_step is None
            or context.optimizer_step != self.last_log_flush_step
        ):
            self._flush(context.epoch, context.optimizer_step)
            self.last_log_flush_step = context.optimizer_step

    def on_error(self, context):
        self._flush(context.epoch, context.optimizer_step)
