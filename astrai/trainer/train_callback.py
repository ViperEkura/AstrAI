import json
import logging
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import IO, Callable, List, Optional, Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm import tqdm

from astrai.factory import BaseFactory
from astrai.moe import (
    RecomputeRouteMismatchError,
    RouteCheckpointPairV0,
    RouteRecomputeDiagnosticsV0,
    RouteRecomputeSummaryV0,
    synchronize_route_recompute_summary,
)
from astrai.parallel import only_on_rank
from astrai.parallel.setup import get_current_device
from astrai.serialization import Checkpoint
from astrai.trainer.metric_util import (
    ctx_get_grad_norm,
    ctx_get_grad_snr,
    ctx_get_loss,
    ctx_get_lr,
    ctx_get_moe_metric,
    ctx_get_val_loss,
)
from astrai.trainer.train_context import TrainContext

logger = logging.getLogger(__name__)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _copy_tokenizer_files(param_path: Optional[str], save_path: str):
    """Snapshot tokenizer files into the checkpoint directory.

    ``param_path`` is the launch model directory (or, on resume, a
    previous self-contained checkpoint), so the copy makes every
    checkpoint independently resumable for online training, which
    loads its tokenizer from ``param_path``.
    """
    if not param_path:
        return
    for name in _TOKENIZER_FILES:
        src = os.path.join(param_path, name)
        dst = os.path.join(save_path, name)
        if not os.path.isfile(src) or (
            os.path.isfile(dst) and os.path.samefile(src, dst)
        ):
            continue
        shutil.copy2(src, dst)


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

    def before_optimizer_step(self, context: TrainContext):
        """Called immediately before every optimizer step (sync step only)."""

    def after_optimizer_step(self, context: TrainContext):
        """Called after the optimizer and scheduler step (sync step only)."""

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

    def before_optimizer_step(self, context: TrainContext):
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
        route_validation: ``"off"`` preserves the existing checkpoint path,
            ``"record"`` publishes diagnostics, and ``"error"`` also aborts
            before the optimizer step if a route differs or is incomplete.
    """

    _ROUTE_VALIDATION_MODES = frozenset(("off", "record", "error"))

    def __init__(
        self,
        modules: Optional[List[type]] = None,
        route_validation: str = "off",
    ):
        if route_validation not in self._ROUTE_VALIDATION_MODES:
            raise ValueError(
                "route_validation must be one of "
                f"{sorted(self._ROUTE_VALIDATION_MODES)}, got {route_validation!r}"
            )
        self.modules = tuple(modules) if modules else ()
        self.route_validation = route_validation
        self.route_diagnostics = RouteRecomputeDiagnosticsV0()
        self.last_route_summary = RouteRecomputeSummaryV0()

    def _checkpoint_forward(self, fn, *args, **kwargs):
        if self.route_validation == "off":
            return torch_checkpoint(fn, *args, use_reentrant=False, **kwargs)

        pair = RouteCheckpointPairV0(self.route_diagnostics)

        def observed_forward(*inner_args, **inner_kwargs):
            output = fn(*inner_args, **inner_kwargs)
            pair.observe(output)
            return output

        # Full recomputation is required so the observer after ``fn`` always
        # sees the second route. This changes work only in the opt-in modes.
        return torch_checkpoint(
            observed_forward,
            *args,
            use_reentrant=False,
            context_fn=pair.context_fn,
            early_stop=False,
            **kwargs,
        )

    def _enable(self, module: nn.Module):
        if self.modules and isinstance(module, self.modules):
            fn = module.forward
            module._original_forward = fn
            module.forward = lambda *a, **kw: self._checkpoint_forward(fn, *a, **kw)

    @staticmethod
    def _disable(module: nn.Module):
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward

    def on_train_begin(self, context: TrainContext):
        if not self.modules:
            return
        self.route_diagnostics.reset()
        self.last_route_summary = RouteRecomputeSummaryV0()
        context.model.apply(self._enable)
        logger.info(
            "Gradient checkpointing enabled (route validation: %s)",
            self.route_validation,
        )

    def on_batch_begin(self, context: TrainContext):
        _ = context
        if self.route_validation != "off":
            self.route_diagnostics.reset()

    def on_batch_end(self, context: TrainContext):
        if self.route_validation == "off":
            return
        local_summary = self.route_diagnostics.snapshot()
        summary = synchronize_route_recompute_summary(
            local_summary,
            device=(
                get_current_device()
                if dist.is_available() and dist.is_initialized()
                else None
            ),
        )
        self.last_route_summary = summary
        context.metrics.update(summary.to_metrics())
        if not summary.has_failure:
            return

        message = (
            "MoE route mismatch across checkpoint recomputation: "
            f"mismatch_pairs={summary.mismatch_pair_count}, "
            f"invalid_pairs={summary.invalid_pair_count}, "
            f"unrecomputed_pairs={summary.unrecomputed_pair_count}, "
            f"rank_inconsistent={summary.rank_observation_inconsistent}"
        )
        if self.route_validation == "error":
            raise RecomputeRouteMismatchError(message)
        if context.rank == 0:
            logger.warning(message)

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
        self._saved = False

    def on_train_begin(self, context: TrainContext):
        self.last_ckpt_step = context.optimizer_step

    def _save_checkpoint(self, context: TrainContext):
        with context.executor.checkpoint_context(context.model) as state_dict:
            if state_dict is not None:
                save_path = os.path.join(
                    self.save_dir,
                    f"epoch_{context.epoch}_step_{context.optimizer_step}",
                )
                extra = self.save_extra_fn(context)
                meta = {
                    **context.config.to_dict(),
                    "optimizer_step": context.optimizer_step,
                }
                policy_version = context.strategy.policy_version
                if policy_version is not None:
                    meta["policy_version"] = policy_version
                context.checkpoint = Checkpoint(
                    state_dict=state_dict,
                    epoch=context.epoch,
                    consumed_samples=context.consumed_samples,
                    config=context.model_config,
                    extra=extra,
                    meta=meta,
                )
                context.checkpoint.save(save_path)
                _copy_tokenizer_files(context.param_path, save_path)
        self.last_ckpt_step = context.optimizer_step
        self._saved = True

    def after_optimizer_step(self, context: TrainContext):
        if context.optimizer_step - self.last_ckpt_step >= self.interval:
            self._save_checkpoint(context)

    def on_train_end(self, context: TrainContext):
        if context.optimizer_step != self.last_ckpt_step:
            self._save_checkpoint(context)

    def on_error(self, context: TrainContext):
        # An interrupted run must always leave at least one checkpoint
        # behind: on a slow start the signal can be handled before the
        # first optimizer step, where optimizer_step == last_ckpt_step
        # and the change-based guard alone would skip the save entirely.
        if not self._saved or context.optimizer_step != self.last_ckpt_step:
            self._save_checkpoint(context)

    @staticmethod
    def save_extra(context: TrainContext) -> dict:
        extra = {}
        for name in CheckpointCallback.extra_keys:
            obj = getattr(context, name, None)
            if obj:
                extra[name] = obj.state_dict()
        critic = getattr(context.strategy, "critic", None)
        if critic is not None:
            extra["value_model"] = critic.state_dict()
            critic_optimizer = getattr(context.strategy, "critic_optimizer", None)
            if critic_optimizer is not None:
                extra["value_optimizer"] = critic_optimizer.state_dict()
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
    def before_optimizer_step(self, context: TrainContext):
        postfix = {
            "step": f"{context.optimizer_step:d}",
            "loss": f"{context.loss:.4f}",
            "lr": f"{context.optimizer.param_groups[-1]['lr']:.2e}",
        }
        if context.grad_norm is not None:
            postfix["grad_norm"] = f"{context.grad_norm:.2f}"
        if context.val_loss is not None:
            postfix["val_loss"] = f"{context.val_loss:.4f}"
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
        ckpt_dir: str,
        save_interval: int,
        metrics: List[str] = None,
        val_step: int = 0,
    ):
        self.last_log_flush_step = None
        self.save_interval = save_interval
        self.metrics = metrics or ["loss", "lr"]
        self.val_step = val_step
        self._next_val_step = 0

        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else Path.cwd() / "checkpoint"

        self.log_cache = []

        self._metric_funcs = {
            "loss": ctx_get_loss,
            "lr": ctx_get_lr,
            "val_loss": ctx_get_val_loss,
            "grad_norm": ctx_get_grad_norm,
            "grad_snr": ctx_get_grad_snr,
            "moe_aux_loss": partial(ctx_get_moe_metric, key="aux_loss"),
            "router_entropy": partial(ctx_get_moe_metric, key="router_entropy"),
            "dead_expert_fraction": partial(
                ctx_get_moe_metric, key="dead_expert_fraction"
            ),
            "load_imbalance_mean": partial(
                ctx_get_moe_metric, key="load_imbalance_mean"
            ),
            "load_imbalance_max": partial(ctx_get_moe_metric, key="load_imbalance_max"),
        }

    def _metrics(self, context: TrainContext, names):
        metrics = dict(context.metrics)
        for name in names:
            metric_fn = self._metric_funcs.get(name)
            if metric_fn is None:
                continue
            value = metric_fn(context)
            if value is not None:
                metrics[name] = value
        selected = set(context.metrics) | set(names)
        selected.discard("*")
        result = {name: metrics[name] for name in selected if name in metrics}
        if result and context.dp_size > 1 and dist.is_initialized():
            metric_names = sorted(result)
            values = torch.tensor(
                [result[name] for name in metric_names],
                dtype=torch.float32,
                device=get_current_device(),
            )
            # dp-dimension average only: cp peers hold values derived from
            # the same batch, so summing them in would double-count.
            values = context.topology.reduce_mean(values)
            result.update(zip(metric_names, values.tolist()))
        return result

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
                # Online strategies evaluate a one-off rollout (leaving
                # the replay cache untouched) via the public hook; None
                # means offline — validate the batch directly.
                loss_output = context.strategy.validate_online(batch)
                if loss_output is None:
                    loss_output = context.strategy(batch)
                total_loss += loss_output["loss"].item()
                num_batches += 1

        # Sum (loss, batches) across dp replicas and take the ratio; the
        # reduce is a no-op single-process, where the clamp preserves the
        # local zero-batch behavior.
        stats = torch.tensor(
            [total_loss, float(num_batches)], device=get_current_device()
        )
        stats = context.topology.reduce_sum(stats)
        avg_loss = (stats[0] / stats[1].clamp(min=1.0)).item()

        context.model.train()
        return avg_loss

    def on_train_begin(self, context: TrainContext):
        self.last_log_flush_step = context.optimizer_step

    @only_on_rank(0)
    def _flush(self, epoch, step):
        log_file = self.ckpt_dir / f"epoch_{epoch}_step_{step}" / "metric.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            for log in self.log_cache:
                f.write(json.dumps(log) + "\n")

    def before_optimizer_step(self, context):
        context.grad_snr_tracker.update(context.model)

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

    def after_optimizer_step(self, context):
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
