"""Metric logging and validation callback."""

from __future__ import annotations

import json
import logging
import shutil
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from astrai.parallel import only_on_rank
from astrai.parallel.setup import get_current_device
from astrai.trainer.callbacks.base import CallbackFactory, TrainCallback
from astrai.trainer.callbacks.gradient_noise import (
    GradientNoiseProbe,
    GradientNoiseTracker,
)
from astrai.trainer.callbacks.metric_util import (
    ctx_get_grad_norm,
    ctx_get_loss,
    ctx_get_lr,
    ctx_get_moe_metric,
    ctx_get_val_loss,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from astrai.trainer.context import TrainContext


@CallbackFactory.register("metric")
class MetricCallback(TrainCallback):
    def __init__(
        self,
        metrics: List[str] = None,
        val_step: int = 0,
        grad_noise_interval: int = 100,
        grad_noise_beta: float = 0.95,
    ):
        self.metrics = metrics or ["loss", "lr"]
        self.val_step = val_step
        if "grad_snr" in self.metrics:
            raise ValueError("grad_snr was retired; use grad_noise_scale")
        self.noise = (
            GradientNoiseTracker(grad_noise_interval, grad_noise_beta)
            if "grad_noise_scale" in self.metrics
            else None
        )
        self.probe = GradientNoiseProbe() if self.noise is not None else None
        self.last_committed_step = 0
        self._next_val_step = 0
        self.log_cache = []
        self.pending_records = []
        self.source_log_path = None

        self._metric_funcs = {
            "loss": ctx_get_loss,
            "lr": ctx_get_lr,
            "val_loss": ctx_get_val_loss,
            "grad_norm": ctx_get_grad_norm,
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
    def _append(self, event_type: str, context: TrainContext, pending=False, **extra):
        entry = {
            "type": event_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": context.epoch,
            "step": context.optimizer_step,
            "consumed_samples": context.consumed_samples,
            **extra,
        }
        (self.pending_records if pending else self.log_cache).append(entry)

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

    def _run_rollout_validation(self, context: TrainContext) -> Dict[str, float]:
        """Validation for online strategies via :class:`RolloutEvaluator`.

        Reports reward statistics under the evaluator's own sampling
        params instead of the training RL loss (degenerate under greedy
        decode or group_size == 1).  The generator toggles the model into
        eval mode itself, so no global mode switch is needed here; the
        replay cache and its cadence stay untouched by construction.
        """
        totals: Dict[str, float] = {}
        num_batches = 0
        for batch in context.val_dataloader:
            metrics = context.val_evaluator.evaluate(batch)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            num_batches += 1
        if num_batches == 0:
            return {}
        return {name: total / num_batches for name, total in totals.items()}

    def on_train_begin(self, context: TrainContext):
        if self.noise is not None:
            context.checkpoint_safe = False
            cfg = context.config
            if (
                not isinstance(context.model, DistributedDataParallel)
                or context.dp_size < 2
                or cfg.cp_size != 1
                or cfg.tp_size != 1
                or cfg.compile_mode is not None
                or cfg.rollout_mode != "sync"
                or context.strategy.policy_version is not None
            ):
                raise ValueError(
                    "grad_noise_scale requires uncompiled synchronous offline DDP"
                )
            context.checkpoint_safe = True
        self.last_committed_step = context.optimizer_step
        if context.checkpoint is not None and context.param_path is not None:
            state_path = Path(context.param_path) / "metric_state.json"
            if state_path.is_file():
                context.checkpoint_safe = False
                self.load_state_dict(
                    json.loads(state_path.read_text()), context.optimizer_step
                )
                context.checkpoint_safe = True
            elif self.noise is not None:
                logger.warning(
                    "Checkpoint has no gradient-noise state; starting metric warmup at step %s",
                    context.optimizer_step,
                )
        if self.probe is not None:
            context.on_shutdown("gradient_noise_probe", self.probe.close)
        if self.val_step > 0:
            next_slot = (
                (context.optimizer_step + self.val_step - 1) // self.val_step
            ) * self.val_step
            # Async rollout observes the pre-update step; offline training
            # advances the counter before calling before_optimizer_step.
            self._next_val_step = next_slot + (
                0 if context.async_rollout is not None else 1
            )
        if context.checkpoint is not None and context.param_path is not None:
            source = Path(context.param_path) / "metric.jsonl"
            if source.is_file():
                self.source_log_path = source

    @only_on_rank(0)
    def write_snapshot(self, destination: Path) -> None:
        """Add the committed metric history to a checkpoint's staging directory."""
        with destination.open("wb") as output:
            if self.source_log_path is not None:
                with self.source_log_path.open("rb") as source:
                    shutil.copyfileobj(source, output)
            for record in self.log_cache:
                output.write((json.dumps(record) + "\n").encode("utf-8"))

    def checkpoint_saved(self, path: Path, published: bool) -> None:
        if not published:
            if self.log_cache:
                raise RuntimeError(
                    f"Checkpoint {path} already exists but has unsaved metrics"
                )
            return
        self.source_log_path = path / "metric.jsonl"
        self.log_cache.clear()

    def state_dict(self) -> Dict:
        return {
            "version": 1,
            "optimizer_step": self.last_committed_step,
            "gradient_noise": self.noise.state_dict()
            if self.noise is not None
            else None,
        }

    def load_state_dict(self, state: Dict, step: int) -> None:
        if state.get("version") != 1 or state.get("optimizer_step") != step:
            raise ValueError("Metric state does not match checkpoint version or step")
        if self.noise is not None:
            if state.get("gradient_noise") is not None:
                self.noise.load_state_dict(state["gradient_noise"], step)
            else:
                logger.warning(
                    "Checkpoint has no gradient-noise history; starting metric warmup"
                )
        self.last_committed_step = step

    def write_state(self, destination: Path, step: int) -> None:
        if step != self.last_committed_step:
            raise ValueError(
                "Metric state does not match the committed checkpoint step"
            )
        destination.write_text(json.dumps(self.state_dict(), indent=2, allow_nan=False))

    def on_batch_begin(self, context):
        if self.noise is None or not context.executor.sync_gradients:
            return
        if (context.optimizer_step + 1) % self.noise.interval == 0:
            self.probe.start(context.model)

    def after_backward(self, context, local_count: float, global_count: float):
        if self.probe is not None:
            self.probe.finish(context, local_count, global_count, self.noise)

    def before_optimizer_step(self, context):
        self.pending_records.clear()

        if (
            context.val_dataloader is not None
            and self.val_step > 0
            and context.optimizer_step >= self._next_val_step
        ):
            if context.val_evaluator is not None:
                context.val_loss = None
                val_metrics = self._run_rollout_validation(context)
                self._next_val_step = context.optimizer_step + self.val_step
                self._append("validation", context, pending=True, **val_metrics)
            else:
                context.val_loss = self._run_validation(context)
                self._next_val_step = context.optimizer_step + self.val_step
                self._append(
                    "validation", context, pending=True, val_loss=context.val_loss
                )

        step_metrics = [m for m in self.metrics if m != "val_loss"]
        self._append(
            "step", context, pending=True, **self._metrics(context, step_metrics)
        )

    def after_optimizer_step(self, context):
        values = (
            self.noise.commit(context.optimizer_step) if self.noise is not None else {}
        )
        self.last_committed_step = context.optimizer_step
        for record in self.pending_records:
            record["step"] = context.optimizer_step
            record["consumed_samples"] = context.consumed_samples
            if record["type"] == "step":
                record.update(values)
        self.log_cache.extend(self.pending_records)
        self.pending_records.clear()

    def on_train_end(self, context):
        self.on_shutdown(context)

    def on_shutdown(self, context):
        self.pending_records.clear()
        if self.noise is not None:
            self.noise.pending = None
            self.probe.close()
