import logging
import random
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from astrai.config import TrainConfig
from astrai.dataset import GlobalBatchSampler
from astrai.parallel.executor.gradients import parameter_gradient
from astrai.parallel.setup import agree_stop, get_rank, spawn_parallel_fn
from astrai.trainer.callbacks import (
    CheckpointCallback,
    GradientCheckpointingCallback,
    GradientClippingCallback,
    MetricCallback,
    ProgressBarCallback,
    TrainCallback,
)
from astrai.trainer.components import stop_loader_workers
from astrai.trainer.context import TrainContext, TrainContextBuilder
from astrai.trainer.rollout.batching import slice_batch
from astrai.trainer.session import TrainSession
from astrai.trainer.step_plan import learner_update_slices

logger = logging.getLogger(__name__)


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed each rank's generators when starting or resuming a run."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self, train_config: TrainConfig, callbacks: Optional[List[TrainCallback]] = None
    ):
        self.train_config = train_config
        default_callbacks = self._get_default_callbacks()
        self.callbacks = (
            default_callbacks + callbacks if callbacks else default_callbacks
        )

    def _get_default_callbacks(self) -> List[TrainCallback]:
        cfg = self.train_config
        return [
            GradientCheckpointingCallback(modules=cfg.gradient_checkpointing_modules),
            MetricCallback(
                metrics=cfg.metrics,
                val_step=cfg.val_step,
                grad_snr_interval=cfg.grad_snr_interval,
            ),
            CheckpointCallback(cfg.ckpt_dir, cfg.ckpt_interval),
            ProgressBarCallback(cfg.n_epoch),
            GradientClippingCallback(cfg.max_grad_norm),
        ]

    def _call_callbacks(self, method_name: str, context: TrainContext):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(context)

    def _after_commit(self, context: TrainContext, *, agree_stop: bool) -> None:
        self._call_callbacks("after_optimizer_step", context)
        if agree_stop:
            self._agree_stop(context)

    @staticmethod
    def _step_policy(context: TrainContext) -> None:
        """Mutate actor weights and publish replicas under one version lock."""
        if context.rollout is None:
            context.optimizer.step()
            return

        def update(version: int):
            context.optimizer.step()
            context.rollout.publish_weights(version, context.model)

        context.rollout.apply_weight_update(None, update)
        context.rollout.step()

    def _commit_global_update(self, context: TrainContext, new_data: bool) -> None:
        """Commit a global update; failed runs retain the previous safe cursor.

        The step advances before the observers run, so the hooks that log or
        validate record the step they belong to; advancing it after would
        label every logged step with the previous one.
        """
        context.checkpoint_safe = False
        context.metric_history.discard_pending()
        committed_samples = context.pending_samples if new_data else 0
        context.optimizer_steps_completed += 1
        if new_data:
            context.consumed_samples += committed_samples
        try:
            self._call_callbacks("before_optimizer_step", context)
            self._step_policy(context)
            context.strategy.after_update()
            context.optimizer.zero_grad()
            if context.scheduler:
                context.scheduler.step()
            context.pending_samples = 0
        except BaseException:
            context.metric_history.discard_pending()
            context.optimizer_steps_completed -= 1
            if new_data:
                context.consumed_samples -= committed_samples
            raise
        context.metric_history.commit(context.optimizer_step, context.consumed_samples)

    def _commit_online_update(
        self, context: TrainContext, batch, new_data: bool
    ) -> None:
        """Account for published actor versions, even if later work fails."""
        context.checkpoint_safe = False
        context.metric_history.discard_pending()
        version_before = context.rollout.policy_version
        try:
            self._call_callbacks("before_optimizer_step", context)
            self._step_policy(context)
        finally:
            if context.rollout.policy_version > version_before:
                context.optimizer_steps_completed += 1
                if new_data:
                    count = batch.prompts.shape[0] * context.dp_size
                    context.consumed_samples += count
                    context.epoch_sample_offset += count
                try:
                    context.strategy.after_update()
                    context.optimizer.zero_grad()
                    if context.scheduler:
                        context.scheduler.step()
                    context.metric_history.commit(
                        context.optimizer_step, context.consumed_samples
                    )
                except BaseException:
                    context.metric_history.discard_pending()
                    raise
                context.checkpoint_safe = True
            else:
                context.metric_history.discard_pending()

    def _train_batch(self, context: TrainContext, batch) -> None:
        context.checkpoint_safe = False
        executor = context.executor
        with executor.accumulate_explicit(context.model, sync=True):
            self._call_callbacks("on_batch_begin", context)
            prepared = context.strategy.prepare_from_rollout(batch)
            update_loss = None
            update_metrics = {}
            for output in context.strategy.loss_outputs(
                prepared, context.config.async_train_microbatch_prompts
            ):
                loss = output["loss"]
                executor.backward(loss)
                detached = loss.detach()
                update_loss = (
                    detached if update_loss is None else update_loss + detached
                )
                for name, value in output["metrics"].items():
                    update_metrics[name] = update_metrics.get(name, 0.0) + float(value)
            if update_loss is None:
                raise RuntimeError("training update has no valid loss")
            context.loss = float(update_loss.item())
            context.metrics = update_metrics
            self._commit_online_update(context, batch, new_data=True)
            self._after_commit(context, agree_stop=False)
            self._call_callbacks("on_batch_end", context)

    def _should_stop(self, context: TrainContext) -> bool:
        """Online training stops between rounds; sync training at an agreed step."""
        if context.config.rollout_mode == "async_round":
            return context.stop_requested
        return context.stop_agreed

    def _agree_stop(self, context: TrainContext) -> bool:
        """Agree on a stop with every rank, so none leaves a peer mid-collective."""
        if context.config.rollout_mode == "async_round":
            return context.stop_requested
        device = next(context.model.parameters()).device
        if agree_stop(context.stop_requested, device):
            context.request_stop()
            context.stop_agreed = True
        return context.stop_agreed

    @staticmethod
    def _window_loss_output(strategy, batch, real_count: int):
        """Combine losses for one update without changing their normalization."""
        outputs = list(strategy.loss_outputs(batch))
        if not outputs:
            raise RuntimeError("global window strategy update has no loss")
        count = float(outputs[0].get("normalizer", real_count))
        if any(
            float(output.get("normalizer", real_count)) != count for output in outputs
        ):
            raise RuntimeError(
                "losses in one global window update must share a normalizer"
            )
        metrics = {}
        for output in outputs:
            for name, value in output["metrics"].items():
                metrics[name] = metrics.get(name, 0.0) + float(value)
        return {
            "loss": sum(output["loss"] for output in outputs),
            "metrics": metrics,
            "normalizer": count,
        }

    @staticmethod
    def _learner_microbatches(
        context: TrainContext, chunks: List[Tuple[Dict[str, Any], int]]
    ) -> List[List[Tuple[Dict[str, Any], int]]]:
        """Split each chunk into prompt-group microbatches of one update.

        Without ``rl_microbatch_prompts`` every chunk is its own single
        backward. With it, each microbatch backpropagates its local loss sum
        immediately, and the update's shared global denominator is applied
        once after the last one. Ranks pad to the same per-chunk schedule
        with zero-weight rows, so every collective forward/backward matches.
        """
        size = context.config.rl_microbatch_prompts
        if size is None:
            return [[chunk] for chunk in chunks]
        plan = []
        for batch, real_count in chunks:
            total = next(
                value.shape[0]
                for value in batch.values()
                if isinstance(value, torch.Tensor) and value.ndim
            )
            pieces = []
            # A placeholder chunk keeps its one row, still at zero weight.
            for begin in range(0, real_count or 1, size):
                end = min(begin + size, real_count or 1)
                pieces.append(
                    (
                        slice_batch(batch, slice(begin, end), total),
                        end - begin if real_count else 0,
                    )
                )
            plan.append((batch, total, pieces))
        device = next(context.model.parameters()).device
        lengths = torch.tensor(
            [len(pieces) for _, _, pieces in plan], dtype=torch.int64, device=device
        )
        if context.dp_size > 1 and dist.is_initialized():
            dist.all_reduce(
                lengths, op=dist.ReduceOp.MAX, group=context.topology.dp_group
            )
        for (batch, total, pieces), length in zip(plan, lengths.tolist()):
            pieces.extend(
                (slice_batch(batch, slice(0, 1), total), 0)
                for _ in range(length - len(pieces))
            )
        return [pieces for _, _, pieces in plan]

    def _run_global_update(
        self,
        context: TrainContext,
        chunks: List[Tuple[Dict[str, Any], int]],
        window_samples: int,
        window_end: int,
        final_update: bool,
    ) -> None:
        """Reduce true objective numerators before committing one update."""
        local_count = 0.0
        local_loss = 0.0
        local_metrics: Dict[str, float] = {}
        plan = self._learner_microbatches(context, chunks)
        for index, pieces in enumerate(plan):
            for piece, (batch, real_count) in enumerate(pieces):
                with context.executor.accumulate_explicit(
                    context.model,
                    sync=index + 1 == len(plan) and piece + 1 == len(pieces),
                ):
                    if piece == 0:
                        self._call_callbacks("on_batch_begin", context)
                    output = self._window_loss_output(
                        context.strategy, batch, real_count
                    )
                    count = (
                        float(output.get("normalizer", real_count))
                        if real_count
                        else 0.0
                    )
                    context.executor.backward(output["loss"] * count)
                    local_count += count
                    reported_loss = float(output["loss"].detach().item())
                    if context.config.cp_size > 1:
                        reported_loss = output["metrics"]["task_loss"]
                    local_loss += reported_loss * count
                    for name, value in output["metrics"].items():
                        local_metrics[name] = (
                            local_metrics.get(name, 0.0) + value * count
                        )
                    if piece + 1 == len(pieces):
                        self._call_callbacks("on_batch_end", context)

        device = next(context.model.parameters()).device
        totals = torch.tensor(
            [local_count, local_loss, *local_metrics.values()],
            dtype=torch.float64,
            device=device,
        )
        context.topology.reduce_sum(totals)
        denominator = float(totals[0].item())
        if denominator <= 0:
            raise RuntimeError("global update contains no valid training targets")
        # DDP averages gradients over replicas (and CP peers, whose token
        # loss already carries the CP factor). Restore the global objective.
        context.executor.scale_gradients(context.model, context.dp_size / denominator)
        critic = getattr(context.strategy, "critic", None)
        if critic is not None:
            for parameter in critic.parameters():
                gradient = parameter_gradient(parameter)
                if gradient is not None:
                    context.topology.reduce_sum(gradient)
                    gradient.mul_(1.0 / denominator)
        context.loss = float(totals[1].item()) / denominator
        context.metrics = {
            name: float(totals[index + 2].item()) / denominator
            for index, name in enumerate(local_metrics)
        }
        context.metrics["loss"] = context.loss
        if final_update:
            context.pending_samples = window_samples
        self._commit_global_update(context, new_data=final_update)
        if final_update:
            context.epoch_sample_offset = window_end
            context.checkpoint_safe = True
        self._after_commit(context, agree_stop=final_update)

    def _train_global_epoch(self, context: TrainContext, epoch: int) -> None:
        if context.epoch < epoch and context.epoch_sample_offset == context.dataset_len:
            context.epoch_sample_offset = 0
        context.epoch = epoch
        self._call_callbacks("on_epoch_begin", context)
        sampler = context.dataloader.batch_sampler
        if not isinstance(sampler, GlobalBatchSampler):
            raise RuntimeError("global cursor requires GlobalBatchSampler")
        sampler.set_epoch(epoch, context.epoch_sample_offset)
        if not context.stop_agreed:
            self._agree_stop(context)
        iterator = iter(context.dataloader)
        context.train_iterator = iterator
        context.on_shutdown("train_loader", partial(stop_loader_workers, iterator))
        for window_start, window_samples, microbatches in sampler.windows():
            if context.stop_agreed:
                break
            batches = [
                (next(iterator), sampler.real_count(window_start, slot))
                for slot in range(microbatches)
            ]
            context.checkpoint_safe = False
            online = context.rollout is not None
            if online:
                prepared = []
                for batch, _real_count in batches:
                    result, _ = context.rollout(batch)
                    prepared.append(
                        {
                            key: value.detach().cpu()
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in context.strategy.prepare_from_rollout(
                                result
                            ).items()
                        }
                    )
            else:
                prepared = []

            for selection, final_update in learner_update_slices(
                window_samples,
                context.config.rl_update_epochs if online else 1,
                context.config.rl_minibatch_prompts if online else None,
            ):
                chunks = []
                for slot, (batch, real_count) in enumerate(batches):
                    if online:
                        local_first = (
                            slot * context.dp_size * context.config.batch_per_device
                            + context.topology.dp_rank * context.config.batch_per_device
                        )
                        indices = [
                            index
                            for index in range(real_count)
                            if selection.start <= local_first + index < selection.stop
                        ]
                        source = prepared[slot]
                        total = next(
                            value.shape[0]
                            for value in source.values()
                            if isinstance(value, torch.Tensor) and value.ndim
                        )
                        chunk = slice_batch(source, indices or [0], total)
                        chunks.append((chunk, len(indices)))
                    else:
                        chunks.append((batch, real_count))
                self._run_global_update(
                    context,
                    chunks,
                    window_samples,
                    window_start + window_samples,
                    final_update=final_update,
                )

        if not context.stop_agreed:
            stop_loader_workers(iterator)
            dropped = context.dataset_len - context.epoch_sample_offset
            if context.config.epoch_tail_policy == "drop" and dropped > 0:
                context.consumed_samples += dropped
                context.epoch_sample_offset = 0
        else:
            dropped = 0
        self._call_callbacks("on_epoch_end", context)
        if dropped:
            context.epoch = epoch + 1

    def _train_epoch(self, context: TrainContext, epoch: int) -> None:
        if context.config.rollout_mode != "async_round":
            self._train_global_epoch(context, epoch)
            return
        context.epoch = epoch
        context.dataloader.sampler.set_epoch(epoch)
        context.dataloader.sampler.iter = context.epoch_sample_offset // context.dp_size
        self._call_callbacks("on_epoch_begin", context)
        loader_iterator = iter(context.dataloader)
        batches = context.rollout.iter_rounds(
            loader_iterator, lambda: context.stop_requested
        )
        iterator = iter(batches)
        context.train_iterator = iterator
        context.on_shutdown(
            "train_loader", partial(stop_loader_workers, loader_iterator)
        )
        exhausted = False
        while not self._should_stop(context):
            try:
                batch = next(iterator)
            except StopIteration:
                exhausted = True
                break
            self._train_batch(context, batch)
        self._call_callbacks("on_epoch_end", context)
        if exhausted:
            context.epoch_sample_offset = 0
            context.epoch = epoch + 1

    def _trainer_loop(self, param_path: Optional[str] = None, resume: bool = False):
        # Seed before the context is built so initialisation and data drawing start
        # from a defined state. The sampler has its own epoch-seeded generator.
        seed_everything(self.train_config.random_seed, get_rank())
        with TrainSession(
            lambda: (
                TrainContextBuilder(self.train_config)
                .with_param_path(param_path, resume=resume)
                .build()
            ),
            self._call_callbacks,
        ) as context:
            context.model.train()
            for epoch in range(context.epoch, context.config.n_epoch):
                if context.stop_agreed or self._agree_stop(context):
                    break
                self._train_epoch(context, epoch)

            if context.stop_requested:
                logger.warning(
                    "Training interrupted by signal, saving emergency checkpoint..."
                )

    def train(self, param_path: Optional[str] = None, resume: bool = False):
        cfg = self.train_config
        spawn_parallel_fn(
            self._trainer_loop,
            backend=cfg.backend,
            world_size=cfg.nprocs,
            master_addr=cfg.master_addr,
            master_port=cfg.master_port,
            device_type=cfg.device_type,
            start_method=cfg.start_method,
            process_group_timeout=cfg.process_group_timeout,
            param_path=param_path,
            resume=resume,
        )
