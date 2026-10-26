import logging
import random
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from astrai.config import TrainConfig
from astrai.dataset import GlobalBatchSampler
from astrai.parallel.setup import agree_stop, get_rank, spawn_parallel_fn
from astrai.trainer.callbacks import (
    CallbackFactory,
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
        metric_callback = CallbackFactory.create_checked(
            "metric",
            metrics=cfg.metrics,
            val_step=cfg.val_step,
            grad_snr_interval=cfg.grad_snr_interval,
        )
        callbacks = [
            CallbackFactory.create_checked(
                "gradient_checkpointing",
                modules=cfg.gradient_checkpointing_modules,
            ),
            metric_callback,
            CallbackFactory.create_checked(
                "checkpoint",
                cfg.ckpt_dir,
                cfg.ckpt_interval,
                metric_callback=metric_callback,
            ),
            CallbackFactory.create_checked("progress_bar", cfg.n_epoch),
            CallbackFactory.create_checked("gradient_clipping", cfg.max_grad_norm),
        ]
        return callbacks

    def _call_callbacks(self, method_name: str, context: TrainContext):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(context)

    def _after_commit(self, context: TrainContext, *, agree_stop: bool) -> None:
        self._call_callbacks("after_optimizer_step", context)
        if agree_stop:
            self._agree_stop(context)

    def _commit_offline_update(self, context: TrainContext, new_data: bool) -> None:
        """Apply one step; both counters roll back if the step itself fails.

        The step advances before the observers run, so the hooks that log or
        validate record the step they belong to; advancing it after would
        label every logged step with the previous one.
        """
        committed_samples = context.pending_samples if new_data else 0
        context.optimizer_steps_completed += 1
        if new_data:
            context.consumed_samples += committed_samples
        try:
            self._call_callbacks("before_optimizer_step", context)
            context.strategy.optimizer_step(context.optimizer)
            context.optimizer.zero_grad()
            if context.scheduler:
                context.scheduler.step()
            context.pending_samples = 0
        except BaseException:
            context.optimizer_steps_completed -= 1
            if new_data:
                context.consumed_samples -= committed_samples
            raise

    def _commit_online_update(
        self, context: TrainContext, batch, new_data: bool
    ) -> None:
        """Publish one policy version; the cursor follows the version, not the call."""
        self._call_callbacks("before_optimizer_step", context)
        coordinator = context.async_rollout
        version_before = coordinator.policy_version
        try:
            context.strategy.optimizer_step(context.optimizer)
        finally:
            # A version advance marks a committed step, even if later
            # publication fails and an error checkpoint is needed.
            if coordinator.policy_version > version_before:
                context.optimizer.zero_grad()
                if context.scheduler:
                    context.scheduler.step()
                context.optimizer_steps_completed += 1
                if new_data:
                    count = batch.prompts.shape[0] * context.dp_size
                    context.consumed_samples += count
                    context.epoch_sample_offset += count
                context.checkpoint_safe = True

    def _train_batch(self, context: TrainContext, batch) -> None:
        context.checkpoint_safe = False
        executor = context.executor
        with executor.accumulate_explicit(context.model, sync=True):
            self._call_callbacks("on_batch_begin", context)
            for update in context.strategy.training_updates(
                batch, context.config.async_train_microbatch_prompts
            ):
                update_loss = None
                update_metrics = {}
                for output in update:
                    loss = output["loss"]
                    executor.backward(loss)
                    detached = loss.detach()
                    update_loss = (
                        detached if update_loss is None else update_loss + detached
                    )
                    for name, value in output["metrics"].items():
                        update_metrics[name] = update_metrics.get(name, 0.0) + float(
                            value
                        )
                if update_loss is None:
                    raise RuntimeError("training update has no valid loss")
                context.loss = float(update_loss.item())
                context.metrics = update_metrics
                self._commit_online_update(context, batch, new_data=True)
                self._after_commit(context, agree_stop=False)
            self._call_callbacks("on_batch_end", context)

    def _should_stop(self, context: TrainContext) -> bool:
        """Online training stops between rounds; sync training at an agreed step."""
        if context.async_rollout is not None:
            return context.stop_requested
        return context.stop_agreed

    def _agree_stop(self, context: TrainContext) -> bool:
        """Agree on a stop with every rank, so none leaves a peer mid-collective."""
        if context.async_rollout is not None:
            return context.stop_requested
        device = next(context.model.parameters()).device
        if agree_stop(context.stop_requested, device):
            context.request_stop()
            context.stop_agreed = True
        return context.stop_agreed

    @staticmethod
    def _window_loss_output(strategy, batch, real_count: int):
        """Use the offline strategy hook without letting it add hidden commits."""
        if strategy.policy_version is not None:
            # Online batches have already been collected and prepared for the
            # current learner update by _train_global_epoch.
            return strategy.compute_loss_output(batch)
        updates = iter(strategy.training_updates(batch))
        try:
            outputs = list(next(updates))
        except StopIteration as error:
            raise RuntimeError("global window strategy produced no update") from error
        if next(updates, None) is not None:
            raise RuntimeError(
                "global window strategy must yield one update per microbatch"
            )
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
        for index, (batch, real_count) in enumerate(chunks):
            with context.executor.accumulate_explicit(
                context.model, sync=index + 1 == len(chunks)
            ):
                self._call_callbacks("on_batch_begin", context)
                output = self._window_loss_output(context.strategy, batch, real_count)
                count = (
                    float(output.get("normalizer", real_count)) if real_count else 0.0
                )
                context.executor.backward(output["loss"] * count)
                local_count += count
                reported_loss = float(output["loss"].detach().item())
                if context.config.cp_size > 1:
                    reported_loss = output["metrics"]["task_loss"]
                local_loss += reported_loss * count
                for name, value in output["metrics"].items():
                    local_metrics[name] = local_metrics.get(name, 0.0) + value * count
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
                if parameter.grad is not None:
                    context.topology.reduce_sum(parameter.grad)
                    parameter.grad.mul_(1.0 / denominator)
        context.loss = float(totals[1].item()) / denominator
        context.metrics = {
            name: float(totals[index + 2].item()) / denominator
            for index, name in enumerate(local_metrics)
        }
        context.metrics["loss"] = context.loss
        if final_update:
            context.pending_samples = window_samples
        self._commit_offline_update(context, new_data=final_update)
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
            online = context.strategy.policy_version is not None
            if online:
                prepared = []
                for batch, _real_count in batches:
                    prepared.append(
                        {
                            key: value.detach().cpu()
                            if isinstance(value, torch.Tensor)
                            else value
                            for key, value in context.strategy.prepare_rollout_batch(
                                batch
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
                            if isinstance(value, torch.Tensor)
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
        if context.async_rollout is None:
            self._train_global_epoch(context, epoch)
            return
        context.epoch = epoch
        context.dataloader.sampler.set_epoch(epoch)
        context.dataloader.sampler.iter = context.epoch_sample_offset // context.dp_size
        self._call_callbacks("on_epoch_begin", context)
        loader_iterator = iter(context.dataloader)
        batches = context.async_rollout.iter_rounds(
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
