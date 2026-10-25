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

    def _commit_update(self, context: TrainContext, batch, new_data: bool = True):
        if context.async_rollout is not None:
            self._commit_online_update(context, batch, new_data)
        else:
            self._commit_offline_update(context, new_data)
        self._after_commit(context, agree_stop=context.async_rollout is None)

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
                    context.consumed_samples += batch.prompts.shape[0] * context.dp_size

    def _train_batch(
        self,
        context: TrainContext,
        batch,
        force_sync: bool = False,
        batch_samples: Optional[int] = None,
    ) -> None:
        executor = context.executor
        complete_window = (
            getattr(context.strategy, "complete_update_windows", False)
            and context.async_rollout is None
        )
        if complete_window:
            context.checkpoint_safe = False
            context.loss = 0.0
            context.metrics = {"loss": 0.0, "empty_update": 1.0}
        accumulation = (
            executor.accumulate(context.model, force_sync=True)
            if force_sync
            else executor.accumulate(context.model)
        )
        with accumulation:
            if context.async_rollout is None:
                if batch_samples is None:
                    batch_samples = context.config.batch_per_device * context.dp_size
                context.pending_samples += batch_samples
            self._call_callbacks("on_batch_begin", context)
            microbatch_prompts = (
                context.config.async_train_microbatch_prompts
                if context.async_rollout is not None
                else None
            )
            # A batch can produce several updates (replayed rollout rounds) or
            # none (mid-window). Steps follow every commit; the data cursor
            # moves once per batch, because a replayed batch is one position.
            committed = False
            for update in context.strategy.training_updates(batch, microbatch_prompts):
                update_loss = None
                update_metrics = {}
                for output in update:
                    loss = output["loss"]
                    executor.backward(loss / executor.grad_accum_steps)
                    detached = loss.detach()
                    update_loss = (
                        detached if update_loss is None else update_loss + detached
                    )
                    for name, value in output["metrics"].items():
                        if complete_window and name not in {
                            "loss",
                            "policy_loss",
                            "kl_loss",
                            "padding_microbatches",
                        }:
                            update_metrics[name] = float(value)
                        else:
                            update_metrics[name] = update_metrics.get(
                                name, 0.0
                            ) + float(value)
                if update_loss is None:
                    if complete_window:
                        continue
                    raise RuntimeError("training update has no valid loss")
                context.loss = (
                    update_metrics["loss"]
                    if complete_window
                    else float(update_loss.item())
                )
                context.metrics = update_metrics
                if executor.sync_gradients:
                    if (
                        context.async_rollout is None
                        and executor.window_steps < executor.grad_accum_steps
                    ):
                        executor.scale_gradients(
                            context.model,
                            executor.grad_accum_steps / executor.window_steps,
                        )
                    self._commit_update(
                        context, batch, new_data=not committed and not complete_window
                    )
                    committed = True
            if complete_window:
                context.consumed_samples += context.strategy.training_global_prompts
                context.pending_samples = 0
                context.checkpoint_safe = True
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
        outputs = context.strategy.global_training_outputs(chunks)
        if outputs is not None:
            self._run_normalized_update(
                context, outputs, window_samples, window_end, final_update
            )
            return
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

    def _run_normalized_update(
        self, context, outputs, window_samples, window_end, final_update
    ):
        """Commit losses whose strategy already owns the global denominator."""
        metrics = {}
        seen = False
        with context.executor.accumulate_explicit(context.model, sync=True):
            self._call_callbacks("on_batch_begin", context)
            for output in outputs:
                seen = True
                context.executor.backward(output["loss"])
                for name, value in output["metrics"].items():
                    if name in {
                        "loss",
                        "policy_loss",
                        "kl_loss",
                        "padding_microbatches",
                    }:
                        metrics[name] = metrics.get(name, 0.0) + float(value)
                    else:
                        metrics[name] = float(value)
            context.metrics = metrics if seen else {"loss": 0.0, "empty_update": 1.0}
            context.loss = context.metrics["loss"]
            if seen:
                if final_update:
                    context.pending_samples = window_samples
                self._commit_offline_update(context, new_data=final_update)
            elif final_update:
                # A zero-target window consumes data, never an optimizer step.
                context.consumed_samples += window_samples
                context.pending_samples = 0
            if final_update:
                context.epoch_sample_offset = window_end
                context.checkpoint_safe = True
            if seen:
                self._after_commit(context, agree_stop=final_update)
            elif final_update:
                self._agree_stop(context)
            self._call_callbacks("on_batch_end", context)

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
                    result, is_fresh = context.strategy._rollout_runner(batch)
                    if is_fresh:
                        context.strategy._on_rollout_refresh()
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
                group_size = context.config.rl_minibatch_prompts or window_samples
                groups = (window_samples + group_size - 1) // group_size
                total_updates = context.config.rl_update_epochs * groups
            else:
                prepared = []
                group_size = window_samples
                groups = 1
                total_updates = 1

            grouped = online or getattr(
                context.strategy, "complete_update_windows", False
            )
            if grouped and not online:
                prepared = [batch for batch, _ in batches]
                group_size = context.config.rl_minibatch_prompts or window_samples
                groups = (window_samples + group_size - 1) // group_size
                total_updates = groups
            update_index = 0
            for _pass in range(context.config.rl_update_epochs if online else 1):
                for group in range(groups):
                    chunks = []
                    for slot, (batch, real_count) in enumerate(batches):
                        if grouped:
                            local_first = (
                                slot * context.dp_size * context.config.batch_per_device
                                + context.topology.dp_rank
                                * context.config.batch_per_device
                            )
                            indices = [
                                index
                                for index in range(real_count)
                                if group * group_size
                                <= local_first + index
                                < min((group + 1) * group_size, window_samples)
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
                    update_index += 1
                    self._run_global_update(
                        context,
                        chunks,
                        window_samples,
                        window_start + window_samples,
                        final_update=update_index == total_updates,
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
        if context.global_cursor and context.async_rollout is None:
            self._train_global_epoch(context, epoch)
            return
        context.epoch = epoch
        self._call_callbacks("on_epoch_begin", context)
        batches = context.dataloader
        if context.async_rollout is not None:
            batches = context.async_rollout.iter_rounds(
                batches, lambda: context.stop_requested
            )
        # A stop waits for the next commit. The epoch tail follows the
        # configured policy only when the epoch ends naturally.
        if not context.stop_agreed:
            self._agree_stop(context)
        if context.async_rollout is None:
            # Capture this before creating the iterator: worker prefetch can
            # advance the sampler well beyond the batches the loop consumed.
            remaining_samples = len(context.dataloader.sampler)
            total_batches = len(context.dataloader)
            if context.config.epoch_tail_policy == "drop":
                full_batches = remaining_samples // context.config.batch_per_device
                trainable_batches = (
                    full_batches // context.executor.grad_accum_steps
                ) * context.executor.grad_accum_steps
            else:
                trainable_batches = total_batches
        iterator = iter(batches)
        context.train_iterator = iterator
        # The session runs this before any checkpoint is written: a worker
        # killed by the stop signal would otherwise raise mid-save.
        context.on_shutdown("train_loader", partial(stop_loader_workers, iterator))
        batch_index = 0
        while not self._should_stop(context) and (
            context.async_rollout is not None or batch_index < trainable_batches
        ):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if context.async_rollout is None:
                batch_samples = (
                    min(
                        context.config.batch_per_device,
                        remaining_samples
                        - batch_index * context.config.batch_per_device,
                    )
                    * context.dp_size
                )
                self._train_batch(
                    context,
                    batch,
                    force_sync=(
                        context.config.epoch_tail_policy == "flush"
                        and batch_index + 1 == trainable_batches
                    ),
                    batch_samples=batch_samples,
                )
            else:
                self._train_batch(context, batch)
            batch_index += 1
        dropped_samples = 0
        if (
            context.async_rollout is None
            and batch_index == trainable_batches
            and (batch_index > 0 or not context.stop_agreed)
        ):
            # Close prefetched workers before advancing the sampler. Under
            # "drop", no remaining tail batch is passed through the model.
            stop_loader_workers(iterator)
            context.dataloader.sampler.set_epoch(epoch + 1)
            dropped_samples = (
                max(
                    0,
                    remaining_samples
                    - trainable_batches * context.config.batch_per_device,
                )
                * context.dp_size
            )
            context.consumed_samples += dropped_samples
        self._call_callbacks("on_epoch_end", context)
        if dropped_samples:
            # Keep the same optimizer step, but publish a distinct checkpoint
            # for the advanced data cursor if shutdown happens at this boundary.
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
