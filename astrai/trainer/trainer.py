import logging
from typing import List, Optional

import torch.distributed as dist
from torch import Tensor

from astrai.config import TrainConfig
from astrai.parallel.setup import spawn_parallel_fn
from astrai.signal_handler import (
    register_signal_handlers,
    unregister_signal_handlers,
)
from astrai.trainer.callbacks import (
    CallbackFactory,
    TrainCallback,
)
from astrai.trainer.rollout.batching import slice_batch
from astrai.trainer.rollout.types import RolloutVersionError
from astrai.trainer.train_context import TrainContext, TrainContextBuilder

logger = logging.getLogger(__name__)


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
        callbacks = [
            CallbackFactory.create(
                "gradient_checkpointing",
                modules=cfg.gradient_checkpointing_modules,
            ),
            CallbackFactory.create(
                "checkpoint",
                cfg.ckpt_dir,
                cfg.ckpt_interval,
            ),
            CallbackFactory.create(
                "metric",
                ckpt_dir=cfg.ckpt_dir,
                save_interval=cfg.ckpt_interval,
                metrics=cfg.metrics,
                val_step=cfg.val_step,
                grad_snr_interval=cfg.grad_snr_interval,
            ),
            CallbackFactory.create("progress_bar", cfg.n_epoch),
            CallbackFactory.create("gradient_clipping", cfg.max_grad_norm),
        ]
        return callbacks

    def _call_callbacks(self, method_name: str, context: TrainContext):
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(context)

    def _train_async_epoch(self, context: TrainContext) -> None:
        """Keep one rollout round ahead of a single GRPO optimizer update."""
        coordinator = context.async_rollout
        batches = iter(context.dataloader)
        batch = next(batches, None)
        if batch is None:
            return
        handle = coordinator.submit_round(batch)
        while batch is not None and not context.stop_requested:
            try:
                rollout = coordinator.collect_round(handle)
            except RolloutVersionError:
                # Discard the entire old round. No optimizer step has run.
                handle = coordinator.submit_round(batch)
                rollout = coordinator.collect_round(handle)

            next_batch = next(batches, None)
            next_handle = (
                coordinator.submit_round(next_batch) if next_batch is not None else None
            )

            self._call_callbacks("on_batch_begin", context)
            context.strategy._on_rollout_refresh()
            prepared = context.strategy.prepare_from_rollout(rollout)
            total = prepared["prompts"].shape[0]
            masks = prepared["masks"]
            if context.strategy.loss_aggregation == "sequence":
                denominator = int(masks.any(dim=-1).sum().item())
            else:
                denominator = int(masks.sum().item())
            if denominator == 0:
                raise RuntimeError("async GRPO round has no valid response tokens")
            weighted_loss = None
            weighted_metrics = {}
            size = context.config.async_train_microbatch_prompts
            for begin in range(0, total, size):
                indices = list(range(begin, min(begin + size, total)))
                chunk = slice_batch(prepared, indices, total)
                chunk_masks = chunk["masks"]
                if context.strategy.loss_aggregation == "sequence":
                    count = int(chunk_masks.any(dim=-1).sum().item())
                else:
                    count = int(chunk_masks.sum().item())
                if count == 0:
                    continue
                weight = count / denominator
                output = context.strategy.compute_loss_output(chunk)
                loss = output["loss"] * weight
                context.executor.backward(loss)
                detached = loss.detach()
                weighted_loss = (
                    detached if weighted_loss is None else weighted_loss + detached
                )
                for name, value in output["metrics"].items():
                    metric = value.detach() if isinstance(value, Tensor) else value
                    weighted_metrics[name] = weighted_metrics.get(name, 0.0) + (
                        float(metric) * weight
                    )

            context.loss = float(weighted_loss.item())
            context.metrics = weighted_metrics
            self._call_callbacks("before_optimizer_step", context)
            context.checkpoint_safe = False
            version_before = coordinator.policy_version
            try:
                context.strategy.optimizer_step(context.optimizer)
            finally:
                # A version advance is the commit marker. Complete accounting
                # even if the subsequent NCCL channel state update fails.
                if coordinator.policy_version > version_before:
                    context.optimizer.zero_grad()
                    if context.scheduler:
                        context.scheduler.step()
                    context.consumed_samples += total
                    context.optimizer_steps_completed += 1
                    context.checkpoint_safe = True
            self._call_callbacks("after_optimizer_step", context)
            self._call_callbacks("on_batch_end", context)
            batch, handle = next_batch, next_handle

    def _trainer_loop(self, param_path: Optional[str] = None, resume: bool = False):
        context = (
            TrainContextBuilder(self.train_config)
            .with_param_path(param_path, resume=resume)
            .build()
        )
        register_signal_handlers(context)
        executor = context.executor
        self._call_callbacks("on_train_begin", context)

        try:
            context.model.train()

            for epoch in range(context.epoch, context.config.n_epoch):
                if context.stop_requested:
                    break
                context.epoch = epoch
                self._call_callbacks("on_epoch_begin", context)

                if context.async_rollout is not None:
                    self._train_async_epoch(context)
                    self._call_callbacks("on_epoch_end", context)
                    continue

                for batch in context.dataloader:
                    if context.stop_requested:
                        break
                    with executor.accumulate(context.model):
                        self._call_callbacks("on_batch_begin", context)
                        # One batch may expand into several learner updates
                        # (online RL: one rollout round -> minibatches x
                        # update epochs); each yielded step gets its own
                        # backward and, when the accumulation window syncs,
                        # its own optimizer step.
                        last_output = None
                        for loss_output in context.strategy.training_steps(batch):
                            last_output = loss_output
                            stand_loss = loss_output["loss"] / executor.grad_accum_steps
                            executor.backward(stand_loss)

                            if executor.sync_gradients:
                                self._call_callbacks("before_optimizer_step", context)
                                context.strategy.optimizer_step(context.optimizer)
                                context.optimizer.zero_grad()

                                if context.scheduler:
                                    context.scheduler.step()

                                self._call_callbacks("after_optimizer_step", context)
                        context.loss = last_output["loss"].item()
                        context.metrics = last_output["metrics"]
                        context.consumed_samples += (
                            context.config.batch_per_device * context.dp_size
                        )
                        self._call_callbacks("on_batch_end", context)

                self._call_callbacks("on_epoch_end", context)

            if context.stop_requested:
                logger.warning(
                    "Training interrupted by signal, saving emergency checkpoint..."
                )
                self._call_callbacks("on_error", context)

        except Exception as e:
            logger.error("Training failed: %s", str(e), exc_info=True)
            self._call_callbacks("on_error", context)
            raise
        finally:
            if context.async_rollout is not None:
                context.async_rollout.close()
            self._call_callbacks("on_train_end", context)
            if executor.use_distributed and dist.is_initialized():
                dist.barrier()
            unregister_signal_handlers()

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
            param_path=param_path,
            resume=resume,
        )
