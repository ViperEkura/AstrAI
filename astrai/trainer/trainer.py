import logging
from typing import List, Optional

import torch.distributed as dist

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

    def _commit_update(self, context: TrainContext, batch) -> None:
        self._call_callbacks("before_optimizer_step", context)
        coordinator = context.async_rollout
        if coordinator is None:
            context.strategy.optimizer_step(context.optimizer)
            context.optimizer.zero_grad()
            if context.scheduler:
                context.scheduler.step()
        else:
            context.checkpoint_safe = False
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
                    context.consumed_samples += batch.prompts.shape[0]
                    context.optimizer_steps_completed += 1
                    context.checkpoint_safe = True
        self._call_callbacks("after_optimizer_step", context)

    def _train_batch(self, context: TrainContext, batch) -> None:
        executor = context.executor
        with executor.accumulate(context.model):
            self._call_callbacks("on_batch_begin", context)
            microbatch_prompts = (
                context.config.async_train_microbatch_prompts
                if context.async_rollout is not None
                else None
            )
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
                        update_metrics[name] = update_metrics.get(name, 0.0) + float(
                            value
                        )
                if update_loss is None:
                    raise RuntimeError("training update has no valid loss")
                context.loss = float(update_loss.item())
                context.metrics = update_metrics
                if executor.sync_gradients:
                    self._commit_update(context, batch)
            if context.async_rollout is None:
                context.consumed_samples += (
                    context.config.batch_per_device * context.dp_size
                )
            self._call_callbacks("on_batch_end", context)

    def _train_epoch(self, context: TrainContext, epoch: int) -> None:
        context.epoch = epoch
        self._call_callbacks("on_epoch_begin", context)
        batches = context.dataloader
        if context.async_rollout is not None:
            batches = context.async_rollout.iter_rounds(
                batches, lambda: context.stop_requested
            )
        for batch in batches:
            if context.stop_requested:
                break
            self._train_batch(context, batch)
        self._call_callbacks("on_epoch_end", context)

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
                self._train_epoch(context, epoch)

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
