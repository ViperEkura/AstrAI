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

                for batch in context.dataloader:
                    if context.stop_requested:
                        break
                    with executor.accumulate(context.model):
                        self._call_callbacks("on_batch_begin", context)
                        # Each update iterator owns all of its microbatches.
                        # The strategy freezes the denominator before yielding
                        # losses; policy publication follows the final backward.
                        if context.optimizer_steps is not None:
                            context.kwargs["grpo_update_in_progress"] = True
                        last_output = None
                        last_metrics = {"loss": 0.0, "empty_update": 1.0}
                        for update in context.strategy.training_updates(batch):
                            update_output = None
                            additive = {}
                            for loss_output in update:
                                update_output = last_output = loss_output
                                stand_loss = (
                                    loss_output["loss"] / executor.grad_accum_steps
                                )
                                executor.backward(stand_loss)
                                for name in (
                                    "loss",
                                    "policy_loss",
                                    "kl_loss",
                                    "padding_microbatches",
                                ):
                                    if name in loss_output["metrics"]:
                                        additive[name] = (
                                            additive.get(name, 0.0)
                                            + loss_output["metrics"][name]
                                        )
                            if update_output is None:
                                continue
                            last_metrics = {**update_output["metrics"], **additive}
                            if executor.sync_gradients:
                                self._call_callbacks("before_optimizer_step", context)
                                context.strategy.optimizer_step(context.optimizer)
                                context.optimizer.zero_grad()

                                if context.scheduler:
                                    context.scheduler.step()
                                if context.optimizer_steps is not None:
                                    context.optimizer_steps += 1
                                self._call_callbacks("after_optimizer_step", context)
                        context.loss = (
                            last_metrics["loss"]
                            if context.optimizer_steps is not None
                            or last_output is None
                            else last_output["loss"].item()
                        )
                        context.metrics = last_metrics
                        if context.optimizer_steps is not None:
                            context.consumed_samples += (
                                context.strategy.training_global_prompts
                            )
                            context.kwargs["grpo_update_in_progress"] = False
                        else:
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
