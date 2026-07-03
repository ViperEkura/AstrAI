import logging
from typing import List, Optional

from astrai.config import TrainConfig
from astrai.parallel.setup import spawn_parallel_fn
from astrai.trainer.train_callback import (
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
                log_dir=cfg.log_dir,
                save_interval=cfg.ckpt_interval,
                metrics=cfg.metrics,
                val_step=cfg.val_step,
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

    def _trainer_loop(self, resume_dir: Optional[str] = None):
        context = (
            TrainContextBuilder(self.train_config).with_resume_dir(resume_dir).build()
        )
        executor = context.executor
        self._call_callbacks("on_train_begin", context)

        try:
            context.model.train()

            for epoch in range(context.epoch, context.config.n_epoch):
                context.epoch = epoch
                self._call_callbacks("on_epoch_begin", context)

                for batch in context.dataloader:
                    with executor.accumulate(context.model):
                        self._call_callbacks("on_batch_begin", context)
                        loss = context.strategy(batch)
                        context.loss = loss.item()
                        stand_loss = loss / executor.grad_accum_steps
                        executor.backward(stand_loss)
                        context.consumed_samples += (
                            context.config.batch_per_device * context.world_size
                        )
                        self._call_callbacks("on_batch_end", context)

                        if executor.sync_gradients:
                            self._call_callbacks("on_optimizer_step", context)
                            context.optimizer.step()
                            context.optimizer.zero_grad()

                            if context.scheduler:
                                context.scheduler.step()

                self._call_callbacks("on_epoch_end", context)

        except Exception as e:
            logger.error("Training failed: %s", str(e), exc_info=True)
            self._call_callbacks("on_error", context)
            raise
        finally:
            self._call_callbacks("on_train_end", context)

    def train(self, resume_dir: Optional[str] = None):
        cfg = self.train_config
        spawn_parallel_fn(
            self._trainer_loop,
            backend=cfg.backend,
            world_size=cfg.nprocs,
            master_addr=cfg.master_addr,
            master_port=cfg.master_port,
            device_type=cfg.device_type,
            start_method=cfg.start_method,
            resume_dir=resume_dir,
        )
