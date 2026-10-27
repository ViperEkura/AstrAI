"""Progress display callback."""

from __future__ import annotations

import sys
from typing import IO, TYPE_CHECKING, Optional

from tqdm import tqdm

from astrai.parallel import only_on_rank
from astrai.trainer.callbacks.base import TrainCallback
from astrai.trainer.step_plan import global_epoch_steps

if TYPE_CHECKING:
    from astrai.trainer.context import TrainContext


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
        if context.async_rollout is None:
            config = context.config
            total_steps = global_epoch_steps(
                context.dataset_len,
                context.epoch_sample_offset,
                context.samples_per_step,
                config.epoch_tail_policy,
                online=context.strategy.policy_version is not None,
                rl_update_epochs=config.rl_update_epochs,
                rl_minibatch_prompts=config.rl_minibatch_prompts,
            )
        else:
            total_steps = (
                len(context.dataloader) + context.executor.grad_accum_steps - 1
            ) // context.executor.grad_accum_steps
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
