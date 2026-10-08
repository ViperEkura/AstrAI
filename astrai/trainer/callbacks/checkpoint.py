"""Checkpoint persistence callback."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable

from astrai.serialization import Checkpoint
from astrai.trainer.callbacks.base import CallbackFactory, TrainCallback
from astrai.trainer.optional_extras import (
    checkpoint_extras,
    snapshot_component_extras,
)
from astrai.trainer.train_context import TrainContext

logger = logging.getLogger(__name__)

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _copy_tokenizer_files(param_path: str | None, save_path: str):
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
        save_extra_fn: Callable[["TrainContext"], dict] | None = None,
    ):
        self.save_dir = save_dir
        self.interval = interval
        self.weight_only = weight_only
        self.save_extra_fn = save_extra_fn or CheckpointCallback.save_extra
        self.last_ckpt_step = None
        self.last_consumed_samples = None
        self._saved = False

    def on_train_begin(self, context: TrainContext):
        self.last_ckpt_step = context.optimizer_step
        self.last_consumed_samples = context.consumed_samples

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
                if context.optimizer_steps is not None:
                    meta["optimizer_steps"] = context.optimizer_steps
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
        self.last_consumed_samples = context.consumed_samples
        self._saved = True

    def after_optimizer_step(self, context: TrainContext):
        if context.optimizer_steps is not None:
            return
        if context.optimizer_step - self.last_ckpt_step >= self.interval:
            self._save_checkpoint(context)

    def on_batch_end(self, context: TrainContext):
        if (
            context.optimizer_steps is not None
            and context.optimizer_step - self.last_ckpt_step >= self.interval
        ):
            self._save_checkpoint(context)

    def _incomplete_round(self, context):
        if context.optimizer_steps is not None and context.kwargs.get(
            "grpo_update_in_progress"
        ):
            logger.warning(
                "GRPO stopped during an incomplete round; retain the last completed-round checkpoint"
            )
            return True
        return False

    def on_train_end(self, context: TrainContext):
        if self._incomplete_round(context):
            return
        if context.optimizer_step != self.last_ckpt_step or (
            context.optimizer_steps is not None
            and context.consumed_samples != self.last_consumed_samples
        ):
            self._save_checkpoint(context)

    def on_error(self, context: TrainContext):
        if self._incomplete_round(context):
            return
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
        # Global accelerator state (fp8 rings, RNG) via the extras registry.
        extra.update(checkpoint_extras())
        # Strategy-owned tensor state (critic, frozen reference, ...) via the
        # declarative table in optional_extras: which keys exist, where the
        # state lives and how it is snapshotted are declared once there, so
        # the checkpoint callback never grows per-component branches.
        extra.update(
            snapshot_component_extras(
                context.strategy, getattr(context, "config", None)
            )
        )
        return extra
