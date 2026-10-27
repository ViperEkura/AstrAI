"""Checkpoint persistence callback."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from astrai.serialization import Checkpoint
from astrai.trainer.callbacks.base import TrainCallback
from astrai.trainer.optional_extras import (
    snapshot_component_extras,
    snapshot_extras,
)

if TYPE_CHECKING:
    from astrai.trainer.context import TrainContext
    from astrai.trainer.context.metric import MetricSnapshot


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _copy_tokenizer_files(param_path: Optional[str], save_path: str):
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


def _write_metric_snapshot(snapshot: MetricSnapshot, destination: Path) -> None:
    with destination.open("wb") as output:
        if snapshot.source_path is not None:
            with snapshot.source_path.open("rb") as source:
                shutil.copyfileobj(source, output)
        for record in snapshot.records:
            output.write(record.encode("utf-8"))


class CheckpointCallback(TrainCallback):
    """
    Checkpoint callback for trainer.
    """

    def __init__(
        self,
        save_dir: str,
        interval: int,
        save_extra_fn: Optional[Callable[["TrainContext"], Dict]] = None,
    ):
        self.save_dir = save_dir
        self.interval = interval
        self.save_extra_fn = save_extra_fn or CheckpointCallback.save_extra
        self.last_ckpt_step = None
        self.last_ckpt_cursor = None
        self._saved = False

    def on_train_begin(self, context: TrainContext):
        self.last_ckpt_step = context.optimizer_step
        self.last_ckpt_cursor = context.consumed_samples

    def _save_checkpoint(self, context: TrainContext):
        if not context.checkpoint_safe:
            return
        with context.executor.checkpoint_context(context.model) as state_dict:
            if state_dict is not None:
                save_path = os.path.join(
                    self.save_dir,
                    f"epoch_{context.epoch}_step_{context.optimizer_step}",
                )
                extra = self.save_extra_fn(context)
                context.checkpoint = Checkpoint(
                    state_dict=state_dict,
                    epoch=context.epoch,
                    # The cursor includes committed batches and any discarded
                    # epoch tail; the step identifies the saved model weights.
                    consumed_samples=context.consumed_samples,
                    config=context.model_config,
                    extra=extra,
                    meta=self._meta(context),
                )

                snapshot = context.metric_history.snapshot()

                def prepare_staging(staging_path: Path) -> None:
                    _copy_tokenizer_files(context.param_path, str(staging_path))
                    _write_metric_snapshot(snapshot, staging_path / "metric.jsonl")

                published = context.checkpoint.save(
                    save_path, prepare_staging=prepare_staging
                )
                if published:
                    context.metric_history.mark_saved(Path(save_path))
                elif snapshot.records:
                    raise RuntimeError(
                        f"Checkpoint {save_path} already exists but has unsaved metrics"
                    )
        self.last_ckpt_step = context.optimizer_step
        self.last_ckpt_cursor = context.consumed_samples
        self._saved = True

    @staticmethod
    def _meta(context: TrainContext) -> Dict:
        """Run state plus what a report reads; not a copy of the CLI config."""
        cfg = context.config
        meta = {
            "optimizer_step": context.optimizer_step,
            "strategy": cfg.strategy,
            "dp_size": context.dp_size,
            "cp_size": cfg.cp_size,
            "batch_per_device": cfg.batch_per_device,
            "grad_accum_steps": cfg.grad_accum_steps,
            "epoch_tail_policy": cfg.epoch_tail_policy,
            # The resume check compares these against the configured types.
            "optimizer": {"type": cfg.optimizer.type},
            "scheduler": {"type": cfg.scheduler.type},
        }
        meta.update(
            resume_contract_version=1,
            dataset_len=context.dataset_len,
            epoch_sample_offset=context.epoch_sample_offset,
            random_seed=cfg.random_seed,
            tp_size=cfg.tp_size,
            dp_mode=cfg.dp_mode,
            rollout_mode=cfg.rollout_mode,
            rl_update_epochs=cfg.rl_update_epochs,
            rl_minibatch_prompts=cfg.rl_minibatch_prompts,
        )
        policy_version = context.strategy.policy_version
        if policy_version is not None:
            meta["policy_version"] = policy_version
        return meta

    def after_optimizer_step(self, context: TrainContext):
        if context.optimizer_step - self.last_ckpt_step >= self.interval:
            self._save_checkpoint(context)

    def _save_pending(self, context: TrainContext):
        """Persist the current step unless that step is already on disk.

        A discarded epoch tail advances the data cursor without changing the
        optimizer step, so both values determine whether a save is needed.
        """
        if (
            self._saved
            and context.optimizer_step == self.last_ckpt_step
            and context.consumed_samples == self.last_ckpt_cursor
        ):
            return
        self._save_checkpoint(context)

    def on_shutdown(self, context: TrainContext):
        # An interrupted run must always leave a checkpoint behind: on a slow
        # start the signal can arrive before the first optimizer step, where
        # the change-based guard alone would skip the save entirely.
        self._save_pending(context)

    def on_train_end(self, context: TrainContext):
        if (
            context.optimizer_step != self.last_ckpt_step
            or context.consumed_samples != self.last_ckpt_cursor
        ):
            self._save_checkpoint(context)

    @staticmethod
    def save_extra(context: TrainContext) -> Dict:
        extra = snapshot_extras(context)
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
