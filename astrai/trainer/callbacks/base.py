"""Shared callback protocol and optional lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from torch import Tensor

if TYPE_CHECKING:
    from astrai.trainer.context import TrainContext


@dataclass(frozen=True)
class RolloutMetricsBatch:
    """CPU snapshot of freshly scored generations, excluding padding prompts."""

    policy_version: int
    rewards: Tensor
    response_lengths: Tensor
    prompt_texts: Tuple[str, ...]
    finish_reasons: Tuple[Tuple[str, ...], ...]


RolloutMetricTotals = Dict[str, Tuple[float, float]]


@runtime_checkable
class TrainCallback(Protocol):
    """
    Callback interface for trainer.
    """

    def on_train_begin(self, context: TrainContext):
        """Called at the beginning of training."""

    def on_train_end(self, context: TrainContext):
        """Called at the end of training."""

    def on_epoch_begin(self, context: TrainContext):
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, context: TrainContext):
        """Called at the end of each epoch."""

    def on_batch_begin(self, context: TrainContext):
        """Called at the beginning of each batch."""

    def on_batch_end(self, context: TrainContext):
        """Called at the end of each batch."""

    def before_optimizer_step(self, context: TrainContext):
        """Called immediately before every optimizer step (sync step only)."""

    def after_optimizer_step(self, context: TrainContext):
        """Called after the optimizer and scheduler step (sync step only)."""

    def on_rollout_end(
        self, context: TrainContext, batch: RolloutMetricsBatch
    ) -> Optional[RolloutMetricTotals]:
        """Contribute named (numerator, denominator) pairs for a fresh rollout."""

    def on_log(self, context: TrainContext, record: Dict[str, Any]) -> None:
        """Observe one committed metric record on rank zero."""

    def on_shutdown(self, context: TrainContext):
        """Called once when the run ends, before ``on_train_end``.

        Persist whatever must survive the process: a graceful stop and a
        failure both arrive here.  The callbacks must be idempotent, because a
        normal end also passes through it.
        """
