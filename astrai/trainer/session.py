"""Own the training lifecycle, including failures during startup."""

import logging
import sys
from collections.abc import Callable
from typing import Optional

import torch.distributed as dist

from astrai.signal_handler import (
    ignore_worker_deaths,
    register_signal_handlers,
    unregister_signal_handlers,
)
from astrai.trainer.context import TrainContext

logger = logging.getLogger(__name__)


class TrainSession:
    """Run one training context, then hand it back to every observer in order.

    Shutdown is a plan, not a cascade of ``if``s: :meth:`_shutdown_steps` owns
    the order the invariants require, and one failing step does not skip the
    rest.
    """

    def __init__(
        self,
        build_context: Callable[[], TrainContext],
        call_callbacks: Callable[[str, TrainContext], None],
    ) -> None:
        self._build_context = build_context
        self._call_callbacks = call_callbacks
        self.context: Optional[TrainContext] = None
        self._signal_registered = False

    def __enter__(self) -> TrainContext:
        self.context = self._build_context()
        try:
            self._signal_registered = True
            register_signal_handlers(self.context)
            self._call_callbacks("on_train_begin", self.context)
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self.context

    def _shutdown_steps(self, context: TrainContext, failed: bool):
        """The cleanup steps of one session, in the order they must run.

        The registered hooks go first: they stop the DataLoader workers, and a
        worker killed by the same signal that stopped training would otherwise
        raise from PyTorch's SIGCHLD handler in the middle of the checkpoint
        save.  The observers then persist state while the rollout is still
        alive, because an online checkpoint captures state the rollout owns.
        Every step is attempted even when an earlier one raised.
        """
        steps: list[tuple[str, Callable[[], None]]] = [
            ("shutdown_hooks", context.run_shutdown),
            ("discard_pending_metrics", context.metric_history.discard_pending),
            (
                "on_shutdown",
                lambda: self._call_callbacks("on_shutdown", context),
            ),
        ]
        if context.async_rollout is not None:
            steps.append(("close_rollout", context.async_rollout.close))
        steps.append(
            ("on_train_end", lambda: self._call_callbacks("on_train_end", context))
        )
        if not failed and context.executor.use_distributed and dist.is_initialized():
            steps.append(("barrier", dist.barrier))
        if self._signal_registered:
            steps.append(("unregister_signal_handlers", unregister_signal_handlers))
            self._signal_registered = False
        return steps

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        context = self.context
        if context is None:
            return False
        failed = exc_type is not None
        if failed:
            logger.error(
                "Training failed: %s",
                exc_value,
                exc_info=(exc_type, exc_value, traceback),
            )
        cleanup_error = None
        with ignore_worker_deaths():
            for label, action in self._shutdown_steps(context, failed):
                try:
                    action()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
                    logger.exception("Training session cleanup failed (%s)", label)
        if not failed and cleanup_error is not None:
            raise cleanup_error
        return False
