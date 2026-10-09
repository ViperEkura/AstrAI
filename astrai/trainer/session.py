"""Own the training lifecycle, including failures during startup."""

import logging
import sys
from collections.abc import Callable
from typing import Optional

import torch.distributed as dist

from astrai.signal_handler import register_signal_handlers, unregister_signal_handlers
from astrai.trainer.train_context import TrainContext

logger = logging.getLogger(__name__)


class TrainSession:
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

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        context = self.context
        if context is None:
            return False
        cleanup_error = None

        def run(action):
            nonlocal cleanup_error
            try:
                action()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                logger.exception("Training session cleanup failed")

        if exc_type is not None:
            logger.error(
                "Training failed: %s",
                exc_value,
                exc_info=(exc_type, exc_value, traceback),
            )
            run(lambda: self._call_callbacks("on_error", context))
        if context.async_rollout is not None:
            run(context.async_rollout.close)
        run(lambda: self._call_callbacks("on_train_end", context))
        if (
            exc_type is None
            and context.executor.use_distributed
            and dist.is_initialized()
        ):
            run(dist.barrier)
        if self._signal_registered:
            run(unregister_signal_handlers)
            self._signal_registered = False
        if exc_type is None and cleanup_error is not None:
            raise cleanup_error
        return False
