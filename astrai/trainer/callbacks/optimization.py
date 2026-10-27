"""Callbacks for gradient and activation-memory handling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Type

from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from astrai.trainer.callbacks.base import TrainCallback

if TYPE_CHECKING:
    from astrai.trainer.context import TrainContext

logger = logging.getLogger(__name__)


class GradientClippingCallback(TrainCallback):
    """
    Gradient clipping callback for trainer.
    """

    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = max_grad_norm

    def before_optimizer_step(self, context: TrainContext):
        context.grad_norm = context.executor.clip_grad_norm(
            context.model, self.max_grad_norm
        )


class GradientCheckpointingCallback(TrainCallback):
    """
    Activation checkpointing callback — trades compute for memory
    by recomputing specified module activations during the backward pass.

    Args:
        modules: Module types to apply checkpointing to.
    """

    def __init__(self, modules: Optional[List[Type]] = None):
        self.modules = tuple(modules) if modules else ()

    def _enable(self, module: nn.Module):
        if self.modules and isinstance(module, self.modules):
            fn = module.forward
            module._original_forward = fn
            module.forward = lambda *a, **kw: torch_checkpoint(
                fn, *a, use_reentrant=False, **kw
            )

    @staticmethod
    def _disable(module: nn.Module):
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            del module._original_forward

    def on_train_begin(self, context: TrainContext):
        if not self.modules:
            return
        context.model.apply(self._enable)
        logger.info("Gradient checkpointing enabled")

    def on_train_end(self, context: TrainContext):
        context.model.apply(self._disable)
