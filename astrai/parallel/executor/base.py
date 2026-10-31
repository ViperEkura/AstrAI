"""Executor base: parallel-agnostic training surface and checkpoint ownership."""

import contextlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from astrai.factory import BaseFactory
from astrai.parallel.executor.gradients import (
    FP32GradientAccumulator,
    parameter_gradient,
)
from astrai.parallel.setup import get_rank, get_world_size

logger = logging.getLogger(__name__)

_COMPILE_PREFIX = "_orig_mod."


def strip_compile_prefix(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Drop the ``_orig_mod.`` key prefix ``torch.compile`` adds.

    ``OptimizedModule.state_dict()`` prefixes every key, so checkpoints or
    reference-model copies taken from a compiled model fail to load into a
    plain module (strict) or silently load nothing (non-strict).  Stripping
    here, at the single source every consumer reads from, keeps saved keys
    canonical regardless of compile mode.
    """
    if any(key.startswith(_COMPILE_PREFIX) for key in state_dict):
        state_dict = {
            key.removeprefix(_COMPILE_PREFIX): value
            for key, value in state_dict.items()
        }
    return state_dict


@dataclass(frozen=True)
class RolloutCapabilities:
    """Executor support for sharing its training model with inference."""

    supports_in_process: bool
    reason: Optional[str] = None


def broadcast_state_dict(
    state_dict: Optional[Dict[str, torch.Tensor]],
    src: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    """Broadcast a state_dict from *src* rank to all ranks.

    CPU tensors stay on CPU. Accelerator tensors are mapped to each rank's
    current local device before the broadcast, so rank-local model copies do
    not accidentally allocate on the source rank's GPU. All ranks must call
    this collectively.

    On non-distributed runs, returns *state_dict* unchanged.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return state_dict

    rank = dist.get_rank()

    # Broadcast metadata (keys, shapes, dtypes, device kind) so non-src
    # ranks can allocate matching empty tensors on their own device.
    if rank == src:
        metadata = [
            (k, tuple(v.shape), v.dtype, str(v.device)) for k, v in state_dict.items()
        ]
    else:
        metadata = None
    metadata_list = [metadata]
    dist.broadcast_object_list(metadata_list, src=src)
    metadata = metadata_list[0]

    # Non-src ranks reuse a compatible local state dict when one is available.
    # This is the common DDP path and, importantly, preserves each rank's local
    # device. FSDP may only materialize the full state dict on src; in that
    # case allocate on the current local accelerator rather than copying the
    # source rank's device index (for example cuda:0 onto rank 1).
    if rank != src:
        can_reuse_local = state_dict is not None and all(
            key in state_dict
            and tuple(state_dict[key].shape) == shape
            and state_dict[key].dtype == dtype
            for key, shape, dtype, _device in metadata
        )
        if can_reuse_local:
            state_dict = {key: state_dict[key] for key, *_rest in metadata}
        else:

            def local_device(source_device: str) -> torch.device:
                device = torch.device(source_device)
                if device.type == "cuda" and torch.cuda.is_available():
                    return torch.device("cuda", torch.cuda.current_device())
                if (
                    device.type == "xpu"
                    and hasattr(torch, "xpu")
                    and torch.xpu.is_available()
                ):
                    return torch.device("xpu", torch.xpu.current_device())
                return device

            state_dict = {
                key: torch.empty(shape, dtype=dtype, device=local_device(source_device))
                for key, shape, dtype, source_device in metadata
            }

    # Broadcast each tensor in-place.
    for tensor in state_dict.values():
        dist.broadcast(tensor, src=src)

    return state_dict


class BaseExecutor:
    def __init__(self, grad_accum_steps: int = 1, grad_accum_dtype: str = "float32"):
        self.grad_accum_steps = max(grad_accum_steps, 1)
        self._sync_gradients = True
        if grad_accum_dtype not in ("float32", "model"):
            raise ValueError("grad_accum_dtype must be float32 or model")
        self.grad_accum_dtype = grad_accum_dtype
        self._grad_accumulator = None

    def prepare(
        self,
        model_fn: Callable[[], nn.Module],
        optimizer_fn: Optional[Callable[[nn.Module], Optimizer]] = None,
        scheduler_fn: Optional[Callable[[Optimizer], LRScheduler]] = None,
        before_wrap: Optional[Callable[[nn.Module], nn.Module]] = None,
        after_wrap: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> Tuple[nn.Module, Optional[Optimizer], Optional[LRScheduler]]:
        model = model_fn()
        if before_wrap is not None:
            model = before_wrap(model)
        model = self._prepare_model(model)
        if after_wrap is not None:
            model = after_wrap(model)
        optimizer = None
        scheduler = None
        if optimizer_fn is not None:
            optimizer = optimizer_fn(model)
            if self._grad_accumulator is not None and not getattr(
                optimizer, "supports_main_grad", False
            ):
                self._grad_accumulator.close()
                self._grad_accumulator = None
                raise TypeError(
                    "FP32 accumulation requires an optimizer supporting main_grad; "
                    "use FP32Optimizer or grad_accum_dtype='model'"
                )
            if scheduler_fn is not None:
                scheduler = scheduler_fn(optimizer)
        return model, optimizer, scheduler

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        self._prepare_gradients(model)
        return model

    def _prepare_gradients(self, model: nn.Module, bucket_cap_mb: float = 25) -> None:
        if self.grad_accum_dtype == "float32" and any(
            p.requires_grad and p.dtype == torch.bfloat16 for p in model.parameters()
        ):
            self._grad_accumulator = FP32GradientAccumulator(
                model, lambda: self.sync_gradients, bucket_cap_mb
            )

    def rollout_capabilities(self) -> RolloutCapabilities:
        """Describe whether an in-process rollout may share this model."""
        return RolloutCapabilities(supports_in_process=True)

    def model_for_inference(self, model: nn.Module) -> nn.Module:
        """Return the executor-owned model view supported by inference."""
        capabilities = self.rollout_capabilities()
        if not capabilities.supports_in_process:
            raise RuntimeError(
                capabilities.reason or "In-process rollout is unsupported"
            )
        return model

    def _no_sync(self, model: nn.Module):
        return contextlib.nullcontext()

    @contextmanager
    def accumulate_explicit(self, model: nn.Module, sync: bool):
        """Use an explicit collective boundary for a global data window."""
        self._sync_gradients = sync
        if sync:
            yield
        else:
            with self._no_sync(model):
                yield

    def backward(self, loss: torch.Tensor):
        if self._grad_accumulator is not None:
            self._grad_accumulator.begin_backward()
        loss.backward()
        if self._grad_accumulator is not None:
            self._grad_accumulator.finish_backward()

    def model_state_dict(self, model: nn.Module):
        """The checkpoint payload of *model*, without the compile prefix."""
        return strip_compile_prefix(model.state_dict())

    @property
    def writes_checkpoints(self) -> bool:
        """Whether this rank owns the published checkpoint payload."""
        return not self.use_distributed or get_rank() == 0

    @contextmanager
    def checkpoint_context(self, model: nn.Module):
        if self.use_distributed:
            dist.barrier()
        state_dict = self._gather_state_dict(model)
        yield state_dict
        if self.use_distributed:
            dist.barrier()

    def _gather_state_dict(self, model: nn.Module):
        state_dict = self.model_state_dict(model)
        if not self.writes_checkpoints:
            return None
        return state_dict

    @property
    def use_distributed(self) -> bool:
        return get_world_size() > 1

    @property
    def sync_gradients(self) -> bool:
        """Observe the current explicit boundary; optimizer calls are ungated."""
        return self._sync_gradients

    def scale_gradients(self, model: nn.Module, factor: float) -> None:
        if self._grad_accumulator is None:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
            return
        gradients = [
            gradient
            for parameter in model.parameters()
            if (gradient := parameter_gradient(parameter)) is not None
        ]
        if gradients:
            torch._foreach_mul_(gradients, factor)

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> float:
        if self._grad_accumulator is None:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            gradients = [
                gradient
                for parameter in model.parameters()
                if (gradient := parameter_gradient(parameter)) is not None
            ]
            total_norm = torch.nn.utils.get_total_norm(gradients)
            if gradients:
                coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
                torch._foreach_mul_(gradients, coefficient)
        return float(total_norm)


class ExecutorFactory(BaseFactory[BaseExecutor]):
    pass


@ExecutorFactory.register("none")
class NoneExecutor(BaseExecutor):
    pass
