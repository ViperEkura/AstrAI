"""Contiguous main gradients with bucketed asynchronous reduction."""

import contextlib
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.utils import (
    _sync_module_states,
    _verify_param_shape_across_processes,
)


def parameter_gradient(parameter: nn.Parameter) -> Optional[torch.Tensor]:
    """Return the accumulated main gradient, or the native autograd gradient."""
    main_grad = getattr(parameter, "main_grad", None)
    return main_grad if main_grad is not None else parameter.grad


def clip_model_gradients(model, max_norm):
    parameters = list(model.parameters())
    if not any(getattr(p, "main_grad", None) is not None for p in parameters):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    gradients = [g for p in parameters if (g := parameter_gradient(p)) is not None]
    if not gradients:
        return torch.tensor(0.0)
    if all(g.dtype == torch.float32 for g in gradients):
        total = torch.nn.utils.get_total_norm(gradients)
    else:
        total = torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(g, dtype=torch.float32) for g in gradients]
            )
        )
    torch._foreach_mul_(gradients, (max_norm / (total + 1e-6)).clamp(max=1.0))
    return total


def _accumulate_gradient(accumulator, parameter: nn.Parameter) -> None:
    gradient = parameter.grad
    if gradient is None:
        return
    if gradient.is_sparse:
        raise NotImplementedError("Main gradients require dense gradients")
    target = accumulator.views[parameter]
    if parameter.main_grad is None:
        target.copy_(gradient)
        parameter.main_grad = target
    else:
        target.add_(gradient)
    parameter.grad = None
    if accumulator.distributed and accumulator.is_synchronizing:
        accumulator.mark_ready(parameter)


@dataclass
class GradientBucket:
    parameters: List[nn.Parameter]
    buffer: torch.Tensor
    ready: Set[nn.Parameter] = field(default_factory=set)
    work: Optional[Any] = None


class MainGradientAccumulator:
    """Keep compute weights separate from persistent main-gradient storage.

    Following Megatron's main_grad design, each parameter has a view into one
    contiguous communication bucket. The same storage accumulates backward
    contributions and participates in same-dtype all-reduce. No BF16 reducer buckets
    or per-update gradient packing are retained. This preserves precision
    across backward calls, not inside individual BF16 backward operators.
    """

    def __init__(
        self,
        model: nn.Module,
        synchronize: Callable[[], bool],
        bucket_cap_mb: float = 25,
        dtype: torch.dtype = torch.float32,
    ):
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Gradient dtype must be float32 or bfloat16")
        self.dtype = dtype
        self.element_size = torch.empty((), dtype=dtype).element_size()
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if any(isinstance(p, DTensor) for p in self.parameters):
            raise NotImplementedError("Main gradients do not support DTensor")
        if any(p.dtype not in (torch.float32, torch.bfloat16) for p in self.parameters):
            raise ValueError("Main gradients support FP32 and BF16 parameters")
        if bucket_cap_mb <= 0:
            raise ValueError("bucket_cap_mb must be positive")
        if any(hasattr(p, "main_grad") for p in self.parameters):
            raise ValueError("Model already has a main-gradient owner")
        self.synchronize = synchronize
        self.sync_enabled = True
        self.process_group = None
        self.distributed = False
        self.find_unused_parameters = False
        self.views: Dict[nn.Parameter, torch.Tensor] = {}
        self.buckets: List[GradientBucket] = []
        self.parameter_buckets: Dict[nn.Parameter, GradientBucket] = {}
        chunk = []
        size = 0
        # Reverse registration order approximates backward readiness. Every
        # rank uses the same order, including when local hook order differs.
        for parameter in reversed(self.parameters):
            if chunk and (
                size + parameter.numel() * self.element_size > bucket_cap_mb * 1024**2
                or chunk[0].device != parameter.device
            ):
                self._add_bucket(chunk)
                chunk, size = [], 0
            chunk.append(parameter)
            size += parameter.numel() * self.element_size
        if chunk:
            self._add_bucket(chunk)
        self.next_bucket = 0
        self.handles = [
            p.register_post_accumulate_grad_hook(partial(_accumulate_gradient, self))
            for p in self.parameters
        ]

    @property
    def is_synchronizing(self) -> bool:
        return self.sync_enabled and self.synchronize()

    def _add_bucket(self, parameters) -> None:
        storage = torch.empty(
            sum(p.numel() for p in parameters),
            device=parameters[0].device,
            dtype=self.dtype,
        )
        bucket = GradientBucket(parameters, storage)
        self.buckets.append(bucket)
        offset = 0
        for parameter in parameters:
            self.views[parameter] = storage[
                offset : offset + parameter.numel()
            ].view_as(parameter)
            self.parameter_buckets[parameter] = bucket
            parameter.main_grad = None
            offset += parameter.numel()

    def begin_backward(self) -> None:
        if not self.distributed or not self.is_synchronizing:
            return
        self.next_bucket = 0
        for bucket in self.buckets:
            bucket.ready.clear()
            bucket.work = None

    def mark_ready(self, parameter) -> None:
        bucket = self.parameter_buckets[parameter]
        if parameter in bucket.ready:
            raise RuntimeError("A parameter became ready twice in one backward")
        bucket.ready.add(parameter)
        self._launch_ready_buckets()

    def _launch_ready_buckets(self) -> None:
        if not self.distributed:
            return
        while self.next_bucket < len(self.buckets):
            bucket = self.buckets[self.next_bucket]
            if len(bucket.ready) != len(bucket.parameters):
                break
            bucket.work = dist.all_reduce(
                bucket.buffer, group=self.process_group, async_op=True
            )
            self.next_bucket += 1

    def finish_backward(self) -> None:
        if not self.distributed or not self.is_synchronizing:
            return
        missing = [p for p in self.parameters if p.main_grad is None]
        if missing and not self.find_unused_parameters:
            raise RuntimeError(
                "A trainable parameter has no gradient in this accumulation window; "
                "set find_unused_parameters=True for conditional models"
            )
        # A parameter used in an earlier no_sync backward still contributes to
        # the window, even if it is absent from the final microbatch.
        for parameter in missing:
            self.views[parameter].zero_()
        for bucket in self.buckets:
            bucket.ready.update(bucket.parameters)
        self._launch_ready_buckets()
        size = dist.get_world_size(self.process_group)
        used = None
        if self.find_unused_parameters:
            used = torch.tensor(
                [p.main_grad is not None for p in self.parameters],
                dtype=torch.int32,
                device=self.parameters[0].device,
            )
            dist.all_reduce(used, group=self.process_group)
        for bucket in self.buckets:
            bucket.work.wait()
        torch._foreach_mul_([b.buffer for b in self.buckets], 1.0 / size)
        if used is not None:
            for parameter, count in zip(self.parameters, used.tolist()):
                parameter.main_grad = self.views[parameter] if count else None

    def close(self) -> None:
        if not self.handles:
            return
        for handle in self.handles:
            handle.remove()
        for parameter in self.parameters:
            del parameter.main_grad
        self.handles.clear()
        self.views.clear()
        self.parameter_buckets.clear()
        self.buckets.clear()


class GradientDataParallel(nn.Module):
    """Replicated compute module sharing its accumulation/reduction storage."""

    def __init__(
        self,
        module: nn.Module,
        accumulator: MainGradientAccumulator,
        process_group=None,
        forward_sync_buffers: bool = True,
        init_sync: bool = True,
        find_unused_parameters: bool = False,
    ):
        super().__init__()
        if len({p.device for p in module.parameters()}) != 1:
            raise ValueError("Replicated gradient reduction requires one model device")
        self.module = module
        self.training = module.training
        self.accumulator = accumulator
        self.process_group = (
            process_group if process_group is not None else dist.group.WORLD
        )
        self.forward_sync_buffers = forward_sync_buffers
        self.require_forward_param_sync = True
        accumulator.process_group = self.process_group
        accumulator.distributed = True
        accumulator.find_unused_parameters = find_unused_parameters
        if init_sync:
            _verify_param_shape_across_processes(
                self.process_group, list(module.parameters())
            )
            _sync_module_states(module, self.process_group, 250 * 1024**2, 0, [], True)

    def forward(self, *args, **kwargs):
        if self.forward_sync_buffers and self.require_forward_param_sync:
            buffers = list(self.module.buffers())
            if buffers:
                dist._broadcast_coalesced(self.process_group, buffers, 250 * 1024**2, 0)
        self.require_forward_param_sync = self.accumulator.is_synchronizing
        return self.module(*args, **kwargs)

    @contextlib.contextmanager
    def no_sync(self):
        previous = self.accumulator.sync_enabled
        self.accumulator.sync_enabled = False
        try:
            yield
        finally:
            self.accumulator.sync_enabled = previous
