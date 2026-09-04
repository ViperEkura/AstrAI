"""Unified training executor — parallel strategy + gradient accumulation."""

import contextlib
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FSDPModule,
    fully_shard,
)
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from astrai.factory import BaseFactory
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


def broadcast_state_dict(
    state_dict: Optional[Dict[str, torch.Tensor]],
    src: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    """Broadcast a state_dict from *src* rank to all ranks.

    Tensors stay on their original device (GPU) for the broadcast.
    All ranks must call this collectively.

    On non-distributed runs, returns *state_dict* unchanged.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return state_dict

    rank = dist.get_rank()

    # Broadcast metadata (keys, shapes, dtypes, device kind) so non-src
    # ranks can allocate matching empty tensors on their own device.
    if rank == src:
        first = next(iter(state_dict.values()), None)
        on_cuda = first is not None and first.is_cuda
        metadata = [
            (k, tuple(v.shape), v.dtype, on_cuda) for k, v in state_dict.items()
        ]
    else:
        metadata = None
    metadata_list = [metadata]
    dist.broadcast_object_list(metadata_list, src=src)
    metadata = metadata_list[0]

    # Non-src ranks allocate empty tensors with the broadcasted metadata.
    # The allocation must sit on *this* rank's local device: process groups
    # are pinned per-rank via ``device_id=``, so NCCL rejects a tensor
    # allocated on the src rank's device (and the cross-GPU allocation
    # would be wrong here anyway).
    if rank != src:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if any(m[3] for m in metadata)
            else torch.device("cpu")
        )
        state_dict = {
            k: torch.empty(s, dtype=d, device=device) for k, s, d, _ in metadata
        }

    # Broadcast each tensor in-place.
    for tensor in state_dict.values():
        dist.broadcast(tensor, src=src)

    return state_dict


def create_ref_model(
    model_fn: Callable[[], nn.Module],
    executor: Optional["BaseExecutor"] = None,
    model: Optional[nn.Module] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[str] = None,
) -> Optional[nn.Module]:
    """Create a frozen reference model from executor or state dict.

    In distributed mode (FSDP), ``unwrap_model`` returns ``None`` on
    non-rank-0.  The state_dict is broadcast from rank-0 to all ranks
    so every rank gets a complete copy.
    """
    if state_dict is None and executor is not None and model is not None:
        state_dict = executor.unwrap_model(model)

    # FSDP's unwrap_model returns None on non-rank-0. Broadcast from
    # rank-0 so every rank receives a complete state_dict.
    if executor is not None and executor.use_distributed:
        state_dict = broadcast_state_dict(state_dict)

    if state_dict is None:
        return None

    state_dict = strip_compile_prefix(state_dict)
    ref_model = model_fn()
    ref_model.load_state_dict(state_dict)
    ref_model.requires_grad_(False)
    ref_model.eval()
    if device is not None:
        ref_model = ref_model.to(device=device)
    return ref_model


class GradientState:
    def __init__(self, grad_accum_steps: int = 1):
        self.num_steps = max(grad_accum_steps, 1)
        self._step: int = 0
        self._sync_gradients: bool = True

    @property
    def sync_gradients(self) -> bool:
        return self._sync_gradients

    def _do_sync(self):
        self._step += 1
        self._sync_gradients = self._step % self.num_steps == 0


class AccumOptimizer:
    def __init__(self, optimizer: Optimizer, gradient_state: GradientState):
        self.optimizer = optimizer
        self.gradient_state = gradient_state

    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            self.optimizer.step(closure)

    def zero_grad(self):
        if self.gradient_state.sync_gradients:
            self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, d):
        self.optimizer.load_state_dict(d)


class AccumScheduler:
    def __init__(self, scheduler: LRScheduler, gradient_state: GradientState):
        self.scheduler = scheduler
        self.gradient_state = gradient_state

    def step(self):
        if self.gradient_state.sync_gradients:
            self.scheduler.step()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, d):
        self.scheduler.load_state_dict(d)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()


class BaseExecutor:
    def __init__(self, grad_accum_steps: int = 1):
        self.gradient_state = GradientState(grad_accum_steps)

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
            if scheduler_fn is not None:
                scheduler = scheduler_fn(optimizer)
            optimizer = AccumOptimizer(optimizer, self.gradient_state)
            if scheduler is not None:
                scheduler = AccumScheduler(scheduler, self.gradient_state)
        return model, optimizer, scheduler

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        return model

    def _no_sync(self, model: nn.Module):
        return contextlib.nullcontext()

    @contextmanager
    def accumulate(self, model: nn.Module):
        self.gradient_state._do_sync()
        if not self.gradient_state.sync_gradients:
            with self._no_sync(model):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def unwrap_model(self, model: nn.Module):
        return strip_compile_prefix(model.state_dict())

    @contextmanager
    def checkpoint_context(self, model: nn.Module):
        if self.use_distributed:
            dist.barrier()
        state_dict = self._gather_state_dict(model)
        yield state_dict
        if self.use_distributed:
            dist.barrier()

    def _gather_state_dict(self, model: nn.Module):
        state_dict = self.unwrap_model(model)
        if self.use_distributed and get_rank() != 0:
            return None
        return state_dict

    @property
    def use_distributed(self) -> bool:
        return get_world_size() > 1

    @property
    def sync_gradients(self) -> bool:
        return self.gradient_state.sync_gradients

    @property
    def grad_accum_steps(self) -> int:
        return self.gradient_state.num_steps

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> float:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if isinstance(total_norm, torch.Tensor):
            return total_norm.item()
        return total_norm


class ExecutorFactory(BaseFactory[BaseExecutor]):
    pass


@ExecutorFactory.register("none")
class NoneExecutor(BaseExecutor):
    pass


@ExecutorFactory.register("ddp")
class DDPExecutor(BaseExecutor):
    def __init__(
        self,
        grad_accum_steps: int = 1,
        dim: int = 0,
        broadcast_buffers: bool = True,
        init_sync: bool = True,
        process_group=None,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision=None,
        device_mesh=None,
    ):
        super().__init__(grad_accum_steps=grad_accum_steps)
        self._ddp_kwargs = dict(
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            init_sync=init_sync,
            process_group=process_group,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
            delay_all_reduce_named_params=delay_all_reduce_named_params,
            param_to_hook_all_reduce=param_to_hook_all_reduce,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
        )

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if not self.use_distributed:
            logger.warning("DDP backend selected but world_size=1, model not wrapped")
            return model
        local_device = os.environ.get("LOCAL_DEVICE")
        device_index = (
            torch.device(local_device).index if local_device is not None else None
        )
        local_rank = (
            device_index
            if device_index is not None
            else int(os.environ.get("LOCAL_RANK", get_rank()))
        )
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            **self._ddp_kwargs,
        )
        logger.info("Model wrapped with DDP (world_size=%d)", get_world_size())
        return model

    def _no_sync(self, model: nn.Module):
        if isinstance(model, DDP):
            return model.no_sync()
        return contextlib.nullcontext()

    def unwrap_model(self, model: nn.Module):
        if isinstance(model, DDP):
            return strip_compile_prefix(model.module.state_dict())
        return strip_compile_prefix(model.state_dict())


@ExecutorFactory.register("fsdp")
class FSDPExecutor(BaseExecutor):
    """FSDP executor using `torch.distributed.fsdp.fully_shard` (per-module API).

    Wraps each child module individually via ``fully_shard``.
    Skips the root model because ``ABC + Generic[T]`` in the MRO makes
    ``fully_shard``'s dynamic ``__class__`` assignment fail at the CPython level.
    Original ``Parameter`` objects are preserved (as DTensors) — no
    ``FlatParameter``, no ``use_orig_params=True`` hack.
    """

    def __init__(
        self,
        grad_accum_steps: int = 1,
        mesh: Optional[Any] = None,
        mp_policy: Optional[Any] = None,
        reshard_after_forward: bool = False,
    ):
        super().__init__(grad_accum_steps=grad_accum_steps)
        self._mesh = mesh
        self._mp_policy = mp_policy
        self._reshard_after_forward = reshard_after_forward

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if not self.use_distributed:
            logger.warning("FSDP backend selected but world_size=1, model not wrapped")
            return model

        kwargs = dict(
            mesh=self._mesh,
            mp_policy=self._mp_policy,
            reshard_after_forward=self._reshard_after_forward,
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        for child in model.children():
            if isinstance(child, nn.ModuleList):
                for sub in child:
                    fully_shard(sub, **kwargs)
            else:
                fully_shard(child, **kwargs)

        logger.info(
            "FSDP wrapping applied to %d direct children (root skipped for ABC compat)",
            len(list(model.children())),
        )
        return model

    @contextmanager
    def _no_sync(self, model: nn.Module):
        fsdp_modules = [m for m in model.modules() if isinstance(m, FSDPModule)]
        if fsdp_modules:
            for m in fsdp_modules:
                m.set_requires_gradient_sync(False, recurse=True)
            try:
                yield
            finally:
                for m in fsdp_modules:
                    m.set_requires_gradient_sync(True, recurse=True)
        else:
            yield

    def clip_grad_norm(self, model: nn.Module, max_norm: float) -> float:
        if not self.use_distributed:
            return super().clip_grad_norm(model, max_norm)

        # FSDP params are DTensors (sharded across ranks), and
        # get_total_norm reduces them to a globally-reduced replicated
        # DTensor — no extra all-reduce is needed. Manually summing the
        # local slice again would inflate the norm by sqrt(world_size)
        # and over-clip every step.
        total_norm = torch.nn.utils.get_total_norm(
            [p.grad for p in model.parameters() if p.grad is not None],
        )
        if isinstance(total_norm, DTensor):
            total_norm = total_norm.full_tensor()

        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef_clamped)

        return total_norm.item()

    def unwrap_model(self, model: nn.Module):
        if not self.use_distributed:
            return model.state_dict()

        # unshard() and full_tensor() are collective ops — all ranks must
        # participate. Non-rank-0 ranks still call them but discard results.
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.unshard()

        state_dict = model.state_dict()
        result = {}
        for k, v in state_dict.items():
            k = k.removeprefix(_COMPILE_PREFIX)
            if isinstance(v, DTensor):
                full = v.full_tensor()
                if get_rank() == 0:
                    result[k] = full
            elif get_rank() == 0:
                result[k] = v

        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

        if get_rank() != 0:
            return None

        return result
