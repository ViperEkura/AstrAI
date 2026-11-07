"""FSDP executor on the per-module ``fully_shard`` API."""

import logging
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FSDPModule,
    fully_shard,
)
from torch.distributed.tensor import DTensor

from astrai.model.components.decoder_block import DecoderBlock
from astrai.parallel.executor.base import (
    _COMPILE_PREFIX,
    BaseExecutor,
    ExecutorFactory,
    RolloutCapabilities,
)

logger = logging.getLogger(__name__)


@ExecutorFactory.register("fsdp")
class FSDPExecutor(BaseExecutor):
    """FSDP executor using `torch.distributed.fsdp.fully_shard` (per-module API).

    Shards distinct decoder blocks bottom-up, then the root's remaining
    parameters so tied embeddings and output heads have one owner. Generic
    backbones with a ``layers`` ModuleList keep layerwise wrapping support.
    Set ``layerwise=False`` for one coarse root group. Accumulation can keep
    parameters unsharded between microbatches to avoid repeated all-gathers.
    Original ``Parameter`` objects are preserved (as DTensors) — no
    ``FlatParameter``, no ``use_orig_params=True`` hack.
    """

    def __init__(
        self,
        grad_accum_steps: int = 1,
        mesh: Optional[Any] = None,
        mp_policy: Optional[Any] = None,
        reshard_after_forward: bool | int = True,
        layerwise: bool = True,
        keep_unsharded_during_accumulation: bool = True,
        *,
        root_reshard_after_forward: bool | int = False,
    ):
        super().__init__(grad_accum_steps=grad_accum_steps)
        if not isinstance(layerwise, bool):
            raise ValueError("layerwise must be a boolean")
        if not isinstance(keep_unsharded_during_accumulation, bool):
            raise ValueError("keep_unsharded_during_accumulation must be a boolean")
        self._mesh = mesh
        self._mp_policy = mp_policy
        self._reshard_after_forward = reshard_after_forward
        self._root_reshard_after_forward = root_reshard_after_forward
        self._layerwise = layerwise
        self._keep_unsharded_during_accumulation = keep_unsharded_during_accumulation

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if not self.use_distributed:
            logger.warning("FSDP backend selected but world_size=1, model not wrapped")
            return model

        if any(isinstance(module, FSDPModule) for module in model.modules()):
            raise ValueError("FSDP executor requires an unsharded model")
        blocks = []
        if self._layerwise:
            blocks = [
                module for module in model.modules() if isinstance(module, DecoderBlock)
            ]
            if not blocks:
                # Retain support for generic backbones from the layerwise executor.
                for child in model.children():
                    layers = (
                        child
                        if isinstance(child, nn.ModuleList)
                        else getattr(child, "layers", None)
                    )
                    if isinstance(layers, nn.ModuleList):
                        blocks.extend(layers)
                blocks = list(dict.fromkeys(blocks))
        block_modules = {id(module) for block in blocks for module in block.modules()}
        root_params = {
            id(param)
            for module in model.modules()
            if id(module) not in block_modules
            for param in module.parameters(recurse=False)
        }
        owners = {}
        for index, block in enumerate(blocks):
            for param in block.parameters():
                key = id(param)
                if key in owners or key in root_params:
                    raise ValueError(
                        "FSDP cannot shard a parameter shared across distinct decoder blocks or between a block and the root"
                    )
                owners[key] = index

        kwargs = dict(
            mesh=self._mesh,
            mp_policy=self._mp_policy,
            reshard_after_forward=self._reshard_after_forward,
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        for block in blocks:
            fully_shard(block, **kwargs)
        fully_shard(
            model,
            **{**kwargs, "reshard_after_forward": self._root_reshard_after_forward},
        )

        logger.info(
            "FSDP wrapping applied to %d distinct blocks and one root",
            len(blocks),
        )
        return model

    def rollout_capabilities(self):
        if self.use_distributed:
            return RolloutCapabilities(
                supports_in_process=False,
                reason=(
                    "Distributed FSDP online rollout is not supported: inference "
                    "requires a replicated model view, but parameters are sharded"
                ),
            )
        return super().rollout_capabilities()

    @contextmanager
    def _no_sync(self, model: nn.Module):
        fsdp_modules = [m for m in model.modules() if isinstance(m, FSDPModule)]
        if not fsdp_modules:
            yield
            return
        for module in fsdp_modules:
            if self._keep_unsharded_during_accumulation:
                module.set_reshard_after_backward(False, recurse=False)
            module.set_requires_gradient_sync(False, recurse=False)
        try:
            yield
        finally:
            for module in fsdp_modules:
                module.set_requires_gradient_sync(True, recurse=False)
                if self._keep_unsharded_during_accumulation:
                    module.set_reshard_after_backward(True, recurse=False)

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

    def model_state_dict(self, model: nn.Module):
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
                if self.writes_checkpoints:
                    result[k] = full
            elif self.writes_checkpoints:
                result[k] = v

        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

        if not self.writes_checkpoints:
            return None

        return result
