"""Composable update precision and configurable persistent optimizer storage."""

from typing import Dict

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import SGD, AdamW, Optimizer

from astrai.optimization.optimizer.muon.optimizer import MuonAdamW
from astrai.optimization.optimizer.muon.sharded_update import ShardedMuon
from astrai.optimization.optimizer.offload import CPUStateOffload
from astrai.optimization.optimizer.state import (
    ResidentOptimizerState,
    restore_state_precision,
)
from astrai.parallel.executor.gradients import parameter_gradient


class PrecisionOptimizer(Optimizer):
    """Compute updates in FP32 and persist the configured storage formats.

    This owns update precision only. Gradients arrive after model-side
    accumulation/reduction; casting them here cannot recover lost precision.
    CPU offload stores masters and optimizer state in host memory between
    steps. Pinned host state uses a bounded upload/compute/return pipeline.
    Each step completes host copies before returning; forward/backward do
    not overlap with optimizer mutation.
    """

    supports_main_grad = True

    def __init__(
        self,
        optimizer: Optimizer,
        cpu_offload: bool = False,
        master_dtype: torch.dtype = torch.float32,
        state_dtype: torch.dtype = torch.float32,
    ):
        if master_dtype not in (torch.float32, torch.bfloat16) or state_dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("Master/state storage supports float32 and bfloat16")
        self.master_dtype = master_dtype
        self.state_dtype = state_dtype
        groups = optimizer.param_groups
        parameters = [p for group in groups for p in group["params"]]
        if any(isinstance(p, DTensor) for p in parameters):
            raise NotImplementedError(
                "FP32 master weights for DTensor/FSDP/TP parameters are not "
                "validated yet; use FP32 model parameters"
            )
        if any(p.dtype not in (torch.float32, torch.bfloat16) for p in parameters):
            raise ValueError(
                "PrecisionOptimizer supports FP32 and BF16 compute parameters"
            )
        self._optimizers = (
            (optimizer.muon, optimizer.adamw)
            if isinstance(optimizer, MuonAdamW)
            else (optimizer,)
        )
        if any(child.state for child in self._optimizers):
            raise ValueError("Wrap the optimizer before its first update")
        if (
            cpu_offload
            and any(p.is_cuda for p in parameters)
            and any(
                not isinstance(child, (AdamW, SGD, ShardedMuon))
                for child in self._optimizers
            )
        ):
            raise NotImplementedError(
                "CPU state offload supports AdamW, SGD and Muon optimizers"
            )
        super().__init__(groups, optimizer.defaults)
        self.optimizer = optimizer
        self._model_params = parameters
        stage_on_gpu = cpu_offload and any(p.is_cuda for p in parameters)
        if stage_on_gpu and len({p.device for p in parameters}) != 1:
            raise ValueError("CPU state offload requires one CUDA model device")
        self._master_params = []
        for model in self._model_params:
            value = model.detach().to(
                device="cpu" if cpu_offload else model.device,
                dtype=master_dtype,
                copy=True,
            )
            if stage_on_gpu:
                value = value.pin_memory()
            if master_dtype == torch.bfloat16 and model.dtype == torch.float32:
                with torch.no_grad():
                    model.copy_(value)
            self._master_params.append(
                nn.Parameter(value, requires_grad=model.requires_grad)
            )
        masters = dict(zip(self._model_params, self._master_params))
        # Composite optimizers share these group objects with their children.
        for group in groups:
            group["params"] = [masters.get(p, p) for p in group["params"]]
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

        self._offload = (
            CPUStateOffload(
                self._model_params, self._master_params, self._optimizers, state_dtype
            )
            if stage_on_gpu
            else None
        )
        self._storage = self._offload or ResidentOptimizerState(
            optimizer,
            self._model_params,
            self._master_params,
            self._optimizers,
            state_dtype,
        )

    @property
    def master_parameters(self):
        """Compute/source pairs for serialization without exposing storage logic."""
        return tuple(zip(self._model_params, self._master_params))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        result = self._storage.step()
        return loss if closure is not None else result

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self._model_params:
            if getattr(parameter, "main_grad", None) is not None:
                if set_to_none:
                    parameter.main_grad = None
                else:
                    parameter.main_grad.zero_()
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    if parameter.grad.grad_fn is not None:
                        parameter.grad.detach_()
                    else:
                        parameter.grad.requires_grad_(False)
                    parameter.grad.zero_()

    def state_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer.state_dict(),
            "master_params": [p.detach() for p in self._master_params],
            "precision": {
                "master_dtype": str(self.master_dtype).removeprefix("torch."),
                "optimizer_state_dtype": str(self.state_dtype).removeprefix("torch."),
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: Dict):
        if "master_params" not in state_dict or "optimizer" not in state_dict:
            raise ValueError(
                "Checkpoint has no master weights. Start a new run from "
                "its model weights with load_state_dict.optimizer=False; "
                "previously rounded updates cannot be recovered."
            )
        saved = state_dict["master_params"]
        if (
            "precision" not in state_dict
            or "master_dtype" not in state_dict["precision"]
        ):
            raise ValueError("Checkpoint is missing explicit optimizer precision")
        saved_dtype = state_dict["precision"]["master_dtype"]
        if saved_dtype not in ("float32", "bfloat16"):
            raise ValueError("Checkpoint master precision is unsupported")
        if len(saved) != len(self._master_params) or any(
            value.dtype != getattr(torch, saved_dtype) or value.shape != master.shape
            for value, master in zip(saved, self._master_params)
        ):
            raise ValueError("Checkpoint master weights do not match the model")
        # Native loaders cast moments to parameter dtype. Present FP32 update
        # parameters while loading to avoid rounding FP32 state via BF16 masters.
        stored = [master.data for master in self._master_params]
        try:
            for master, value in zip(self._master_params, stored):
                master.data = value.float()
            self.optimizer.load_state_dict(state_dict["optimizer"])
        finally:
            for master, value in zip(self._master_params, stored):
                master.data = value
        restore_state_precision(self._optimizers, self.state_dtype)
        if self._offload is not None:
            self._offload.pin_state()
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        for model, master, value in zip(self._model_params, self._master_params, saved):
            master.copy_(value)
            model.copy_(master)
