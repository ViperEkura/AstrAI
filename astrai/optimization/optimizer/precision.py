"""Persistent FP32 optimizer parameters for BF16 compute models."""

from typing import Dict

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import SGD, AdamW, Optimizer

from astrai.optimization.optimizer.muon.optimizer import MuonAdamW
from astrai.optimization.optimizer.muon.sharded_update import ShardedMuon
from astrai.optimization.optimizer.offload import CPUStateOffload
from astrai.parallel.executor.gradients import parameter_gradient


class FP32Optimizer(Optimizer):
    """Update FP32 masters and copy the rounded result to BF16 parameters.

    This owns update precision only. Gradients arrive after model-side
    accumulation/reduction; casting them here cannot recover lost precision.
    CPU offload stores masters and optimizer state in host memory between
    steps. Pinned host state uses a bounded upload/compute/return pipeline.
    Each step completes host copies before returning; forward/backward do
    not overlap with optimizer mutation.
    """

    supports_main_grad = True

    def __init__(self, optimizer: Optimizer, cpu_offload: bool = False):
        groups = optimizer.param_groups
        parameters = [p for group in groups for p in group["params"]]
        if any(isinstance(p, DTensor) for p in parameters):
            raise NotImplementedError(
                "FP32 master weights for DTensor/FSDP/TP parameters are not "
                "validated yet; use FP32 model parameters"
            )
        if any(p.dtype not in (torch.float32, torch.bfloat16) for p in parameters):
            raise ValueError("FP32Optimizer supports FP32 and BF16 parameters")
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
                dtype=torch.float32,
                copy=True,
            )
            if stage_on_gpu:
                value = value.pin_memory()
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
            CPUStateOffload(self._model_params, self._master_params, self._optimizers)
            if stage_on_gpu
            else None
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._offload is not None:
            self._offload.step()
            return loss
        for model, master in zip(self._model_params, self._master_params):
            gradient = parameter_gradient(model)
            master.grad = None if gradient is None else gradient.detach().float()
        result = self.optimizer.step()
        for model, master in zip(self._model_params, self._master_params):
            model.copy_(master)
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
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: Dict):
        if "master_params" not in state_dict or "optimizer" not in state_dict:
            raise ValueError(
                "Checkpoint has no FP32 master weights. Start a new run from "
                "its model weights with load_state_dict.optimizer=False; "
                "previously rounded updates cannot be recovered."
            )
        saved = state_dict["master_params"]
        if len(saved) != len(self._master_params) or any(
            value.dtype != torch.float32 or value.shape != master.shape
            for value, master in zip(saved, self._master_params)
        ):
            raise ValueError("Checkpoint FP32 master weights do not match the model")
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self._offload is not None:
            self._offload.pin_state()
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        for model, master, value in zip(self._model_params, self._master_params, saved):
            master.copy_(value)
            model.copy_(master)
