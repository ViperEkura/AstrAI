"""Persistent FP32 optimizer parameters for BF16 compute models."""

from typing import Dict

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import SGD, AdamW, Optimizer

from astrai.optimization.optimizer.muon.optimizer import MuonAdamW
from astrai.optimization.optimizer.muon.sharded_update import ShardedMuon
from astrai.parallel.executor.gradients import parameter_gradient


class FP32Optimizer(Optimizer):
    """Update FP32 masters and copy the rounded result to BF16 parameters.

    This owns update precision only. Gradients arrive after model-side
    accumulation/reduction; casting them here cannot recover lost precision.
    CPU offload stores masters and optimizer state in host memory between
    steps. Parameters are staged one at a time for GPU updates; transfers
    are synchronous and are not overlapped with forward/backward.
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
        self._master_params = [
            nn.Parameter(
                p.detach().to(
                    device="cpu" if cpu_offload else p.device,
                    dtype=torch.float32,
                    copy=True,
                ),
                requires_grad=p.requires_grad,
            )
            for p in self._model_params
        ]
        if (
            cpu_offload
            and any(p.is_cuda for p in parameters)
            and any(p.device != parameters[0].device for p in parameters)
        ):
            raise ValueError("CPU state offload requires one CUDA model device")
        self._transfer_buffer = (
            torch.empty(
                4 * max(p.numel() for p in parameters),
                dtype=torch.uint8,
                pin_memory=True,
            )
            if cpu_offload and any(p.is_cuda for p in parameters)
            else None
        )
        masters = dict(zip(self._model_params, self._master_params))
        # Composite optimizers share these group objects with their children.
        for group in groups:
            group["params"] = [masters.get(p, p) for p in group["params"]]
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

    def _buffer(self, model: nn.Parameter):
        return (
            self._transfer_buffer[: model.numel() * model.element_size()]
            .view(model.dtype)
            .view(model.shape)
        )

    def _upload(self, value: torch.Tensor, device):
        buffer = self._buffer(value)
        buffer.copy_(value)
        return buffer.to(device)

    def _store(self, value: torch.Tensor, destination=None):
        buffer = self._buffer(value)
        buffer.copy_(value)
        return buffer.clone() if destination is None else destination.copy_(buffer)

    def _offloaded_step(self):
        models = dict(zip(self._master_params, self._model_params))
        for optimizer in self._optimizers:
            for group in optimizer.param_groups:
                for master in group["params"]:
                    model = models[master]
                    gradient = parameter_gradient(model)
                    if gradient is None:
                        continue
                    host = master.data
                    state = dict(optimizer.state.get(master, {}))
                    try:
                        master.data = self._upload(host, model.device)
                        optimizer.state[master] = {
                            name: self._upload(value, model.device)
                            if isinstance(value, torch.Tensor)
                            else value
                            for name, value in state.items()
                        }
                        master.grad = gradient.detach().float()
                        optimizer.step()
                        model.copy_(master)
                        self._store(master.detach(), host)
                        state = {
                            name: self._store(value, state.get(name))
                            if isinstance(value, torch.Tensor)
                            else value
                            for name, value in optimizer.state[master].items()
                        }
                    finally:
                        master.grad = None
                        master.data = host
                        optimizer.state[master] = state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._transfer_buffer is not None:
            self._offloaded_step()
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
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        for model, master, value in zip(self._model_params, self._master_params, saved):
            master.copy_(value)
            model.copy_(master)
