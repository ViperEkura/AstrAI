"""Persistent FP32 optimizer parameters for BF16 compute models."""

from typing import Dict

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


class FP32Optimizer(Optimizer):
    """Update FP32 masters and copy the rounded result to BF16 parameters.

    This owns update precision only. Gradients arrive after model-side
    accumulation/reduction; casting them here cannot recover lost precision.
    CPU offload moves masters, optimizer state and updates to host memory;
    transfers are synchronous and are not overlapped with forward/backward.
    """

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
        if optimizer.state:
            raise ValueError("Wrap the optimizer before its first update")
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

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for model, master in zip(self._model_params, self._master_params):
            if model.grad is None:
                master.grad = None
            elif self._transfer_buffer is not None and model.is_cuda:
                buffer = self._buffer(model.grad)
                buffer.copy_(model.grad)
                master.grad = buffer.to(dtype=torch.float32, copy=True)
            else:
                master.grad = model.grad.detach().to(
                    device=master.device, dtype=torch.float32
                )
        result = self.optimizer.step()
        for model, master in zip(self._model_params, self._master_params):
            if self._transfer_buffer is not None and model.is_cuda:
                buffer = self._buffer(model)
                buffer.copy_(master)
                model.copy_(buffer)
            else:
                model.copy_(master)
        return loss if closure is not None else result

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self._model_params:
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
