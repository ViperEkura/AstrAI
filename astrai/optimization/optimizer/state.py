"""Persistent optimizer-state formats and resident GPU/CPU updates."""

import torch

from astrai.parallel.executor.gradients import parameter_gradient


def state_dtype(name, value, dtype):
    """Counters retain their range independently of moment storage precision."""
    if not value.is_floating_point():
        return value.dtype
    return torch.float32 if name == "step" else dtype


def cast_state(state, dtype):
    return {
        name: value.to(dtype=state_dtype(name, value, dtype))
        if isinstance(value, torch.Tensor)
        else value
        for name, value in state.items()
    }


def restore_state_precision(optimizers, dtype):
    for optimizer in optimizers:
        for parameter, state in optimizer.state.items():
            optimizer.state[parameter] = cast_state(state, dtype)


class ResidentOptimizerState:
    """Batch native updates in FP32 and persist explicitly selected formats."""

    def __init__(self, optimizer, models, masters, children, stored_dtype):
        self.optimizer = optimizer
        self.models = models
        self.masters = masters
        self.children = children
        self.stored_dtype = stored_dtype
        self.convert = stored_dtype != torch.float32 or any(
            master.dtype != torch.float32 for master in masters
        )

    @torch.no_grad()
    def step(self):
        if not self.convert:
            for model, master in zip(self.models, self.masters):
                gradient = parameter_gradient(model)
                master.grad = None if gradient is None else gradient.detach().float()
            result = self.optimizer.step()
            for model, master in zip(self.models, self.masters):
                model.copy_(master)
            return result

        stored = [master.data for master in self.masters]
        try:
            for model, master, value in zip(self.models, self.masters, stored):
                gradient = parameter_gradient(model)
                master.data = value.float()
                master.grad = None if gradient is None else gradient.detach().float()
            for child in self.children:
                for master in child.state:
                    if master.grad is not None:
                        child.state[master] = cast_state(
                            child.state[master], torch.float32
                        )
            result = self.optimizer.step()
            for model, master, value in zip(self.models, self.masters, stored):
                if value.data_ptr() != master.data_ptr():
                    value.copy_(master)
                # Publish from persistent storage, including its quantization.
                model.copy_(value)
            restore_state_precision(self.children, self.stored_dtype)
            return result
        finally:
            for master, value in zip(self.masters, stored):
                if master.dtype != value.dtype:
                    master.grad = None
                master.data = value
