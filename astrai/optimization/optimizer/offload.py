"""Bounded pinned-host optimizer state staging for GPU updates."""

from collections import deque
from typing import Dict, Optional

import torch
from torch.optim import SGD, AdamW

from astrai.optimization.optimizer.state import state_dtype
from astrai.parallel.executor.gradients import parameter_gradient


def _working_bytes(optimizer, group: Dict, master, state: Dict) -> int:
    """Bound live master/state storage, including lazy first-step moments."""
    size = master.numel() * 4
    stored = sum(
        value.numel() * (4 if value.is_floating_point() else value.element_size())
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    if not state:
        if isinstance(optimizer, AdamW):
            moments = 2
        elif isinstance(optimizer, SGD):
            moments = int(bool(group["momentum"]))
        else:
            moments = 1
        stored = size * moments
    return size + stored


def _upload(value: torch.Tensor, device, dtype=None) -> torch.Tensor:
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _store(value: torch.Tensor, destination: Optional[torch.Tensor] = None, dtype=None):
    if destination is None:
        destination = torch.empty_like(
            value, dtype=dtype or value.dtype, device="cpu", pin_memory=True
        )
    destination.copy_(value, non_blocking=True)
    value.record_stream(torch.cuda.current_stream(value.device))
    return destination


class CPUStateOffload:
    """Overlap upload, GPU update and return with at most two parameters live.

    Persistent masters/state are pinned CPU tensors. Each call drains its
    completion events before returning, so checkpoints can read CPU state
    immediately. Device allocations remain alive through their consumer
    streams; no model parameter is split and Muon still sees full matrices.
    """

    def __init__(
        self, model_params, master_params, optimizers, stored_dtype=torch.float32
    ):
        self.models = dict(zip(master_params, model_params))
        self.optimizers = optimizers
        self.stored_dtype = stored_dtype
        self.device = model_params[0].device
        self.upload_stream = torch.cuda.Stream(device=self.device)
        self.download_stream = torch.cuda.Stream(device=self.device)

    def pin_state(self) -> None:
        """Checkpoint loads create ordinary CPU tensors; restore pinned storage."""
        for optimizer in self.optimizers:
            for state in optimizer.state.values():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        value = value.to(
                            device="cpu",
                            dtype=state_dtype(name, value, self.stored_dtype),
                        )
                        state[name] = value if value.is_pinned() else value.pin_memory()

    @torch.no_grad()
    def step(self) -> None:
        compute = torch.cuda.current_stream(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        reusable = torch.cuda.memory_reserved(
            self.device
        ) - torch.cuda.memory_allocated(self.device)
        budget = min(total // 4, (free + reusable) // 2)
        pending = deque()
        live_bytes = 0

        def drain() -> None:
            nonlocal live_bytes
            complete, values, size = pending.popleft()
            complete.synchronize()
            live_bytes -= size

        try:
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    for master in group["params"]:
                        model = self.models[master]
                        gradient = parameter_gradient(model)
                        if gradient is None:
                            continue
                        host = master.data
                        state = dict(optimizer.state.get(master, {}))
                        size = _working_bytes(optimizer, group, master, state)
                        while pending and (
                            len(pending) >= 2 or live_bytes + size > budget
                        ):
                            drain()
                        try:
                            with torch.cuda.stream(self.upload_stream):
                                gpu_master = _upload(host, self.device, torch.float32)
                                gpu_state = {
                                    name: _upload(
                                        value,
                                        self.device,
                                        state_dtype(name, value, torch.float32),
                                    )
                                    if isinstance(value, torch.Tensor)
                                    else value
                                    for name, value in state.items()
                                }
                                uploaded = torch.cuda.Event()
                                uploaded.record()
                            compute.wait_event(uploaded)
                            gpu_master.record_stream(compute)
                            for value in gpu_state.values():
                                if isinstance(value, torch.Tensor):
                                    value.record_stream(compute)
                            master.data = gpu_master
                            optimizer.state[master] = gpu_state
                            master.grad = gradient.detach().float()
                            optimizer.step()
                            model.copy_(master.to(dtype=host.dtype))
                            computed = torch.cuda.Event()
                            computed.record(compute)
                            with torch.cuda.stream(self.download_stream):
                                self.download_stream.wait_event(computed)
                                _store(master.detach(), host)
                                state = {
                                    name: _store(
                                        value,
                                        state.get(name),
                                        state_dtype(name, value, self.stored_dtype),
                                    )
                                    if isinstance(value, torch.Tensor)
                                    else value
                                    for name, value in optimizer.state[master].items()
                                }
                                complete = torch.cuda.Event()
                                complete.record()
                            values = [gpu_master] + [
                                value
                                for value in gpu_state.values()
                                if isinstance(value, torch.Tensor)
                            ]
                            pending.append((complete, values, size))
                            live_bytes += size
                        finally:
                            master.grad = None
                            master.data = host
                            optimizer.state[master] = state
                        # The pending slot is the sole owner of staging
                        # tensors, so draining it really releases its budget.
                        del gpu_master, gpu_state, values
        finally:
            while pending:
                drain()
            # Also cover an exception before its completion event was queued.
            self.upload_stream.synchronize()
            self.download_stream.synchronize()
            compute.synchronize()
