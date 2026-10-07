"""Shared host-memory policy snapshots and CUDA transfer boundaries."""

import ctypes
import logging
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import torch
from torch import nn

from astrai.parallel.executor import strip_compile_prefix

logger = logging.getLogger(__name__)


def _register_shared(shm, device) -> bool:
    """Pin this process's mapping of the shared weight buffer when possible."""
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
    try:
        with torch.cuda.device(device):
            status = torch.cuda.cudart().cudaHostRegister(pointer, len(shm.buf), 0)
    except (OSError, RuntimeError) as exc:
        logger.warning("CUDA could not register rollout shared memory: %s", exc)
        return False
    if status.value:
        logger.warning("CUDA could not register rollout shared memory: %s", status)
        return False
    return True


def _unregister_shared(shm, device) -> None:
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
    try:
        with torch.cuda.device(device):
            status = torch.cuda.cudart().cudaHostUnregister(pointer)
        if status.value:
            logger.warning(
                "CUDA could not unregister rollout shared memory: %s", status
            )
    except (OSError, RuntimeError) as exc:
        logger.warning("CUDA could not unregister rollout shared memory: %s", exc)


def _shared_views(shm, layout):
    views = {}
    for name, shape, dtype_name, offset, nbytes in layout:
        dtype = getattr(torch, dtype_name)
        if nbytes:
            views[name] = torch.frombuffer(
                shm.buf, dtype=dtype, count=nbytes // dtype.itemsize, offset=offset
            ).reshape(shape)
        else:
            views[name] = torch.empty(shape, dtype=dtype)
    return views


class SharedWeightBuffer:
    """One GPU-to-host snapshot and one shared-memory copy per policy version."""

    def __init__(self, source: nn.Module):
        self._source = strip_compile_prefix(dict(source.state_dict(keep_vars=True)))
        self.layout = []
        offset = 0
        for name, tensor in self._source.items():
            offset = (offset + 63) & ~63
            nbytes = tensor.numel() * tensor.element_size()
            self.layout.append(
                (
                    name,
                    tuple(tensor.shape),
                    str(tensor.dtype).split(".")[-1],
                    offset,
                    nbytes,
                )
            )
            offset += nbytes
        self._shm = SharedMemory(create=True, size=max(offset, 1))
        self._cuda_devices = {
            tensor.device for tensor in self._source.values() if tensor.is_cuda
        }
        self._cuda_device = next(iter(self._cuda_devices), None)
        self.registered = False
        try:
            if self._cuda_device is not None:
                self.registered = _register_shared(self._shm, self._cuda_device)
            self._views = _shared_views(self._shm, self.layout)
            self._staged = (
                {
                    name: torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                    for name, tensor in self._source.items()
                }
                if self._cuda_device is not None and not self.registered
                else {}
            )
        except BaseException:
            if self.registered:
                _unregister_shared(self._shm, self._cuda_device)
            self._shm.close()
            self._shm.unlink()
            raise
        self.version: Optional[int] = None
        self._pending = set()
        self.snapshot_seconds = 0.0
        self.fanout_seconds = 0.0

    @property
    def name(self) -> str:
        return self._shm.name

    def snapshot(self, version: int) -> None:
        if self._pending:
            raise RuntimeError("previous weight transfer has not been acknowledged")
        started = time.perf_counter()
        with torch.no_grad():
            for name, source in self._source.items():
                destination = (
                    self._views[name]
                    if self.registered or not source.is_cuda
                    else self._staged[name]
                )
                destination.copy_(source, non_blocking=source.device.type == "cuda")
            for device in self._cuda_devices:
                torch.cuda.current_stream(device).synchronize()
            if not self.registered and self._staged:
                for name, source in self._source.items():
                    if source.is_cuda:
                        self._views[name].copy_(self._staged[name])
        self.version = version
        self.snapshot_seconds += time.perf_counter() - started

    def begin_fanout(self, indices: list[int]) -> None:
        if self._pending:
            raise RuntimeError("previous weight transfer is still in flight")
        self._pending = set(indices)

    def acknowledge(self, index: int) -> None:
        if index not in self._pending:
            raise RuntimeError("unexpected weight transfer acknowledgement")
        self._pending.remove(index)

    def close(self) -> None:
        self._views.clear()
        if self.registered:
            _unregister_shared(self._shm, self._cuda_device)
        self._shm.close()
        self._shm.unlink()


def _copy_shared_weights(views, layout, target, pinned, device):
    if set(target) != {entry[0] for entry in layout}:
        raise RuntimeError("rollout replica state keys differ from learner")
    with torch.no_grad(), torch.cuda.device(device):
        for name, shape, dtype_name, _, _ in layout:
            dest = target[name]
            if tuple(dest.shape) != shape:
                raise RuntimeError(f"rollout replica shape differs for {name}")
            shared = views[name]
            if shared.dtype != getattr(torch, dtype_name):
                raise RuntimeError(f"rollout shared dtype differs for {name}")
            host = shared
            if pinned is not None:
                host = pinned[name]
                host.copy_(shared)
            dest.copy_(host, non_blocking=True)
        # Both the ACK and the next shared-memory publication require H2D
        # completion. Registered mappings and fallback pinned buffers persist.
        torch.cuda.current_stream(device).synchronize()
