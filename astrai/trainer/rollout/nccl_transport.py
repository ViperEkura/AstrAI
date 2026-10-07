"""Versioned NCCL broadcast of learner weights to rollout processes."""

from datetime import timedelta
from time import perf_counter
from typing import List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from torch import nn

from astrai.parallel.executor import strip_compile_prefix


class NCCLWeightChannel:
    """Broadcast model state in deterministic order without a host weight copy.

    Rank zero holds the learner model. Each other rank holds its own frozen
    replica. The control pipe selects a version; this channel moves only CUDA
    tensors, and the pipe ACK follows CUDA completion on the receiving rank.
    """

    def __init__(
        self,
        model: nn.Module,
        policy_version: int,
        expected_layout: Optional[List[Tuple[str, Tuple[int, ...], str]]] = None,
    ):
        state = strip_compile_prefix(dict(model.state_dict(keep_vars=True)))
        if not state:
            raise ValueError("NCCL rollout requires a nonempty model state")
        self.layout = [
            (name, tuple(tensor.shape), str(tensor.dtype).split(".")[-1])
            for name, tensor in state.items()
        ]
        if expected_layout is not None:
            if set(state) != {name for name, _, _ in expected_layout}:
                raise RuntimeError("rollout replica state keys differ from learner")
            for name, shape, dtype in expected_layout:
                tensor = state[name]
                if (
                    tuple(tensor.shape) != shape
                    or str(tensor.dtype).split(".")[-1] != dtype
                ):
                    raise RuntimeError(
                        f"rollout replica state layout differs for {name}"
                    )
            self.layout = expected_layout
        self._tensors = [state[name] for name, _, _ in self.layout]
        devices = {tensor.device for tensor in self._tensors}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("NCCL rollout state must reside on one CUDA device")
        self.device = next(iter(devices))
        self.version = policy_version
        self.broadcast_seconds = 0.0
        self.fanout_seconds = 0.0
        self._pending: Set[int] = set()
        self._store = None
        self._group = None
        self._connected = False
        self._rank: Optional[int] = None

    def prepare_rendezvous(self, world_size: int, timeout_s: float) -> int:
        """Keep rank zero's rendezvous server alive through worker startup."""
        self._store = dist.TCPStore(
            "127.0.0.1",
            0,
            world_size,
            True,
            timedelta(seconds=timeout_s),
            wait_for_workers=False,
        )
        return self._store.port

    def connect(self, rank: int, world_size: int, port: int, timeout_s: float) -> None:
        if self._connected:
            return
        torch.cuda.set_device(self.device)
        store = self._store
        if rank != 0:
            store = dist.TCPStore(
                "127.0.0.1",
                port,
                world_size,
                False,
                timedelta(seconds=timeout_s),
            )
        if store is None:
            raise RuntimeError("NCCL rendezvous store was not prepared")
        self._group = dist.ProcessGroupNCCL(
            store, rank, world_size, timedelta(seconds=timeout_s)
        )
        self._store = store
        self._rank = rank
        self._connected = True

    def broadcast(self) -> None:
        if not self._connected:
            raise RuntimeError("NCCL weight channel is not connected")
        started = perf_counter()
        pending = []
        with torch.no_grad(), torch.cuda.device(self.device):
            for tensor in self._tensors:
                if tensor.numel() == 0:
                    continue
                wire = tensor.detach()
                if not wire.is_contiguous():
                    wire = (
                        wire.contiguous()
                        if self._rank == 0
                        else torch.empty_like(
                            wire, memory_format=torch.contiguous_format
                        )
                    )
                pending.append((self._group.broadcast(wire, 0), wire, tensor))
            for work, _, _ in pending:
                work.wait()
            if self._rank != 0:
                for _, wire, tensor in pending:
                    if wire.data_ptr() != tensor.data_ptr():
                        tensor.copy_(wire)
            torch.cuda.synchronize(self.device)
        self.broadcast_seconds += perf_counter() - started

    def mark_committed(self, version: int) -> None:
        if self._pending:
            raise RuntimeError("previous weight broadcast has not been acknowledged")
        if version <= self.version:
            raise ValueError("policy version must advance")
        self.version = version

    def begin_fanout(self, indices: List[int]) -> None:
        if self._pending:
            raise RuntimeError("previous weight broadcast is still in flight")
        self._pending = set(indices)

    def acknowledge(self, index: int) -> None:
        if index not in self._pending:
            raise RuntimeError("unexpected weight broadcast acknowledgement")
        self._pending.remove(index)

    def close(self, force: bool = False) -> None:
        if self._group is not None:
            if force:
                self._group.abort()
            else:
                self._group.shutdown()
        self._group = None
        self._connected = False
        self._store = None
