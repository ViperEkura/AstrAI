"""Distribute full-matrix Muon NS work across replicated DDP parameters."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.optim._muon import _adjust_lr

from astrai.extension.kernel.muon_ns import is_available, muon_ns


def _make_buckets(params, world_size: int, bucket_bytes: int):
    """Greedily balance NS FLOPs within bounded BF16 communication buckets."""
    buckets, current, elements = [], [], 0
    for index, param in enumerate(params):
        if current and (elements + param.numel()) * 2 > bucket_bytes:
            buckets.append(current)
            current, elements = [], 0
        current.append(index)
        elements += param.numel()
    if current:
        buckets.append(current)
    plans = []
    for indices in buckets:
        costs = [0] * world_size
        counts = [0] * world_size
        entries = []
        for index in sorted(
            indices,
            key=lambda i: params[i].numel() * min(params[i].shape),
            reverse=True,
        ):
            rows, cols = params[index].shape
            m, n = min(rows, cols), max(rows, cols)
            owner = min(range(world_size), key=lambda rank: (costs[rank], rank))
            entries.append((index, owner, counts[owner]))
            counts[owner] += params[index].numel()
            costs[owner] += 4 * m * m * n + 2 * m * m * m
        plans.append((entries, max(counts)))
    return plans


class ReplicatedMuon:
    """NS ownership only; momentum and checkpoint state remain replicated.

    The process group must contain DDP replicas with already synchronized full
    gradients. A matrix is never split for NS. All ranks execute matching
    bucket collectives, then apply exactly the same gathered BF16 updates.
    """

    def __init__(self, process_group, bucket_bytes: int = 256 * 2**20):
        if not dist.is_initialized():
            raise ValueError(
                "replicated Muon requires an initialized DDP process group"
            )
        if bucket_bytes < 2:
            raise ValueError("bucket_bytes must be at least two")
        self.process_group = process_group
        self.bucket_bytes = bucket_bytes
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self._plans = {}

    @torch.no_grad()
    def step(self, params, grads, bufs, group) -> bool:
        if self.world_size == 1:
            return False
        # Check participation before selecting any collective schedule. This
        # also handles unused gradients and empty groups without rank hangs.
        lr = (
            group["lr"].item() if isinstance(group["lr"], torch.Tensor) else group["lr"]
        )
        supported = (
            group["nesterov"]
            and is_available()
            and all(
                not isinstance(p, DTensor)
                and p.is_cuda
                and p.ndim == 2
                and p.dtype in (torch.bfloat16, torch.float32)
                for p in params
            )
        )
        metadata = (
            [
                (
                    tuple(p.shape),
                    str(p.dtype),
                    isinstance(p, DTensor),
                    p.grad is not None,
                )
                for p in group["params"]
            ],
            group["momentum"],
            group["nesterov"],
            tuple(group["ns_coefficients"]),
            group["ns_steps"],
            group["eps"],
            lr,
            group["weight_decay"],
            group["adjust_lr_fn"],
        )
        gathered = [None] * self.world_size
        dist.all_gather_object(
            gathered, (metadata, bool(supported)), group=self.process_group
        )
        if any(item[0] != metadata for item in gathered):
            raise RuntimeError(
                "replicated Muon requires matching parameter groups and gradient presence on all ranks"
            )
        if not params:
            return True
        if not all(item[1] for item in gathered):
            return False
        # Nesterov=False retains the existing path: BF16 scratch reuse can
        # mutate the momentum tensor and must not diverge on non-owner ranks.
        key = tuple((id(param), tuple(param.shape)) for param in params)
        if key not in self._plans:
            self._plans[key] = _make_buckets(params, self.world_size, self.bucket_bytes)
        momentum = group["momentum"]
        for entries, count in self._plans[key]:
            local = torch.zeros(count, dtype=torch.bfloat16, device=params[0].device)
            updates = torch.empty(
                count * self.world_size, dtype=local.dtype, device=local.device
            )
            for index, owner, offset in entries:
                param, grad, buf = params[index], grads[index], bufs[index]
                buf.lerp_(grad, 1 - momentum)
                if owner == self.rank:
                    update = muon_ns(
                        grad.lerp(buf, momentum),
                        group["ns_coefficients"],
                        group["ns_steps"],
                        group["eps"],
                    )
                    local[offset : offset + param.numel()].copy_(update.reshape(-1))
            dist.all_gather_into_tensor(updates, local, group=self.process_group)
            for index, owner, offset in entries:
                param = params[index]
                start = owner * count + offset
                update = updates[start : start + param.numel()].view(param.shape)
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
                param.mul_(1 - lr * group["weight_decay"])
                param.add_(update, alpha=-adjusted_lr)
        return True
