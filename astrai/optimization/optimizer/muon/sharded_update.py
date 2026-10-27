"""Muon updates for local and sharded matrix parameters."""

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, optim
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.optim._muon import (
    _adjust_lr,
    _single_tensor_muon,
    _zeropower_via_newtonschulz,
)

from astrai.extension.backend.newton_schulz import newton_schulz

MatrixTriple = Tuple[Tensor, Tensor, Tensor]


def _resolve_backend(use_ns_kernels: bool, ns_backend: Optional[str]) -> str:
    if ns_backend is None:
        return "auto" if use_ns_kernels else "torch"
    if ns_backend == "cuda":
        return "auto"
    if ns_backend not in ("torch", "auto"):
        raise ValueError("ns_backend must be torch, cuda or auto")
    return ns_backend


def _batched_sharded_matrices(
    triples: List[MatrixTriple], batch_size: int
) -> Iterator[List[MatrixTriple]]:
    buckets: Dict[Tuple[Any, ...], List[MatrixTriple]] = {}
    for param, grad, buf in triples:
        key = (
            tuple(grad.shape),
            grad.device,
            grad.dtype,
            buf.dtype,
            param.dtype,
            id(grad.device_mesh),
            tuple(grad.placements),
        )
        buckets.setdefault(key, []).append((param, grad, buf))
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            yield bucket[start : start + batch_size]


def _single_tensor_muon_reuse_buffers(
    params: List[Tensor],
    grads: List[Tensor],
    bufs: List[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
    has_complex: bool,
    backend: str = "torch",
    batch_ns_updates: bool = False,
    ns_batch_size: int = 4,
) -> None:
    """Apply local Muon updates with reusable buffers and optional batching."""
    if has_complex:
        raise ValueError("Complex parameters are not supported")
    lr = lr.item() if isinstance(lr, Tensor) else lr
    buckets: Dict[Tuple[Any, ...], List[MatrixTriple]] = {}
    for param, grad, buf in zip(params, grads, bufs):
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")
        key = (tuple(grad.shape), grad.device, grad.dtype, buf.dtype, param.dtype)
        buckets.setdefault(key, []).append((param, grad, buf))
    batch_size = ns_batch_size if batch_ns_updates else 1
    for bucket in buckets.values():
        for start in range(0, len(bucket), batch_size):
            chunk = bucket[start : start + batch_size]
            updates = []
            for _, grad, buf in chunk:
                buf.lerp_(grad, 1 - momentum)
                updates.append(grad.lerp(buf, momentum) if nesterov else buf)
            packed = torch.stack(updates) if len(chunk) > 1 else updates[0]
            output = newton_schulz(
                packed, ns_coefficients, ns_steps, eps, backend=backend
            )
            for index, (param, _, buf) in enumerate(chunk):
                update = output[index] if len(chunk) > 1 else output
                if len(chunk) > 1 and not nesterov and buf.dtype == torch.bfloat16:
                    buf.copy_(packed[index])
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)
                param.mul_(1 - lr * weight_decay)
                param.add_(update, alpha=-adjusted_lr)


def _validate_sharded_matrix(param: DTensor) -> None:
    if param.ndim != 2 or param.device_mesh.ndim != 1 or len(param.placements) != 1:
        raise ValueError("sharded Muon requires a 2D logical matrix on a 1D mesh")
    placement = param.placements[0]
    if not isinstance(placement, (Shard, Replicate)) or (
        isinstance(placement, Shard) and placement.dim not in (0, 1)
    ):
        raise ValueError("sharded Muon supports only Shard(0), Shard(1) or Replicate")


def step_sharded_muon(
    chunks: Iterable[List[MatrixTriple]],
    group: Mapping,
    backend: str,
    *,
    reuse_ns_buffers: bool = False,
) -> None:
    """Apply gathered Newton-Schulz updates and restore parameter shards."""
    lr = group["lr"].item() if isinstance(group["lr"], Tensor) else group["lr"]
    momentum = group["momentum"]
    nesterov = group["nesterov"]
    for chunk in chunks:
        updates = []
        for _, grad, buf in chunk:
            buf.lerp_(grad, 1 - momentum)
            updates.append(grad.lerp(buf, momentum) if nesterov else buf)
        packed = torch.stack(updates) if len(updates) > 1 else updates[0]
        full = packed.full_tensor()
        coefficients = group["ns_coefficients"]
        steps = group["ns_steps"]
        eps = group["eps"]
        if backend == "auto" or reuse_ns_buffers:
            ortho = newton_schulz(full, coefficients, steps, eps, backend=backend)
        elif full.ndim == 2:
            ortho = _zeropower_via_newtonschulz(full, coefficients, steps, eps)
        else:
            ortho = torch.stack(
                [
                    _zeropower_via_newtonschulz(item, coefficients, steps, eps)
                    for item in full.unbind(0)
                ]
            )
        # Gathering and packing break the non-Nesterov momentum alias. Restore
        # the normalized full input to every original shard before the next step.
        if not nesterov and full.dtype == torch.bfloat16:
            normalized = distribute_tensor(
                full, packed.device_mesh, packed.placements, src_data_rank=None
            )
            buffers = (
                torch.unbind(normalized, dim=0) if len(updates) > 1 else (normalized,)
            )
            for (_, _, buf), normalized_buf in zip(chunk, buffers):
                buf.copy_(normalized_buf)
        sharded = distribute_tensor(
            ortho, packed.device_mesh, packed.placements, src_data_rank=None
        )
        results = torch.unbind(sharded, dim=0) if len(updates) > 1 else (sharded,)
        for (param, _, _), update in zip(chunk, results):
            adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
            param.mul_(1 - lr * group["weight_decay"])
            param.add_(update, alpha=-adjusted_lr)


class ShardedMuon(optim.Muon):
    """Muon that gathers matrix DTensors for global Newton-Schulz matmuls."""

    def __init__(
        self,
        params,
        *,
        reuse_ns_buffers: bool = False,
        use_ns_kernels: bool = False,
        ns_batch_size: int = 4,
        ns_backend: Optional[str] = None,
        batch_ns_updates: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        if not isinstance(ns_batch_size, int) or ns_batch_size < 1:
            raise ValueError("ns_batch_size must be a positive integer")
        if batch_ns_updates is not None and not isinstance(batch_ns_updates, bool):
            raise ValueError("batch_ns_updates must be a boolean or None")
        self.reuse_ns_buffers = reuse_ns_buffers
        self.ns_batch_size = ns_batch_size
        self.ns_backend = _resolve_backend(use_ns_kernels, ns_backend)
        self.batch_ns_updates = (
            use_ns_kernels if batch_ns_updates is None else batch_ns_updates
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            bufs: List[Tensor] = []
            self._init_group(group, params, grads, bufs)
            plain, sharded = [], []
            for param, grad, buf in zip(params, grads, bufs):
                (sharded if isinstance(param, DTensor) else plain).append(
                    (param, grad, buf)
                )
            for param, _, _ in sharded:
                _validate_sharded_matrix(param)

            if plain:
                pp, gg, bb = (list(t) for t in zip(*plain))
                if (
                    self.reuse_ns_buffers
                    or self.ns_backend == "auto"
                    or self.batch_ns_updates
                ):
                    _single_tensor_muon_reuse_buffers(
                        pp,
                        gg,
                        bb,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_coefficients=group["ns_coefficients"],
                        ns_steps=group["ns_steps"],
                        eps=group["eps"],
                        adjust_lr_fn=group["adjust_lr_fn"],
                        has_complex=False,
                        backend=self.ns_backend,
                        batch_ns_updates=self.batch_ns_updates,
                        ns_batch_size=self.ns_batch_size,
                    )
                else:
                    _single_tensor_muon(
                        pp,
                        gg,
                        bb,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        nesterov=group["nesterov"],
                        ns_coefficients=group["ns_coefficients"],
                        ns_steps=group["ns_steps"],
                        eps=group["eps"],
                        adjust_lr_fn=group["adjust_lr_fn"],
                        has_complex=False,
                    )
            if sharded:
                chunks = _batched_sharded_matrices(
                    sharded, self.ns_batch_size if self.batch_ns_updates else 1
                )
                step_sharded_muon(
                    chunks,
                    group,
                    self.ns_backend,
                    reuse_ns_buffers=self.reuse_ns_buffers,
                )
        return loss
