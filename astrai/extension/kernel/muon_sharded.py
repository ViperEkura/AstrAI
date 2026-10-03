"""Experimental row-sharded Muon update with fused local CUDA kernels."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from astrai.extension.loader import get_module
from astrai.extension.loader import is_available as _kernel_available

_MAX_GRAM_BUCKET_BYTES = 64 * 2**20
ShardItem = tuple[DTensor, DTensor, DTensor, float]


def is_available() -> bool:
    """Return whether the local sharded Muon CUDA kernels are loadable."""
    return _kernel_available("muon_ns")


def supports(param: DTensor, grad: DTensor, momentum_buffer: DTensor) -> bool:
    """Select the safe 1-D mesh, dim-0 sharded tall-matrix case."""
    return (
        is_available()
        and param.ndim == 2
        and param.shape[0] >= 4 * param.shape[1]
        and param.shape[0] >= param.device_mesh.size(0)
        and 2 * param.shape[1] ** 2 <= _MAX_GRAM_BUCKET_BYTES
        and param.device.type == "cuda"
        and param.dtype in (torch.bfloat16, torch.float32)
        and param.dtype == grad.dtype == momentum_buffer.dtype
        and param.device_mesh.ndim == 1
        and param.placements == grad.placements == momentum_buffer.placements
        and len(param.placements) == 1
        and isinstance(param.placements[0], Shard)
        and param.placements[0].dim == 0
        and all(
            tensor.to_local().is_contiguous()
            for tensor in (param, grad, momentum_buffer)
        )
    )


def step_(
    param: DTensor,
    grad: DTensor,
    momentum_buffer: DTensor,
    *,
    lr: float,
    adjusted_lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> None:
    """Apply one Muon step without materializing the global matrix."""
    steps_(
        [(param, grad, momentum_buffer, adjusted_lr)],
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )


def steps_(
    items: list[ShardItem],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> None:
    """Pack several independent matrix updates into bounded NCCL buckets."""
    if not items:
        return
    if ns_steps < 1 or ns_steps >= 100 or len(ns_coefficients) != 3:
        raise ValueError("invalid Muon Newton-Schulz configuration")
    mesh = items[0][0].device_mesh
    for param, grad, momentum_buffer, _ in items:
        if not supports(param, grad, momentum_buffer) or param.device_mesh != mesh:
            raise ValueError("unsupported sharded Muon layout, dtype or mesh")

    bucket = []
    bucket_bytes = 0
    for item in items:
        gram_bytes = 2 * item[0].shape[1] ** 2
        if bucket and bucket_bytes + gram_bytes > _MAX_GRAM_BUCKET_BYTES:
            _step_bucket_(
                bucket,
                lr,
                weight_decay,
                momentum,
                nesterov,
                ns_coefficients,
                ns_steps,
                eps,
            )
            bucket = []
            bucket_bytes = 0
        bucket.append(item)
        bucket_bytes += gram_bytes
    if bucket:
        _step_bucket_(
            bucket,
            lr,
            weight_decay,
            momentum,
            nesterov,
            ns_coefficients,
            ns_steps,
            eps,
        )


def _step_bucket_(
    items: list[ShardItem],
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> None:
    module = get_module("muon_ns")
    prepared = []
    for param, grad, momentum_buffer, adjusted_lr in items:
        local_param = param.to_local()
        update, partial_squares = module.prepare(
            grad.to_local(), momentum_buffer.to_local(), momentum, nesterov
        )
        prepared.append((local_param, update, partial_squares, adjusted_lr))

    norm_squared = module.reduce_partials([entry[2] for entry in prepared])
    process_group = items[0][0].device_mesh.get_group(0)
    dist.all_reduce(norm_squared, group=process_group)

    total_gram_elements = sum(entry[1].size(1) ** 2 for entry in prepared)
    gram_bucket = torch.empty(
        (total_gram_elements,), device=norm_squared.device, dtype=torch.bfloat16
    )
    states = []
    offset = 0
    for index, (local_param, update, _, adjusted_lr) in enumerate(prepared):
        module.normalize_(update, norm_squared[index], eps)
        x = update.T
        small_dim = x.size(0)
        gram = gram_bucket.narrow(0, offset, small_dim**2).view(small_dim, small_dim)
        offset += small_dim**2
        states.append(
            [
                local_param,
                adjusted_lr,
                x,
                gram,
                torch.empty(
                    (small_dim, small_dim), device=x.device, dtype=torch.bfloat16
                ),
                torch.empty_like(x),
            ]
        )

    a, b, c = ns_coefficients
    for _ in range(ns_steps):
        for state in states:
            x, gram = state[2], state[3]
            module.gram_(x, gram)
        dist.all_reduce(gram_bucket, group=process_group)
        for state in states:
            x, gram, polynomial, next_x = state[2:]
            module.ns_update_(x, gram, polynomial, next_x, a, b, c)
            state[2], state[5] = next_x, x

    for local_param, adjusted_lr, x, _, _, _ in states:
        module.finish_(local_param, x.T.contiguous(), lr, weight_decay, adjusted_lr)


__all__ = ["is_available", "step_", "steps_", "supports"]
