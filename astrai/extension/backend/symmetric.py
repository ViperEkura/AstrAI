"""Symmetric BLAS operations with planned CUDA dispatch and Torch fallback.

SYRK: out = alpha * X X.T + beta * C (C fully symmetric).
SYMM: out = alpha * S X + beta * C (S fully symmetric).
These out operations do not provide autograd and require separate output storage.
"""

import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from astrai.extension.kernel import symmetric as cuda
from astrai.extension.policy import symmetric as plan
from astrai.extension.runtime.dispatch import (
    ImplRecord,
    Spec,
    axis,
    cache_token,
    register_family,
    resolve,
)


def _torch_syrk(
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    tile: Optional[str] = None,
) -> None:
    if beta == 0:
        (torch.mm if x.ndim == 2 else torch.bmm)(x, x.transpose(-2, -1), out=output)
        if alpha != 1:
            output.mul_(alpha)
    else:
        (torch.addmm if x.ndim == 2 else torch.baddbmm)(
            addend, x, x.transpose(-2, -1), alpha=alpha, beta=beta, out=output
        )


def _torch_symm(
    symmetric: Tensor,
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    tile: Optional[str] = None,
    raster: int = 1,
) -> None:
    if beta == 0:
        (torch.mm if x.ndim == 2 else torch.bmm)(symmetric, x, out=output)
        if alpha != 1:
            output.mul_(alpha)
    else:
        (torch.addmm if x.ndim == 2 else torch.baddbmm)(
            addend, symmetric, x, alpha=alpha, beta=beta, out=output
        )


def _axes(
    operation: str,
    *matrices: Tensor,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    tile: Optional[str] = None,
    raster: int = 1,
) -> Dict[str, Any]:
    x = matrices[-2]
    operands = matrices + ((addend,) if beta != 0 and addend is not None else ())
    return {
        "cuda": all(plan.supports(matrix) for matrix in operands)
        and (
            tile is None
            or any(
                candidate["name"] == tile
                and plan.layout(x) in candidate["input_layouts"]
                for candidate in cuda.tiles(operation)
            )
        ),
        # Retain the registry axis name for existing dispatch overrides.
        "measured": plan.probe(
            operation, x, output=matrices[-1], addend=beta != 0
        ).backend
        == "cuda",
    }


def _records(operation: str) -> List[ImplRecord]:
    cuda_op = cuda.syrk_out if operation == "syrk" else cuda.symm_out
    torch_op = _torch_syrk if operation == "syrk" else _torch_symm
    return [
        ImplRecord(
            operation,
            "measured",
            cuda_op,
            axis("cuda").truthy() & axis("measured").truthy(),
            priority=0,
        ),
        # Explicit selection exposes every legal tile/shape for the sweep.
        ImplRecord(
            operation,
            "cuda",
            cuda_op,
            axis("cuda").truthy(),
            priority=1,
            faithful=False,
        ),
        ImplRecord(operation, "torch", torch_op, Spec.always(), priority=99),
    ]


for _operation in ("syrk", "symm"):
    register_family(
        _operation,
        partial(_axes, _operation),
        partial(_records, _operation),
        lambda operation=_operation: _records(operation)[-1],
    )


_selection_cache: Dict[Tuple[Any, ...], Callable[..., None]] = {}


def _selection_key(operation: str, matrices, kwargs) -> Optional[Tuple[Any, ...]]:
    token = cache_token(operation)
    if token is None:
        return None
    addend = kwargs.get("addend")
    beta = kwargs.get("beta", 0) != 0
    operands = matrices + ((addend,) if beta and addend is not None else ())
    metadata = tuple(
        (
            tuple(x.shape),
            x.stride(),
            x.dtype,
            x.device,
            x.requires_grad,
            x.data_ptr() % 16,
        )
        for x in operands
    )
    return (
        token,
        plan.revision(),
        operation,
        metadata,
        beta,
        kwargs.get("tile"),
        kwargs.get("raster"),
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.are_deterministic_algorithms_enabled(),
    )


def select(
    operation: str, *matrices: Tensor, backend: Optional[str] = None, **kwargs: Any
) -> Callable[..., None]:
    """Resolve once; callers can reuse the selected function inside a recurrence."""
    if operation not in ("syrk", "symm"):
        raise ValueError("operation must be syrk or symm")
    if backend == "torch":
        return _torch_syrk if operation == "syrk" else _torch_symm
    key = _selection_key(operation, matrices, kwargs) if backend is None else None
    cached = _selection_cache.get(key) if key is not None else None
    if cached is not None:
        return cached
    selected = resolve(operation, *matrices, explicit=backend, **kwargs).record.obj
    decision = plan.probe(
        operation, matrices[-2], output=matrices[-1], addend=kwargs.get("beta", 0) != 0
    )
    options = {"tile": kwargs.get("tile") or decision.tile}
    if operation == "symm":
        options["raster"] = kwargs.get("raster", decision.raster)
    function = partial(selected, **options)
    if key is not None:
        if len(_selection_cache) >= 128:
            _selection_cache.clear()
        _selection_cache[key] = function
    return function


def _validate(
    x: Tensor,
    output: Tensor,
    shape,
    operands,
    addend: Optional[Tensor],
    alpha: float,
    beta: float,
) -> None:
    if any(t.ndim != x.ndim for t in operands) or tuple(output.shape) != tuple(shape):
        raise ValueError("matrix shape mismatch")
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("coefficients must be finite")
    if beta != 0 and addend is None:
        raise ValueError("nonzero beta requires an addend")
    if beta != 0:
        if addend.shape != output.shape:
            raise ValueError("addend shape mismatch")
        operands = (*operands, addend)
    for tensor in operands:
        if tensor.device != output.device or tensor.dtype != output.dtype:
            raise ValueError("matrix device or dtype mismatch")
        if torch._C._is_alias_of(output, tensor):
            raise ValueError("output must not alias inputs")


def syrk_out(
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: Optional[str] = None,
    tile: Optional[str] = None,
) -> None:
    """Compute alpha * X X.T + beta * C into output."""
    if x.ndim not in (2, 3):
        raise ValueError("expected a matrix or matrix batch")
    _validate(
        x, output, (*x.shape[:-2], x.size(-2), x.size(-2)), (x,), addend, alpha, beta
    )
    select(
        "syrk",
        x,
        output,
        backend=backend,
        addend=addend,
        alpha=alpha,
        beta=beta,
        tile=tile,
    )(x, output, addend=addend, alpha=alpha, beta=beta)


def symm_out(
    symmetric: Tensor,
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: Optional[str] = None,
    tile: Optional[str] = None,
    raster: Optional[int] = None,
) -> None:
    """Compute alpha * S X + beta * C into output."""
    _validate(x, output, x.shape, (symmetric, x), addend, alpha, beta)
    if x.ndim not in (2, 3) or symmetric.shape != (
        *x.shape[:-2],
        x.size(-2),
        x.size(-2),
    ):
        raise ValueError("symmetric matrix shape mismatch")
    options = {"tile": tile}
    if raster is not None:
        options["raster"] = raster
    select(
        "symm",
        symmetric,
        x,
        output,
        backend=backend,
        addend=addend,
        alpha=alpha,
        beta=beta,
        **options,
    )(symmetric, x, output, addend=addend, alpha=alpha, beta=beta)
