"""Symmetric BLAS operations with measured CUDA dispatch and Torch fallback.

SYRK: out = alpha * X X.T + beta * C (C fully symmetric).
SYMM: out = alpha * S X + beta * C (S fully symmetric).
These out operations do not provide autograd and require separate output storage.
"""

import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor

from astrai.extension.kernel import symmetric as cuda
from astrai.extension.policy import symmetric as plan
from astrai.extension.runtime.dispatch import (
    ImplRecord,
    Spec,
    axis,
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
        torch.mm(x, x.T, out=output)
        if alpha != 1:
            output.mul_(alpha)
    else:
        torch.addmm(addend, x, x.T, alpha=alpha, beta=beta, out=output)


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
        torch.mm(symmetric, x, out=output)
        if alpha != 1:
            output.mul_(alpha)
    else:
        torch.addmm(addend, symmetric, x, alpha=alpha, beta=beta, out=output)


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
        "cuda": all(plan.supports(matrix) for matrix in operands),
        "measured": plan.probe(operation, x, addend=beta != 0).backend == "cuda",
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


def select(
    operation: str, *matrices: Tensor, backend: Optional[str] = None, **kwargs: Any
) -> Callable[..., None]:
    """Resolve once; callers can reuse the selected function inside a recurrence."""
    selected = resolve(operation, *matrices, explicit=backend, **kwargs).record.obj
    decision = plan.probe(operation, matrices[-2], addend=kwargs.get("beta", 0) != 0)
    options = {"tile": kwargs.get("tile") or decision.tile}
    if operation == "symm":
        options["raster"] = kwargs.get("raster", decision.raster)
    return partial(selected, **options)


def _validate(
    x: Tensor,
    output: Tensor,
    shape,
    operands,
    addend: Optional[Tensor],
    alpha: float,
    beta: float,
) -> None:
    if any(t.ndim != 2 for t in operands) or tuple(output.shape) != tuple(shape):
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
    if x.ndim != 2:
        raise ValueError("expected a matrix")
    _validate(x, output, (x.size(0), x.size(0)), (x,), addend, alpha, beta)
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
    if symmetric.shape != (x.size(0), x.size(0)):
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
