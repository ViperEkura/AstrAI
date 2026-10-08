"""CUDA adapters for symmetric BLAS operations."""

from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Optional

from torch import Tensor

from astrai.extension.runtime.loader import get_module
from astrai.extension.runtime.loader import is_available as _available


def is_available() -> bool:
    return _available("symmetric")


def tiles(operation: str) -> List[Dict[str, Any]]:
    """Enumerate compiled GEMM recipes available to a symmetric operation."""
    return get_module("symmetric").tiles(operation)


@lru_cache(maxsize=256)
def _plan(
    operation: str,
    rows: int,
    cols: int,
    batch_size: int,
    input_layout: str,
    output_layout: str,
    addend: bool,
    device: int,
) -> Dict[str, Any]:
    return get_module("symmetric").plan(
        operation, rows, cols, batch_size, input_layout, output_layout, addend, device
    )


def plan(
    operation: str,
    rows: int,
    cols: int,
    batch_size: int = 1,
    input_layout: str = "row",
    output_layout: str = "row",
    addend: bool = False,
    device: int = 0,
) -> Dict[str, Any]:
    """Inspect a cached geometry decision without allocating or launching tensors.

    Device ordinals and matrix metadata fully determine a decision. Returned
    resource and candidate dictionaries are independent copies of cached data.
    An empty result means no compiled recipe is eligible.
    """
    return deepcopy(
        _plan(
            operation,
            rows,
            cols,
            batch_size,
            input_layout,
            output_layout,
            addend,
            device,
        )
    )


def syrk_out(
    x: Tensor,
    output: Tensor,
    *,
    addend: Optional[Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    tile: Optional[str] = None,
) -> None:
    """Write alpha * X X.T + beta * C; C must be fully symmetric."""
    get_module("symmetric").syrk_out(
        x,
        output,
        addend,
        alpha,
        beta,
        tile or ("64x64x32_W16x32_S2" if x.is_contiguous() else "64x64x64_W16x32_S2"),
    )


def symm_out(
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
    """Write alpha * S X + beta * C; S must be fully symmetric."""
    get_module("symmetric").symm_out(
        symmetric, x, output, addend, alpha, beta, tile or "64x64x32_W16x32_S2", raster
    )
