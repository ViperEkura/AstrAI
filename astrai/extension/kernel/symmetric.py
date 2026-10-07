"""CUDA adapters for symmetric BLAS operations."""

from typing import Any, Dict, List, Optional

from torch import Tensor

from astrai.extension.runtime.loader import get_module
from astrai.extension.runtime.loader import is_available as _available


def is_available() -> bool:
    return _available("symmetric")


def tiles(operation: str) -> List[Dict[str, Any]]:
    """Enumerate compiled GEMM recipes available to a symmetric operation."""
    return get_module("symmetric").tiles(operation)


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
    get_module("symmetric").syrk_out(x, output, addend, alpha, beta, tile or "wmma64")


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
