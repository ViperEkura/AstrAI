"""CUDA adapters for Newton-Schulz matrix operations and iteration."""

from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor

from astrai.extension.runtime.loader import get_module
from astrai.extension.runtime.loader import is_available as _available


def is_available() -> bool:
    return _available("newton_schulz")


def tiles(operation: str) -> List[Dict[str, Any]]:
    """Enumerate compiled GEMM recipes available to a symmetric operation."""
    return get_module("newton_schulz").tiles(operation)


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
    mode: str,
) -> Dict[str, Any]:
    args = (
        operation,
        rows,
        cols,
        batch_size,
        input_layout,
        output_layout,
        addend,
        device,
    )
    module = get_module("newton_schulz")
    return module.plan(*args) if mode == "model" else module.plan(*args, mode)


def plan(
    operation: str,
    rows: int,
    cols: int,
    batch_size: int = 1,
    input_layout: str = "row",
    output_layout: str = "row",
    addend: bool = False,
    device: int = 0,
    mode: str = "model",
) -> Dict[str, Any]:
    """Inspect a cached native tile decision without allocating or launching tensors.

    Device ordinals and matrix metadata fully determine a decision. Returned
    Resource and candidate dictionaries are independent copies of cached data.
    Mode is "model" by default; "geometry" reproduces the previous ranking.
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
            mode,
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
    options = {} if tile is None else {"tile": tile}
    get_module("newton_schulz").syrk_out(x, output, addend, alpha, beta, **options)


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
    options = {"raster": raster}
    if tile is not None:
        options["tile"] = tile
    get_module("newton_schulz").symm_out(
        symmetric, x, output, addend, alpha, beta, **options
    )


def iterate(
    x: Tensor,
    gram: Tensor,
    polynomial: Tensor,
    work: Tensor,
    spare: Tensor,
    final: Optional[Tensor],
    steps: int,
    a: float,
    b: float,
    c: float,
    choices: Tuple[Tuple[bool, str, int], ...],
) -> Tensor:
    """Launch the selected SYRK and SYMM stages of an NS recurrence."""
    return get_module("newton_schulz").iterate(
        x, gram, polynomial, work, spare, final, steps, a, b, c, choices
    )
