"""Optional measured plans and native Newton-Schulz matrix planning."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
from torch import Tensor

from astrai.extension.kernel.newton_schulz import is_available, tiles
from astrai.extension.kernel.newton_schulz import plan as heuristic_plan


@dataclass(frozen=True)
class Plan:
    backend: str = "torch"
    tile: Optional[str] = None
    raster: int = 1


Key = Tuple[str, int, int, int, bool, str, str, int]


def _key(row: Mapping[str, Any]) -> Key:
    return (
        row["operation"],
        int(row["cc"]),
        int(row["rows"]),
        int(row["cols"]),
        bool(row.get("addend", False)),
        row.get("input_layout", "row"),
        row.get("output_layout", "row"),
        int(row.get("batch_size", 1)),
    )


_revision = 0
_heuristic = True
_rows: List[Dict[str, Any]] = []
_plans: Dict[Key, Plan] = {}


def configure(
    rows: Optional[Iterable[Mapping[str, Any]]] = None,
    *,
    heuristic: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Replace measured rows atomically; no argument reads the current table.

    Every row has operation, cc, rows, cols and backend. Optional fields include
    addend, tile, raster, input/output layout and batch_size (default 1).
    Replacing rows defaults to table-only dispatch; heuristic=True enables
    native model fallback for missing keys. A matching Torch row always wins.
    With no arguments, the current table and dispatch mode are unchanged.
    Duplicate keys are errors, and failed validation leaves both modes intact.
    """
    global _rows, _plans, _revision, _heuristic
    if heuristic is not None and not isinstance(heuristic, bool):
        raise ValueError("heuristic must be bool")
    if rows is not None:
        copied = [dict(row) for row in rows]
        plans: Dict[Key, Plan] = {}
        vocabulary: Dict[str, Dict[str, Any]] = {}
        for row in copied:
            operation, cc, m, n, addend, input_layout, output_layout, batch_size = _key(
                row
            )
            backend = row["backend"]
            raster = row.get("raster", 1)
            if (
                operation not in ("syrk", "symm")
                or backend not in ("torch", "cuda")
                or input_layout not in ("row", "column")
                or output_layout not in ("row", "column")
            ):
                raise ValueError("invalid symmetric operation or backend")
            if (
                m < 1
                or n < 1
                or not 1 <= batch_size <= 65535
                or cc < 0
                or not isinstance(raster, int)
                or abs(raster) > 32
            ):
                raise ValueError("invalid geometry, capability or raster")
            key = (operation, cc, m, n, addend, input_layout, output_layout, batch_size)
            if key in plans:
                raise ValueError("duplicate symmetric plan key")
            if backend == "cuda":
                if m % 64 or n % 64:
                    raise ValueError("CUDA plan dimensions must be multiples of 64")
                if operation not in vocabulary:
                    vocabulary[operation] = {
                        tile["name"]: tile for tile in tiles(operation)
                    }
                if row.get("tile") not in vocabulary[operation]:
                    raise ValueError("unknown symmetric tile")
                if (
                    input_layout
                    not in vocabulary[operation][row["tile"]]["input_layouts"]
                ):
                    raise ValueError("tile does not support input layout")
            plans[key] = Plan(backend, row.get("tile"), raster)
        _rows, _plans = copied, plans
        _heuristic = heuristic if heuristic is not None else False
        _revision += 1
    elif heuristic is not None:
        _heuristic = heuristic
        _revision += 1
    return [dict(row) for row in _rows]


@contextmanager
def override(
    rows: Iterable[Mapping[str, Any]], *, heuristic: bool = False
) -> Iterator[None]:
    """Scope table-only rows or a hybrid plan, restoring both after errors."""
    global _rows, _plans, _revision, _heuristic
    saved_rows, saved_plans, saved_heuristic = _rows, _plans, _heuristic
    configure(rows, heuristic=heuristic)
    try:
        yield
    finally:
        _rows, _plans, _heuristic = saved_rows, saved_plans, saved_heuristic
        _revision += 1


def revision() -> int:
    """Version of the measured table, including scoped restoration."""
    return _revision


def layout(x: Tensor) -> Optional[str]:
    """Recognize dense BLAS layouts without materializing a transpose."""
    if x.ndim not in (2, 3):
        return None
    if x.ndim == 3 and (
        not 1 <= x.size(0) <= 65535 or x.stride(0) != x.size(-2) * x.size(-1)
    ):
        return None
    if x.is_contiguous():
        return "row"
    if x.stride(-2) == 1 and x.stride(-1) == x.size(-2):
        return "column"
    return None


@lru_cache(maxsize=None)
def _capability(device: torch.device) -> Tuple[int, int]:
    # Device identities are concrete CUDA indices after tensor allocation.
    # Capability is fixed for a process; reduction/determinism flags are not.
    return torch.cuda.get_device_capability(device)


def supports(x: Tensor) -> bool:
    """Common CUDA matrix capability for dense row/column-major matrices."""
    return (
        x.ndim in (2, 3)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and (layout(x) is not None)
        and x.data_ptr() % 16 == 0
        and min(x.shape[-2:]) >= 64
        and x.size(-2) % 64 == 0
        and x.size(-1) % 64 == 0
        and x.numel() <= 2147483647
        and x.size(-2) ** 2 <= 2147483647
        and _capability(x.device)[0] >= 8
        and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        and not torch.are_deterministic_algorithms_enabled()
        and not x.requires_grad
        and is_available()
    )


@lru_cache(maxsize=256)
def _heuristic_decision(
    operation: str,
    rows: int,
    cols: int,
    batch_size: int,
    input_layout: str,
    output_layout: str,
    addend: bool,
    device: int,
) -> Plan:
    metadata = heuristic_plan(
        operation, rows, cols, batch_size, input_layout, output_layout, addend, device
    )
    if not metadata:
        return Plan()
    return Plan("cuda", metadata["tile"], metadata["raster"])


def probe(
    operation: str,
    x: Tensor,
    *,
    output: Optional[Tensor] = None,
    addend: bool = False,
    input_layout: Optional[str] = None,
    output_layout: Optional[str] = None,
) -> Plan:
    """Inspect a measured or modeled decision without launching a kernel."""
    if operation not in ("syrk", "symm"):
        raise ValueError("operation must be syrk or symm")
    if not supports(x) or (output is not None and not supports(output)):
        return Plan()
    major, minor = _capability(x.device)
    rows, cols = x.size(-2), x.size(-1)
    batch_size = x.size(0) if x.ndim == 3 else 1
    input_layout = input_layout or layout(x)
    output_layout = output_layout or (layout(output) if output is not None else "row")
    key = (
        operation,
        major * 10 + minor,
        rows,
        cols,
        addend,
        input_layout,
        output_layout,
        batch_size,
    )
    if key in _plans:
        return _plans[key]
    if not _heuristic:
        return Plan()
    return _heuristic_decision(
        operation,
        rows,
        cols,
        batch_size,
        input_layout,
        output_layout,
        addend,
        x.device.index,
    )
