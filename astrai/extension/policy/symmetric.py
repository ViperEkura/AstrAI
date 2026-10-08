"""Measured and geometry plans for symmetric operations, independent of optimizers.

Exact measured rows take precedence over the generic geometry heuristic. Sweep
results can replace the table without rebuilding CUDA kernels. Plan keys describe
operation, device capability and matrix metadata; no model names participate.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
from torch import Tensor

from astrai.extension.kernel.symmetric import is_available, tiles
from astrai.extension.kernel.symmetric import plan as heuristic_plan


@dataclass(frozen=True)
class Plan:
    backend: str = "torch"
    tile: Optional[str] = None
    raster: int = 1


Key = Tuple[str, int, int, int, bool, str, str, int]
# Seeded by interleaved CUDA-event measurements; configure() can replace them.
_DEFAULT_ROWS = [
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 1536,
        "addend": False,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "64x64x32_W16x32_S2",
        "raster": 1,
    },
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": False,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "64x64x32_W16x32_S3",
        "raster": 1,
    },
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 1536,
        "addend": True,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "64x64x32_W16x32_S2",
        "raster": 1,
    },
    {
        "operation": "symm",
        "cc": 120,
        "rows": 256,
        "cols": 1536,
        "addend": True,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "32x32x32_W16x16_S2",
        "raster": 0,
    },
    {
        "operation": "symm",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": True,
        "input_layout": "row",
        "output_layout": "column",
        "backend": "cuda",
        "tile": "128x128x32_W32x32_S2",
        "raster": -2,
    },
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 1536,
        "addend": False,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "128x128x32_W32x32_S2",
        "raster": 1,
        "batch_size": 4,
    },
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": False,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "128x128x32_W32x32_S2",
        "raster": 1,
        "batch_size": 4,
    },
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 1536,
        "addend": True,
        "batch_size": 4,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "128x128x32_W32x32_S2",
        "raster": 1,
    },
    {
        "operation": "symm",
        "cc": 120,
        "rows": 256,
        "cols": 1536,
        "addend": True,
        "batch_size": 4,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "64x64x32_W16x32_S2",
        "raster": 0,
    },
    {
        "operation": "symm",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": True,
        "batch_size": 4,
        "input_layout": "row",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "128x128x32_W32x32_S2",
        "raster": -2,
    },
    {
        "operation": "symm",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": True,
        "batch_size": 4,
        "input_layout": "column",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "128x128x32_W32x32_S2",
        "raster": -2,
    },
    {
        "operation": "symm",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": True,
        "batch_size": 4,
        "input_layout": "row",
        "output_layout": "column",
        "backend": "cuda",
        "tile": "128x128x32_W32x32_S2",
        "raster": -1,
    },
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": False,
        "batch_size": 1,
        "input_layout": "column",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "64x64x64_W16x16_S2",
        "raster": 1,
    },
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": False,
        "batch_size": 4,
        "input_layout": "column",
        "output_layout": "row",
        "backend": "cuda",
        "tile": "128x128x64_W64x32_S2",
        "raster": 1,
    },
]

# Exact Torch winners from the same tile sweep; no global size threshold.
_DEFAULT_ROWS += [
    {
        "operation": operation,
        "cc": 120,
        "rows": rows,
        "cols": cols,
        "addend": addend,
        "batch_size": batch,
        "input_layout": input_layout,
        "output_layout": "row",
        "backend": "torch",
    }
    for operation, rows, cols, addend, batch, input_layout in (
        ("syrk", 192, 384, False, 4, "row"),
        ("syrk", 192, 192, True, 4, "row"),
        ("syrk", 256, 256, False, 4, "row"),
        ("syrk", 256, 256, True, 4, "row"),
        ("syrk", 256, 1536, False, 4, "row"),
        ("symm", 1536, 1536, True, 4, "row"),
        ("syrk", 256, 256, False, 1, "row"),
        ("syrk", 256, 256, True, 1, "row"),
        ("syrk", 256, 1536, False, 1, "row"),
        ("symm", 1536, 1536, True, 1, "row"),
        ("symm", 1536, 6912, True, 1, "row"),
        ("symm", 1536, 6912, True, 1, "column"),
    )
]


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
_rows: List[Dict[str, Any]] = [dict(row) for row in _DEFAULT_ROWS]
_plans: Dict[Key, Plan] = {
    _key(row): Plan(row["backend"], row.get("tile"), row.get("raster", 1))
    for row in _rows
}


def configure(
    rows: Optional[Iterable[Mapping[str, Any]]] = None,
    *,
    heuristic: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Replace measured rows atomically; no argument reads the current table.

    Every row has operation, cc, rows, cols and backend. Optional fields include
    addend, tile, raster, input/output layout and batch_size (default 1).
    Replacing rows defaults to table-only dispatch; heuristic=True enables
    geometry fallback for missing keys. A matching Torch row always wins.
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
    """Inspect a measured or geometry decision without launching a kernel."""
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
