"""Measured shape plans for symmetric operations, independently of optimizers.

Unknown shapes use Torch. Sweep results can replace the table without rebuilding
CUDA kernels. Plan keys describe the operation, compute capability and matrix
geometry; no model or parameter names participate in dispatch.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
from torch import Tensor

from astrai.extension.kernel.symmetric import is_available, tiles


@dataclass(frozen=True)
class Plan:
    backend: str = "torch"
    tile: Optional[str] = None
    raster: int = 1


Key = Tuple[str, int, int, int, bool]
# Seeded by interleaved CUDA-event measurements; configure() can replace them.
_DEFAULT_ROWS = [
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "backend": "cuda",
        "tile": "64x64x32_W16x32_S2",
    },
    {
        "operation": "syrk",
        "cc": 120,
        "rows": 1536,
        "cols": 1536,
        "addend": True,
        "backend": "cuda",
        "tile": "wmma64",
    },
    {
        "operation": "symm",
        "cc": 120,
        "rows": 256,
        "cols": 1536,
        "addend": True,
        "backend": "cuda",
        "tile": "64x64x32_W16x32_S3",
        "raster": 0,
    },
    {
        "operation": "symm",
        "cc": 120,
        "rows": 1536,
        "cols": 6912,
        "addend": True,
        "backend": "cuda",
        "tile": "128x128x32_W32x32_S2",
        "raster": 0,
    },
]


def _key(row: Mapping[str, Any]) -> Key:
    return (
        row["operation"],
        int(row["cc"]),
        int(row["rows"]),
        int(row["cols"]),
        bool(row.get("addend", False)),
    )


_rows: List[Dict[str, Any]] = [dict(row) for row in _DEFAULT_ROWS]
_plans: Dict[Key, Plan] = {
    _key(row): Plan(row["backend"], row.get("tile"), row.get("raster", 1))
    for row in _rows
}


def configure(
    rows: Optional[Iterable[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Replace measured rows atomically; no argument reads the current table.

    Every row has operation, cc, rows, cols, backend, optional addend, tile and raster.
    An empty table disables automatic CUDA selection. Duplicate keys are errors.
    """
    global _rows, _plans
    if rows is not None:
        copied = [dict(row) for row in rows]
        plans: Dict[Key, Plan] = {}
        vocabulary: Dict[str, set] = {}
        for row in copied:
            operation, cc, m, n, addend = _key(row)
            backend = row["backend"]
            raster = row.get("raster", 1)
            if operation not in ("syrk", "symm") or backend not in ("torch", "cuda"):
                raise ValueError("invalid symmetric operation or backend")
            if (
                m < 1
                or n < 1
                or cc < 0
                or not isinstance(raster, int)
                or abs(raster) > 32
            ):
                raise ValueError("invalid geometry, capability or raster")
            key = (operation, cc, m, n, addend)
            if key in plans:
                raise ValueError("duplicate symmetric plan key")
            if backend == "cuda":
                if m % 64 or n % 64:
                    raise ValueError("CUDA plan dimensions must be multiples of 64")
                if operation not in vocabulary:
                    vocabulary[operation] = {tile["name"] for tile in tiles(operation)}
                if row.get("tile") not in vocabulary[operation]:
                    raise ValueError("unknown symmetric tile")
            plans[key] = Plan(backend, row.get("tile"), raster)
        _rows, _plans = copied, plans
    return [dict(row) for row in _rows]


@contextmanager
def override(rows: Iterable[Mapping[str, Any]]) -> Iterator[None]:
    """Temporarily replace measured rows, restoring even when a call fails."""
    global _rows, _plans
    saved_rows, saved_plans = _rows, _plans
    configure(rows)
    try:
        yield
    finally:
        _rows, _plans = saved_rows, saved_plans


def supports(x: Tensor, *, packed: bool = False) -> bool:
    """Common CUDA matrix capability; packing may make a strided view contiguous."""
    return (
        x.ndim == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and (packed or x.is_contiguous())
        and (packed or x.data_ptr() % 16 == 0)
        and min(x.shape) >= 64
        and x.size(0) % 64 == 0
        and x.size(1) % 64 == 0
        and x.numel() <= 2147483647
        and x.size(0) ** 2 <= 2147483647
        and torch.cuda.get_device_capability(x.device)[0] >= 8
        and torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        and not torch.are_deterministic_algorithms_enabled()
        and not x.requires_grad
        and is_available()
    )


def probe(
    operation: str, x: Tensor, *, packed: bool = False, addend: bool = False
) -> Plan:
    """Inspect a measured decision without launching any kernel."""
    if operation not in ("syrk", "symm"):
        raise ValueError("operation must be syrk or symm")
    if not supports(x, packed=packed):
        return Plan()
    major, minor = torch.cuda.get_device_capability(x.device)
    return _plans.get(
        (operation, major * 10 + minor, x.size(0), x.size(1), addend), Plan()
    )
