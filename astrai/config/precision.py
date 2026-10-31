"""Serializable training precision, independent of optimizer placement."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PrecisionConfig:
    """Explicit compute, accumulation, persistent-state and export formats."""

    compute_dtype: Optional[str] = None
    gradient_dtype: str = "float32"
    master_dtype: str = "float32"
    optimizer_state_dtype: str = "float32"
    export_dtype: Optional[str] = None

    def __post_init__(self) -> None:
        supported = {"float32", "bfloat16"}
        for name in ("compute_dtype", "master_dtype", "optimizer_state_dtype"):
            value = getattr(self, name)
            if (
                name != "compute_dtype" or value is not None
            ) and value not in supported:
                raise ValueError(f"precision.{name} must be float32 or bfloat16")
        if self.gradient_dtype not in supported | {"model"}:
            raise ValueError(
                "precision.gradient_dtype must be float32, bfloat16 or model"
            )
        if self.export_dtype not in supported | {"float16", None}:
            raise ValueError(
                "precision.export_dtype must be float32, bfloat16, float16 or None"
            )
