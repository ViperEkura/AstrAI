"""Muon Newton-Schulz CUDA kernel wrapper."""

from collections.abc import Sequence

import torch

from astrai.extension.loader import get_module
from astrai.extension.loader import is_available as _kernel_available


def is_available() -> bool:
    """Return whether the Muon NS CUDA extension can be loaded."""
    return _kernel_available("muon_ns")


def muon_ns(
    grad: torch.Tensor,
    ns_coefficients: Sequence[float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    """Run Muon's Newton-Schulz orthogonalization in one extension call."""
    if not grad.is_cuda:
        raise ValueError("grad must be a CUDA tensor")
    if grad.ndim != 2:
        raise ValueError("grad must be a 2D matrix")
    if not grad.is_floating_point():
        raise ValueError("grad must have a floating-point dtype")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must contain exactly three values")
    if ns_steps < 1 or ns_steps >= 100:
        raise ValueError("ns_steps must be in [1, 99]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return get_module("muon_ns").muon_ns(
        grad.contiguous(),
        [float(value) for value in ns_coefficients],
        int(ns_steps),
        float(eps),
    )


__all__ = ["is_available", "muon_ns"]
