"""Newton-Schulz recurrence, independent of optimizer and parameter routing."""

from typing import Tuple

import torch
from torch import Tensor

from astrai.extension.backend.symmetric import select
from astrai.extension.policy.symmetric import probe


def newton_schulz(
    matrix: Tensor,
    coefficients: Tuple[float, float, float],
    steps: int = 5,
    eps: float = 1e-7,
    *,
    backend: str = "torch",
) -> Tensor:
    """Orthogonalize a matrix with BF16 NS and reusable scratch buffers.

    backend="auto" uses measured symmetric operation plans; "torch" retains
    Torch arithmetic. Coefficients and iteration count are caller supplied.
    BF16 inputs are normalized in place, matching Torch Muon semantics, but
    subsequent iterations never overwrite the normalized caller storage.
    """
    if matrix.ndim != 2 or len(coefficients) != 3 or not 0 <= steps < 100:
        raise ValueError("invalid Newton-Schulz matrix, coefficients or steps")
    if backend not in ("torch", "auto"):
        raise ValueError("backend must be torch or auto")
    a, b, c = coefficients
    x = matrix.bfloat16()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    rank_k = probe("syrk", x, packed=True) if backend == "auto" else None
    pack = rank_k is not None and rank_k.backend == "cuda"
    x.div_(x.norm().clamp(min=eps))
    if pack:
        x = x.contiguous()
    if steps == 0:
        return x.T if tall else x
    gram = torch.empty((x.size(0), x.size(0)), dtype=x.dtype, device=x.device)
    polynomial = torch.empty_like(gram)
    next_x = torch.empty_like(x)
    explicit = "torch" if backend == "torch" else None
    gram_op = select("syrk", x, gram, backend=explicit)
    polynomial_op = select(
        "syrk", gram, polynomial, backend=explicit, addend=gram, alpha=c, beta=b
    )
    update_op = select(
        "symm", polynomial, x, next_x, backend=explicit, addend=x, beta=a
    )
    input_alias = torch._C._is_alias_of(x, matrix)
    for iteration in range(steps):
        gram_op(x, gram)
        polynomial_op(gram, polynomial, addend=gram, alpha=c, beta=b)
        update_op(polynomial, x, next_x, addend=x, beta=a)
        if iteration == 0 and input_alias and steps > 1:
            x = next_x
            next_x = torch.empty_like(x)
        else:
            x, next_x = next_x, x
    result = x.T if tall else x
    return result.contiguous() if pack else result
