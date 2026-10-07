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
    x.div_(x.norm().clamp(min=eps))
    if steps == 0:
        return x.T if tall else x
    gram = torch.empty((x.size(0), x.size(0)), dtype=x.dtype, device=x.device)
    polynomial = torch.empty_like(gram)
    explicit = "torch" if backend == "torch" else None
    # The first/last BLAS calls change layout directly when the measured Gram
    # plan prefers row-major scratch; external tensors keep their orientation.
    row_work = (
        backend == "auto"
        and not x.is_contiguous()
        and steps > 1
        and probe("syrk", x, input_layout="row").backend == "cuda"
    )
    work = (
        torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if row_work
        else torch.empty_like(x)
    )
    spare = torch.empty_like(work) if steps > 1 else work
    final = torch.empty_like(x) if row_work else None
    first_gram = select("syrk", x, gram, backend=explicit)
    gram_op = select("syrk", work, gram, backend=explicit) if row_work else first_gram
    polynomial_op = select(
        "syrk", gram, polynomial, backend=explicit, addend=gram, alpha=c, beta=b
    )
    first_update = select(
        "symm", polynomial, x, work, backend=explicit, addend=x, beta=a
    )
    update_op = (
        select("symm", polynomial, work, spare, backend=explicit, addend=work, beta=a)
        if row_work
        else first_update
    )
    final_update = (
        select("symm", polynomial, work, final, backend=explicit, addend=work, beta=a)
        if final is not None
        else update_op
    )
    for iteration in range(steps):
        (first_gram if iteration == 0 else gram_op)(x, gram)
        polynomial_op(gram, polynomial, addend=gram, alpha=c, beta=b)
        output = (
            final
            if final is not None and iteration == steps - 1
            else (work if iteration % 2 == 0 else spare)
        )
        operation = (
            first_update
            if iteration == 0
            else final_update
            if output is final
            else update_op
        )
        operation(polynomial, x, output, addend=x, beta=a)
        x = output
    return x.T if tall else x
