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
    """Orthogonalize a matrix or independent matrix batch with BF16 NS.

    backend="auto" uses symmetric dispatch plans; "torch" retains
    Torch arithmetic. Coefficients and iteration count are caller supplied.
    A rank-3 input has independent Frobenius normalization per matrix.
    BF16 inputs are normalized in place, matching Torch Muon semantics, but
    subsequent iterations never overwrite the normalized caller storage.
    """
    if matrix.ndim not in (2, 3) or len(coefficients) != 3 or not 0 <= steps < 100:
        raise ValueError("invalid Newton-Schulz matrix, coefficients or steps")
    if backend not in ("torch", "auto"):
        raise ValueError("backend must be torch or auto")
    a, b, c = coefficients
    x = matrix.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.transpose(-2, -1)
    x.div_(
        (x.norm() if x.ndim == 2 else x.norm(dim=(-2, -1), keepdim=True)).clamp(min=eps)
    )
    if steps == 0:
        return x.transpose(-2, -1) if tall else x
    gram = torch.empty(
        (*x.shape[:-2], x.size(-2), x.size(-2)), dtype=x.dtype, device=x.device
    )
    polynomial = torch.empty_like(gram)
    explicit = "torch" if backend == "torch" else None
    # The first/last BLAS calls change layout directly when the selected Gram
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
    return x.transpose(-2, -1) if tall else x
