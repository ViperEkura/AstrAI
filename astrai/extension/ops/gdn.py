"""Gated DeltaNet preparation CUDA kernel wrapper.

``gated_deltanet_fwd`` folds the layout work the chunked Gated DeltaNet operators need into two
launches: L2-normalizing the query/key rows, transposing q/k/v from the layer's
token-major layout to head-major, and pre-scanning the gate into a chunk-local
cumsum. Done with torch, those steps cost more than every Gated DeltaNet kernel combined
(53% of the operator's wall clock at T=2048).

Raises ``RuntimeError`` when the kernel is not built; callers fall back to the
torch path.
"""

from typing import Optional

import torch

from astrai.extension.loader import get_module


def gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
    chunk: int = 64,
) -> Optional[tuple]:
    """Normalize, transpose and gate-scan in one pass.

    Args:
        q/k/v: ``[B, T, H, D]`` bf16 carrying the projection layout, i.e. the
            *token* axis is the contiguous one (memory order ``[B, H, D, T]``).
            That is what the layer's projections and local convolution produce;
            a plain contiguous ``[B, T, H, D]`` tensor is rejected rather than
            silently misread.
        g/beta: ``[B, T, H]`` fp32, contiguous.
        eps: L2-normalization epsilon, added inside the rsqrt.
        chunk: gate scan window; must be a power of two in ``[32, 1024]``.

    Returns:
        ``(q_hat, k_hat, v_hat, g_cumsum, beta_hat)`` where the first three are
        head-major ``[B, H, T, D]`` bf16 and the last two are head-major
        ``[B, H, T]`` fp32.
    """
    mod = get_module("gated_deltanet")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(1) != 1:
            raise ValueError(
                f"{name} must carry the projection layout (token axis contiguous), "
                f"got strides {tuple(tensor.stride())}"
            )
    for name, tensor in (("g", g), ("beta", beta)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    return tuple(mod.gated_deltanet_fwd(q, k, v, g, beta, eps, chunk))


__all__ = ["gdn_fwd"]
