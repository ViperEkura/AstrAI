"""Quantize-kernel interface adapter (the only module touching the pybind).

Attention-style thin wrappers: one Python entry per binding of the compiled
``quantize`` module, called directly — no torch.library dispatch layer.
Optional arguments (``ring_state``) keep native Optional semantics at the
pybind boundary, and in-place buffer updates (the delayed-scaling ring fold,
like attention's KV-cache appends) happen on-stream without mutation
declarations. CUDA-only: non-CUDA or unsupported inputs raise from the
binding's TORCH_CHECKs.

- ``quantize(x, scale, fmt, transposed=False) -> (x8|x8T, amax)`` — BF16/FP16/FP32
  → FP8 with fused amax (``transposed`` picks the orientation; arity is fixed)
- ``quantize_dual(x, scale, fmt) -> (x8, x8T, amax)`` — both orientations, one read

``scale`` is the quantization multiplier (device scalar); ``fmt`` is the fp8
output dtype (``torch.float8_e4m3fn`` / ``torch.float8_e5m2``) — the binding
validates and dispatches on the dtype itself. ``amax`` is *produced only by
the delayed-scaling
ring fold* (the kernel's fused reduction) or measured by the caller; the
returned value is the ring's self-cleaned slot when ``ring_state`` is given,
else ``None`` (the kernel runs a pure scale+cast — no fused-amax pass).

Policy (scales, amax history, delayed scaling, autocast) lives in
``astrai.extension.quantize``; this module is stateless.
"""

from typing import Optional, Tuple

import torch

from astrai.extension.loader import get_module


def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    fmt: torch.dtype = torch.float8_e4m3fn,
    transposed: bool = False,
    ring_state: Optional[torch.Tensor] = None,
    hist_idx: int = 0,
    fp8_max: float = 448.0,
    pow2_margin: float = 1.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Float (bf16/fp16/fp32) -> FP8 quantize (scale-then-cast).

    ``scale`` is the quantization multiplier (device scalar); ``fmt`` is the
    fp8 output dtype (E4M3 or E5M2) — the binding validates and dispatches
    on it. ``transposed=True`` swaps ``x8`` for ``x8T``, the
    ``[cols][rows]`` row-major transpose of the quantized input — the
    K-contiguous operand orientation NT GEMMs want — at the same 2-tuple
    arity.

    ``ring_state`` (a 1D float32 CUDA buffer laid out
    ``[hist n | scale | legacy | amax | done]``) switches on the in-kernel
    delayed-scaling fold: the kernel's last block folds the amax into
    ``hist[hist_idx]`` and publishes the next scale as
    ``max(hist) / fp8_max / pow2_margin``. The returned ``amax`` is then the
    self-cleaned persistent slot (reads zero).

    No ``ring_state`` means a pure scale+cast: no fused-amax reduction and
    ``amax`` is ``None``. Callers that want the amax reduce the input
    themselves (dynamic scaling measures ``x.abs().amax()``).
    """
    return get_module("quantize").quantize(
        x,
        scale,
        fmt,
        transposed,
        ring_state,
        hist_idx,
        fp8_max,
        pow2_margin,
    )


def quantize_dual(
    x: torch.Tensor,
    scale: torch.Tensor,
    fmt: torch.dtype = torch.float8_e4m3fn,
    ring_state: Optional[torch.Tensor] = None,
    hist_idx: int = 0,
    fp8_max: float = 448.0,
    pow2_margin: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Dual-orientation quantize: one read of ``x`` produces both the
    row-major ``x8`` and its transposed ``x8T`` (plus ``amax``), for tensors
    consumed by GEMMs in both orientations (backward ``g``).

    ``ring_state`` switches on the in-kernel delayed-scaling fold exactly as
    in :func:`quantize`; with no ``ring_state`` the kernel runs a pure
    scale+cast and ``amax`` is ``None``.
    """
    return get_module("quantize").quantize_dual(
        x,
        scale,
        fmt,
        ring_state,
        hist_idx,
        fp8_max,
        pow2_margin,
    )


__all__ = ["quantize", "quantize_dual"]
