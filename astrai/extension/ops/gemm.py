"""GEMM-kernel interface adapter (the only module touching the pybind).

One adapter per compiled module: this file covers the ``gemm`` module's
``quant_gemm`` binding, stateless and called directly, plus the planner
introspection bindings the autotuner uses (``astrai.extension.gemm_autotune``
— installed here as a note() hook, one flag check when disabled). The mma
consumes bf16 fragments (or the native fp8 mma for symmetric fp8 pairs);
int8 and fp8 operands dequantize in-register between the smem read and the
mma — never a separate F2F pass — and per-operand scales fold
multiplicatively into the epilogue.

- ``quant_gemm(a, b, a_scale, b_scale) -> bf16`` — one entry for every
  dtype pairing: W16A16 (bf16 x bf16, no scales), W8A16 (bf16 x int8,
  ``b_scale`` required), W8A8 (int8 x int8, both scales), W-F8A16 (bf16 x
  fp8, ``b_scale`` optional), and symmetric fp8 (matching formats, scales
  optional — the fp8 training path)

Scales are contiguous float32 CUDA tensors: one element (per-tensor
scalar) or the operand's extent — per-row activations ``a_scale[m]`` /
per-channel weights ``b_scale[n]``. The ``trans_a``/``trans_b`` flags name
the math (``True`` = operand laid out ``[contract][rows]``);
inner-transposed views fold into the kernel layout at zero copy. ``bias``
(CUDA bf16 1D of length n) fuses into the epilogue.

Policy (fp8 recipes / autocast, int8 quantizers) lives in
``astrai.extension.quantize``; this module is stateless.
"""

import os
from typing import Optional

import torch

from astrai.extension.loader import get_module

# The autotune hook (gemm_autotune.GemmAutotuner.note): None unless
# ASTR_GEMM_AUTOTUNE=1 or enable() installed one — the off path costs one
# flag check.
_autotuner: Optional[object] = None
_autotuner_resolved = False


def set_autotuner(tuner: object) -> None:
    """Install (or, with None, remove) the runtime autotune hook."""
    global _autotuner
    _autotuner = tuner


def _maybe_enable_autotune() -> None:
    global _autotuner_resolved, _autotuner
    _autotuner_resolved = True
    if os.environ.get("ASTR_GEMM_AUTOTUNE", "") == "1":
        from astrai.extension import gemm_autotune

        gemm_autotune.enable()


def quant_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    trans_a: bool = False,
    trans_b: bool = True,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Quantized GEMM: ``(a @ b) * a_scale * b_scale (+ bias)``.

    The operand dtypes pick the kernel (see the module docstring for the
    pairing table); int8 operands require their dequant scale, fp8
    operands take one optionally, bf16 takes none. The result is bf16.
    """
    global _autotuner_resolved
    if not _autotuner_resolved:
        _maybe_enable_autotune()
    if _autotuner is not None:
        _autotuner.note(a, b, a_scale, b_scale, trans_a, trans_b, bias)
    return get_module("gemm").quant_gemm(a, b, a_scale, b_scale, trans_a, trans_b, bias)


__all__ = ["quant_gemm", "set_autotuner"]
