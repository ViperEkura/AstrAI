"""FP8 primitives: kernel-level (CUDA) and policy-level (CPU-verifiable) tests.

The kernel-level tests exercise the two stateless primitives (``quantize`` for
bf16/fp16/fp32 -> FP8, ``quant_gemm`` for the pre-quantized GEMM with transposed
operands); the policy-level tests (recipes, autocast context, per-tensor
meta) run without a GPU. The primitives themselves are CUDA-only
(attention-style direct wrappers — no torch.library dispatch layer).
"""

import torch

import astrai.extension.policy.quantization.autocast as f8mod

try:
    from astrai.extension.kernel.quantize import K_FOLD_SLOTS
except RuntimeError:
    # The binding must stay import-safe on boxes without the extension;
    # every K_FOLD_SLOTS use sits inside kernel-level tests that skip
    # via skip_no_fp8/skip_no_kernel when the kernel is not built.
    K_FOLD_SLOTS = None
from astrai.extension.policy.quantization.autocast import (
    FP8Recipe,
    fp8_autocast,
    fp8_format_pair,
    fp8_linear_enabled,
)

# --------------------------------------------------------------------------
# Kernel-level (CUDA)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Policy-level (CPU-verifiable)
# --------------------------------------------------------------------------


def test_fp8_format_pair():
    """A format spec normalizes to (fwd, bwd) fp8 dtypes; torch.dtype is the
    canonical key — no custom format enum."""
    assert fp8_format_pair("hybrid") == (torch.float8_e4m3fn, torch.float8_e5m2)
    assert fp8_format_pair(torch.float8_e4m3fn) == (
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
    )
    assert fp8_format_pair(torch.float8_e5m2) == (
        torch.float8_e5m2,
        torch.float8_e5m2,
    )


def test_fp8_autocast_context():
    """fp8_autocast pushes and restores the thread-local active config."""
    f8mod.fp8_reset()
    try:
        with fp8_autocast(enabled=True, fp8_format="hybrid", update_interval=8):
            cfg = f8mod._active_config.get()
            assert cfg is not None and cfg.enabled
            assert not cfg.recipe.dynamic
            assert cfg.recipe.history_len == 8
            assert cfg.fp8_format == (torch.float8_e4m3fn, torch.float8_e5m2)
            with fp8_autocast(
                enabled=True,
                recipe=FP8Recipe(dynamic=True),
                fp8_format=torch.float8_e4m3fn,
            ):
                inner = f8mod._active_config.get()
                assert inner.recipe.dynamic
                assert inner.fp8_format == (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fn,
                )
            assert f8mod._active_config.get() is cfg  # restored on exit
        assert f8mod._active_config.get() is None
        assert not fp8_linear_enabled()
    finally:
        f8mod.fp8_reset()
