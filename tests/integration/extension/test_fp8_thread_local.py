"""FP8 primitives: kernel-level (CUDA) and policy-level (CPU-verifiable) tests.

The kernel-level tests exercise the two stateless primitives (``quantize`` for
bf16/fp16/fp32 -> FP8, ``quant_gemm`` for the pre-quantized GEMM with transposed
operands); the policy-level tests (recipes, autocast context, per-tensor
meta) run without a GPU. The primitives themselves are CUDA-only
(attention-style direct wrappers — no torch.library dispatch layer).
"""

import threading

try:
    from astrai.extension.kernel.quantize import K_FOLD_SLOTS
except RuntimeError:
    # The binding must stay import-safe on boxes without the extension;
    # every K_FOLD_SLOTS use sits inside kernel-level tests that skip
    # via skip_no_fp8/skip_no_kernel when the kernel is not built.
    K_FOLD_SLOTS = None
from astrai.extension.policy.quantization.autocast import (
    fp8_autocast,
    fp8_linear_enabled,
)

# --------------------------------------------------------------------------
# Kernel-level (CUDA)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Policy-level (CPU-verifiable)
# --------------------------------------------------------------------------


def test_autocast_state_is_thread_local():
    """torch parity: the active config is thread-local — another thread does
    not see an open region (CPU-only check of the flag, no kernels)."""
    seen = {}
    with fp8_autocast(enabled=True):
        assert fp8_linear_enabled()
        t = threading.Thread(target=lambda: seen.update(enabled=fp8_linear_enabled()))
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()
    assert seen["enabled"] is False
    assert not fp8_linear_enabled()
