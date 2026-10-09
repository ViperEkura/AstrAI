"""Quantized-GEMM primitives: W8A16 / W8A8 / W16A16 kernel-level tests plus
the int8 policy layer (quantizers). Kernel tests exercise the stateless
wrappers in ``astrai.extension.kernel.gemm`` against torch references built
from the same quantized values; policy tests check the quantizers'
contracts.
"""

import pytest
import torch

from astrai.extension.kernel.gemm import quant_gemm

# bf16-output comparisons: the kernel's dequant and fp32 accumulation are
# exact, so the residual is output rounding plus accumulation-order
# differences against the torch reference — the same tolerance class the
# C harness uses.
ATOL, RTOL = 0.25, 0.02


class TestQuantGemmCPUValidation:
    def test_cpu_rejected(self):
        x = torch.randn(8, 16).to(torch.bfloat16)
        w8 = torch.zeros(8, 16, dtype=torch.int8)
        s = torch.ones(8)
        with pytest.raises(RuntimeError, match="CUDA"):
            quant_gemm(x, w8, b_scale=s)
