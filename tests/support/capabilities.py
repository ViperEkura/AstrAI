import pytest
import torch

from astrai.extension import KERNEL_NAMES, is_available

CUDA_AVAIL = torch.cuda.is_available()
KERNEL_AVAIL = CUDA_AVAIL and all(is_available(k) for k in KERNEL_NAMES)
FP8_AVAIL = (
    CUDA_AVAIL
    and is_available("quantize")
    and torch.cuda.get_device_capability() >= (8, 9)
)
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAIL, reason="CUDA not available")
skip_lt2_cuda = pytest.mark.skipif(
    not CUDA_AVAIL or torch.cuda.device_count() < 2,
    reason="two-rank spawn tests need two CUDA devices",
)
skip_no_kernel = pytest.mark.skipif(not KERNEL_AVAIL, reason="CUDA kernels not built")
skip_no_fp8 = pytest.mark.skipif(
    not FP8_AVAIL,
    reason="fused FP8 MMA requires a built kernel and compute capability 8.9+",
)
