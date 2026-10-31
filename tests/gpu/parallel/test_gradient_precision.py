"""Real NCCL coverage of FP32 accumulation and optimizer updates."""

from pathlib import Path

import torch.multiprocessing as mp

from tests.integration.parallel.test_gradient_precision import _distributed_check
from tests.support.capabilities import skip_lt2_cuda


@skip_lt2_cuda
def test_nccl_fp32_accumulation_and_update_parity(tmp_path):
    mp.spawn(
        _distributed_check,
        args=(2, Path(tmp_path / "nccl").as_uri(), True, False, "nccl"),
        nprocs=2,
        join=True,
    )
