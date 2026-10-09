"""Exercise compiled GEMM schedules with independent runtime switches."""

import re

import pytest
import torch

from astrai.extension import kernel, plan
from tests.support.capabilities import skip_no_kernel


@skip_no_kernel
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.int8, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize(
    "staging", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("k", [128, 129])
def test_compiled_schedule_switches(dtype, staging, k, capfd):
    caps = kernel.gemm.capabilities()
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and not caps["fp8"]:
        pytest.skip("no compiled FP8 MMA for this device")
    # Small integers are exact in every operand format and in FP32 accumulation.
    a = torch.randint(-2, 3, (65, k), device="cuda").to(dtype)
    b = torch.randint(-2, 3, (129, k), device="cuda").to(dtype)
    scale = torch.ones(1, device="cuda") if dtype == torch.int8 else None
    expected = (a.float() @ b.float().T).to(torch.bfloat16)
    tma, mx = staging
    with plan.override(tma=tma, mx=mx, log=True):
        output = kernel.gemm.quant_gemm(a, b, scale, scale)
        torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    launches = [
        line for line in capfd.readouterr().err.splitlines() if "-> tile " in line
    ]
    assert launches
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert (" mx " in launches[-1]) == (fp8 and mx and caps["mx"])
    # Odd K cannot form the TMA descriptors, so it must use cp.async.
    assert (" tma " in launches[-1]) == (tma and caps["tma"] and k == 128)


@skip_no_kernel
def test_unaligned_base_uses_cpasync_plan(capfd):
    m, n, k = 65, 129, 128
    storage = torch.randint(-2, 3, (m * k + 1,), device="cuda").to(torch.bfloat16)
    a = storage[1:].view(m, k)
    b = torch.randint(-2, 3, (n, k), device="cuda").to(torch.bfloat16)
    with plan.override(planner="model", tma=False):
        cp = kernel.gemm.probe(m, n, k)
    with plan.override(planner="model", tma=True, log=True):
        output = kernel.gemm.quant_gemm(a, b)
        torch.cuda.synchronize()
    torch.testing.assert_close(
        output, (a.float() @ b.float().T).to(torch.bfloat16), atol=0, rtol=0
    )
    log = capfd.readouterr().err
    chosen = re.search(r"\[gemm-plan\] model .* -> cta(\d+) s(\d+)", log)
    assert chosen is not None
    assert tuple(map(int, chosen.groups())) == (cp["cta"], cp["k_stages"])
    assert " tma " not in log
