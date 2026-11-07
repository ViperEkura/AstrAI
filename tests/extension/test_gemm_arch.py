"""Exercise compiled GEMM schedules with independent runtime switches."""

import re

import pytest
import torch

from astrai.extension import kernel, plan
from tests.support.capabilities import skip_no_kernel


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("axis", ["m", "n", "k"])
@pytest.mark.parametrize("value", [0, -1, 2**31, 2**32 + 17])
def test_probe_rejects_invalid_native_extents(axis, value):
    shape = dict(m=17, n=23, k=33)
    shape[axis] = value
    with pytest.raises(RuntimeError, match="greater than zero|fit native int"):
        kernel.gemm.probe(**shape)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("batch", [0, -1, 65536, 2**32 + 1])
def test_probe_rejects_invalid_batch(batch):
    with pytest.raises(RuntimeError, match="batch"):
        kernel.gemm.probe(17, 23, 33, batch=batch)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("axis", ["m", "n", "k"])
def test_launch_rejects_oversized_expanded_extents(axis):
    shape = dict(m=1, n=1, k=1)
    shape[axis] = 2**32 + 17
    storage = torch.ones(1, device="cuda", dtype=torch.bfloat16)
    a = storage.as_strided((shape["m"], shape["k"]), (0, 0))
    b = storage.as_strided((shape["n"], shape["k"]), (0, 0))
    with pytest.raises(RuntimeError, match="fit native int"):
        kernel.gemm.quant_gemm(a, b)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("side", ["a", "b"])
def test_launch_rejects_empty_operand_batches(side):
    a = torch.ones(1, 17, 33, device="cuda", dtype=torch.bfloat16)
    b = torch.ones(1, 23, 33, device="cuda", dtype=torch.bfloat16)
    if side == "a":
        a = a[:0]
    else:
        b = b[:0]
    with pytest.raises(RuntimeError, match="batch dimensions must be positive"):
        kernel.gemm.quant_gemm(a, b)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("side", ["a", "b"])
def test_launch_rejects_leading_stride_overflow(side):
    a = torch.ones(1, 33, device="cuda", dtype=torch.bfloat16)
    b = torch.ones_like(a)
    if side == "a":
        a = a.as_strided(a.shape, (2**31, 1))
    else:
        b = b.as_strided(b.shape, (2**31, 1))
    with pytest.raises(RuntimeError, match="leading stride"):
        kernel.gemm.quant_gemm(a, b)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("case", ["inner", "bias"])
def test_invalid_metadata_rejected_before_operand_copies(case):
    a = torch.ones(34, 66, device="cuda", dtype=torch.bfloat16)[::2, ::2]
    b = torch.ones(46, 66, device="cuda", dtype=torch.bfloat16)[::2, ::2]
    bias = None
    if case == "inner":
        b = b[:, :-1]
    else:
        bias = torch.zeros(23, device="cuda", dtype=torch.float32)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        with pytest.raises(RuntimeError, match="inner dim|bias"):
            kernel.gemm.quant_gemm(a, b, bias=bias)
    assert not {event.key for event in prof.key_averages()} & {
        "aten::contiguous",
        "aten::clone",
        "aten::copy_",
    }


@pytest.mark.cuda
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


@pytest.mark.cuda
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


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize(
    "trans_a,trans_b", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("batched_a", [False, True])
def test_strided_operand_fallback_keeps_storage_alive(trans_a, trans_b, batched_a):
    m, n, k, batch = 17, 23, 33, 2

    def operand(rows, columns, batched):
        shape = (batch, rows * 2, columns * 2) if batched else (rows * 2, columns * 2)
        storage = torch.randint(-2, 3, shape, device="cuda").to(torch.bfloat16)
        return storage[..., ::2, ::2]

    a = operand(k if trans_a else m, m if trans_a else k, batched_a)
    b = operand(n if trans_b else k, k if trans_b else n, not batched_a)
    assert a.stride(-1) != 1 and a.stride(-2) != 1
    assert b.stride(-1) != 1 and b.stride(-2) != 1
    lhs = a.transpose(-1, -2) if trans_a else a
    rhs = b.transpose(-1, -2) if trans_b else b
    expected = (lhs.float() @ rhs.float()).to(torch.bfloat16)
    output = kernel.gemm.quant_gemm(a, b, trans_a=trans_a, trans_b=trans_b)
    # More fallback allocations on the same stream may reuse temporary storage.
    for _ in range(3):
        kernel.gemm.quant_gemm(-a, b, trans_a=trans_a, trans_b=trans_b)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_operand_device_guard_covers_packing_and_restores_current_device():
    original = torch.cuda.current_device()
    target = (original + 1) % torch.cuda.device_count()
    a = torch.randint(-2, 3, (34, 66), device=f"cuda:{target}").to(torch.bfloat16)[
        ::2, ::2
    ]
    b = torch.randint(-2, 3, (46, 66), device=f"cuda:{target}").to(torch.bfloat16)[
        ::2, ::2
    ]
    output = kernel.gemm.quant_gemm(a, b)
    assert torch.cuda.current_device() == original
    assert output.device == a.device
    torch.testing.assert_close(
        output, (a.float() @ b.float().T).to(torch.bfloat16), atol=0, rtol=0
    )
