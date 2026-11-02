"""Shared activation casts in the composed FP8 linear path."""

import pytest
import torch
import torch.nn.functional as F

from astrai.extension.policy.quantization.autocast import fp8_autocast
from astrai.extension.runtime.loader import get_module
from tests.support.capabilities import skip_no_fp8

FMT_A = torch.float8_e4m3fn
FMT_B = torch.float8_e5m2


def _gemm():
    return get_module("gemm")


def _call(x, weight):
    return _gemm().fp8_linear(x, weight, None, True, False, False, 16, 0, FMT_A, FMT_B)


@pytest.fixture(autouse=True)
def _clean_fp8_state():
    gemm = _gemm()
    gemm.fp8_reset()
    gemm.fp8_set_act_cache(True)
    yield
    gemm.fp8_reset()
    gemm.fp8_set_act_cache(True)


@pytest.mark.cuda
@skip_no_fp8
def test_shared_input_cast_is_reused_with_bitwise_gradients():
    torch.manual_seed(101)
    x = torch.randn(24, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    y1 = _call(x, w1)
    y2 = _call(x, w2)
    (y1.float().sum() + y2.float().sum()).backward()
    torch.cuda.synchronize()

    m1 = _gemm().fp8_debug_meta(w1)
    m2 = _gemm().fp8_debug_meta(w2)
    assert m1["x"]["idx"] == m2["x"]["idx"] == 1
    assert m1["x"]["state"].data_ptr() == m2["x"]["state"].data_ptr()
    assert x.grad is not None and w1.grad is not None and w2.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.cuda
@skip_no_fp8
def test_cache_off_and_on_are_bitwise_equivalent_across_steps():
    torch.manual_seed(102)
    x0 = torch.randn(19, 64, device="cuda", dtype=torch.bfloat16)
    x1 = torch.randn(19, 64, device="cuda", dtype=torch.bfloat16) * 0.4
    w10 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    w20 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)

    def run(enabled):
        gemm = _gemm()
        gemm.fp8_reset()
        gemm.fp8_set_act_cache(enabled)
        x = x0.detach().clone().requires_grad_()
        w1 = w10.detach().clone().requires_grad_()
        w2 = w20.detach().clone().requires_grad_()
        out0 = (_call(x, w1), _call(x, w2))
        (out0[0].float().square().sum() + out0[1].float().square().sum()).backward()
        grads0 = (x.grad.clone(), w1.grad.clone(), w2.grad.clone())
        with torch.no_grad():
            w1.add_(0.001)
            w2.add_(0.001)
        gemm.fp8_clear_act_cache()
        x_next = x1.detach().clone().requires_grad_()
        out1 = (_call(x_next, w1), _call(x_next, w2))
        return (*out0, *out1, *grads0)

    ref = run(False)
    cached = run(True)
    torch.cuda.synchronize()
    for got, expected in zip(cached, ref, strict=True):
        assert torch.equal(got, expected)


@pytest.mark.cuda
@skip_no_fp8
def test_cache_key_uses_tensor_identity_and_version():
    torch.manual_seed(103)
    gemm = _gemm()
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)

    _call(x, w1)
    _call(x.clone(), w2)  # same values, different TensorImpl: must miss
    first = gemm.fp8_debug_meta(w1)["x"]["state"]
    other = gemm.fp8_debug_meta(w2)["x"]["state"]
    assert first.data_ptr() != other.data_ptr()

    gemm.fp8_clear_act_cache()
    _call(x, w1)
    with torch.no_grad():
        x.add_(0.01)  # same TensorImpl, bumped version: stale cast must miss
    _call(x, w2)
    assert (
        gemm.fp8_debug_meta(w1)["x"]["state"].data_ptr()
        != gemm.fp8_debug_meta(w2)["x"]["state"].data_ptr()
    )


@pytest.mark.cuda
@skip_no_fp8
def test_dynamic_scaling_does_not_use_activation_cache():
    torch.manual_seed(105)
    gemm = _gemm()
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    for weight in (w1, w2):
        gemm.fp8_linear(x, weight, None, True, False, True, 16, 0, FMT_A, FMT_B)
    assert gemm.fp8_state_dict()["entries"] == []


@pytest.mark.cuda
@skip_no_fp8
def test_cache_is_bounded_and_evicts_least_recently_used_entry():
    torch.manual_seed(105)
    gemm = _gemm()
    inputs = [
        torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(10)
    ]
    weights = [
        torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) for _ in range(10)
    ]
    for x, weight in zip(inputs, weights, strict=True):
        _call(x, weight)

    # The newest entry is reused; the oldest entry must fold again after eviction.
    _call(inputs[-1], weights[-1])
    assert gemm.fp8_debug_meta(weights[-1])["x"]["idx"] == 1
    _call(inputs[0], weights[0])
    assert gemm.fp8_debug_meta(weights[0])["x"]["idx"] == 2


@pytest.mark.cuda
@skip_no_fp8
def test_cache_is_cleared_when_autocast_region_exits():
    torch.manual_seed(104)
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    gemm = _gemm()

    with fp8_autocast(enabled=True):
        F.linear(x, w1)
        F.linear(x, w2)
    assert gemm.fp8_debug_meta(w1)["x"]["idx"] == 1
    assert (
        gemm.fp8_debug_meta(w1)["x"]["state"].data_ptr()
        == gemm.fp8_debug_meta(w2)["x"]["state"].data_ptr()
    )
    with fp8_autocast(enabled=True):
        F.linear(x, w1)
    assert gemm.fp8_debug_meta(w1)["x"]["idx"] == 2


@pytest.mark.cuda
@skip_no_fp8
def test_cache_invalidates_when_input_moves_device_without_version_change():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    gemm = _gemm()
    x = torch.ones(16, 64, device="cuda:0", dtype=torch.bfloat16)
    w0 = torch.ones(32, 64, device="cuda:0", dtype=torch.bfloat16)
    w1 = w0.to("cuda:1")
    _call(x, w0)
    version = x._version
    identity = x._cdata
    x.data = x.to("cuda:1")
    assert x._version == version and x._cdata == identity
    out = _call(x, w1)
    assert out.device == x.device
    assert gemm.fp8_debug_meta(w1)["x"]["state"].device == x.device
    gemm.fp8_set_act_cache(False)
    ref = _call(x.clone(), w1.clone())
    assert torch.equal(out, ref)
