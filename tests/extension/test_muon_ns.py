import pytest
import torch

from astrai.extension.kernel.muon_ns import is_available, muon_ns
from astrai.extension.loader import get_module
from astrai.optim.muon_adamw import _zeropower_reuse_buffers

CUDA_AVAIL = torch.cuda.is_available()
skip_no_muon_ns = pytest.mark.skipif(
    not CUDA_AVAIL or not is_available(),
    reason="Muon NS CUDA kernel is not available",
)


@pytest.mark.parametrize("shape", [(1, 8), (8, 1), (64, 32), (32, 64), (48, 48)])
@pytest.mark.parametrize("ns_steps", [1, 3, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@skip_no_muon_ns
def test_muon_ns_matches_torch_reference(shape, ns_steps, dtype):
    torch.manual_seed(17)
    grad = torch.randn(shape, device="cuda", dtype=dtype)
    coefficients = (3.4445, -4.7750, 2.0315)
    expected = _zeropower_reuse_buffers(grad.clone(), coefficients, ns_steps, 1e-7)
    actual = muon_ns(grad.clone(), coefficients, ns_steps, 1e-7)

    assert actual.shape == grad.shape
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected), (
        f"Muon NS mismatch for shape={shape}, ns_steps={ns_steps}: "
        f"max_abs={(actual.float() - expected.float()).abs().max().item()}"
    )


@skip_no_muon_ns
def test_muon_ns_accepts_float_input_and_returns_bf16():
    torch.manual_seed(23)
    grad = torch.randn((32, 16), device="cuda", dtype=torch.float32)
    result = muon_ns(grad, (3.4445, -4.7750, 2.0315), 5, 1e-7)

    assert result.dtype == torch.bfloat16
    assert result.shape == grad.shape
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    "shape",
    [
        (64, 128),
        (256, 1024),
        (512, 1024),
        (1024, 1024),
        (1024, 2048),
        (2048, 1024),
        (1536, 1536),
    ],
)
@pytest.mark.parametrize("ns_steps", [1, 5])
@pytest.mark.parametrize("coefficients", [(3.4445, -4.7750, 2.0315), (2.0, -1.0, 0.5)])
@skip_no_muon_ns
def test_fused_polynomial_matches_aten_gemm(shape, ns_steps, coefficients):
    torch.manual_seed(31)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)
    module = get_module("muon_ns")
    expected = module.muon_ns(grad.clone(), coefficients, ns_steps, 1e-7, False)
    actual = module.muon_ns(grad.clone(), coefficients, ns_steps, 1e-7, True)
    assert torch.equal(actual, expected), (
        f"shape={shape}, ns_steps={ns_steps}, "
        f"max_abs={(actual.float() - expected.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("coefficients", [(3.4445, -4.7750, 2.0315), (2.0, -1.0, 0.5)])
@pytest.mark.parametrize("eps", [1e-7, 1e-2])
@skip_no_muon_ns
def test_muon_ns_coefficients_and_eps(coefficients, eps):
    torch.manual_seed(29)
    grad = torch.randn((17, 9), device="cuda", dtype=torch.bfloat16) * 1e-4
    expected = _zeropower_reuse_buffers(grad.clone(), coefficients, 3, eps)
    actual = muon_ns(grad.clone(), coefficients, 3, eps)
    assert torch.equal(actual, expected), (
        f"max_abs={(actual.float() - expected.float()).abs().max().item()}"
    )


@skip_no_muon_ns
def test_muon_ns_accepts_noncontiguous_input():
    grad = torch.randn((19, 11), device="cuda", dtype=torch.float32).T
    assert not grad.is_contiguous()
    expected = _zeropower_reuse_buffers(grad.clone(), (3.4445, -4.775, 2.0315), 5, 1e-7)
    actual = muon_ns(grad.clone(), (3.4445, -4.775, 2.0315), 5, 1e-7)
    assert torch.equal(actual, expected)


def test_muon_ns_rejects_cpu_input():
    with pytest.raises(ValueError, match="CUDA"):
        muon_ns(torch.ones((2, 2)), (3.4445, -4.775, 2.0315), 5, 1e-7)


@pytest.mark.parametrize("shape", [(512, 2048), (1536, 6144), (6144, 1536)])
@pytest.mark.parametrize("seed", [23, 31])
@pytest.mark.parametrize("ns_steps", [1, 5])
@skip_no_muon_ns
def test_direct_gram_preserves_wide_reduction_order(shape, seed, ns_steps):
    # A different Gram GEMM algorithm can change a few BF16 rounding decisions,
    # which later NS iterations amplify. Exercise both layouts and multiple seeds.
    torch.manual_seed(seed)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)
    module = get_module("muon_ns")
    expected = module.muon_ns(grad, (3.4445, -4.775, 2.0315), ns_steps, 1e-7, False)
    actual = module.muon_ns(grad, (3.4445, -4.775, 2.0315), ns_steps, 1e-7, True)
    assert torch.equal(actual, expected), (
        f"shape={shape}, seed={seed}, ns_steps={ns_steps}, "
        f"max_abs={(actual.float() - expected.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("eps", [1e-7, 0.1001])
@pytest.mark.parametrize("scale", [0.0, 1e-5, 1.0])
@skip_no_muon_ns
def test_fused_normalization_preserves_bf16_clamp_rounding(eps, scale):
    torch.manual_seed(7)
    grad = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16) * scale
    module = get_module("muon_ns")
    expected = module.muon_ns(grad.clone(), (3.4445, -4.775, 2.0315), 5, eps, False)
    actual = module.muon_ns(grad.clone(), (3.4445, -4.775, 2.0315), 5, eps, True)
    assert torch.equal(actual, expected)


@skip_no_muon_ns
def test_muon_ns_respects_disabled_bf16_reduced_precision():
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.manual_seed(23)
        grad = torch.randn((1024, 2048), device="cuda")
        expected = _zeropower_reuse_buffers(grad, (3.4445, -4.775, 2.0315), 5, 1e-7)
        actual = muon_ns(grad, (3.4445, -4.775, 2.0315), 5, 1e-7)
        assert torch.equal(actual, expected)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
