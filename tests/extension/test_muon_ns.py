"""Numerics, layout selection, and Graph replay for Muon's NS kernels."""

import pytest
import torch

from astrai.extension.backend.newton_schulz import newton_schulz
from astrai.extension.kernel.symmetric import (
    is_available,
    symm_out,
    syrk_out,
)
from astrai.extension.policy import symmetric as plan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="Muon NS CUDA kernels are unavailable",
)

COEFFICIENTS = (3.4445, -4.775, 2.0315)
PRODUCTION_SHAPES = [(256, 1536), (1536, 1536), (1536, 6912), (6912, 1536)]


@pytest.mark.parametrize("shape", [(64, 128), (256, 512), (1536, 6912)])
def test_syrk_and_polynomial_match_normalized_torch(shape):
    torch.manual_seed(23)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x.div_(x.norm().clamp_min(1e-7))
    gram = torch.empty((shape[0], shape[0]), device="cuda", dtype=torch.bfloat16)
    polynomial = torch.empty_like(gram)
    syrk_out(x, gram)
    syrk_out(gram, polynomial, addend=gram, beta=COEFFICIENTS[1], alpha=COEFFICIENTS[2])
    expected_gram = x @ x.T
    expected_polynomial = torch.addmm(
        expected_gram,
        expected_gram,
        expected_gram,
        beta=COEFFICIENTS[1],
        alpha=COEFFICIENTS[2],
    )
    assert torch.equal(gram, gram.T)
    assert torch.equal(polynomial, polynomial.T)
    torch.testing.assert_close(gram, expected_gram, atol=0.002, rtol=0.01)
    torch.testing.assert_close(polynomial, expected_polynomial, atol=0.002, rtol=0.01)


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_newton_schulz_production_shapes_have_bounded_error(shape, dtype):
    torch.manual_seed(17)
    gradient = torch.randn(shape, device="cuda", dtype=dtype)
    expected = newton_schulz(gradient.clone(), COEFFICIENTS, 5, 1e-7)
    actual = newton_schulz(gradient.clone(), COEFFICIENTS, 5, 1e-7, backend="auto")
    difference = (actual.float() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert actual.is_contiguous()
    assert difference.max().item() <= (0.005 if shape[0] == 256 else 0.002)
    assert (difference.norm() / expected.float().norm()).item() <= 0.01


def test_unmeasured_shape_keeps_torch_path():
    gradient = torch.randn((192, 384), device="cuda", dtype=torch.float32)
    expected = newton_schulz(gradient, COEFFICIENTS, 5, 1e-7)
    actual = newton_schulz(gradient, COEFFICIENTS, 5, 1e-7, backend="auto")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES)
def test_newton_schulz_graph_replay_uses_current_gradient(shape):
    torch.manual_seed(37)
    gradient = torch.randn(shape, device="cuda", dtype=torch.float32)
    newton_schulz(gradient, COEFFICIENTS, 5, 1e-7, backend="auto")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = newton_schulz(gradient, COEFFICIENTS, 5, 1e-7, backend="auto")
    gradient.copy_(torch.randn_like(gradient))
    graph.replay()
    expected = newton_schulz(gradient, COEFFICIENTS, 5, 1e-7, backend="auto")
    assert torch.equal(output, expected)


def test_symm_matches_torch():
    x = torch.randn((256, 1536), device="cuda", dtype=torch.bfloat16)
    x.div_(x.norm())
    symmetric = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
    symmetric = (symmetric + symmetric.T) * 0.5
    symmetric.div_(symmetric.norm())
    output = torch.empty_like(x)
    symm_out(symmetric, x, output, addend=x, beta=COEFFICIENTS[0])
    expected = torch.addmm(x, symmetric, x, beta=COEFFICIENTS[0])
    torch.testing.assert_close(output, expected, atol=0.0001, rtol=0.01)


def test_kernels_reject_overlapping_storage():
    storage = torch.empty(12288, device="cuda", dtype=torch.bfloat16)
    x = storage[:8192].view(64, 128)
    overlapping = storage[4096:8192].view(64, 64)
    with pytest.raises(RuntimeError, match="must not alias"):
        syrk_out(x, overlapping)
    square = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="must not alias"):
        syrk_out(square, square, addend=square, beta=-4.775, alpha=2.0315)
    with pytest.raises(RuntimeError, match="must not alias"):
        symm_out(square, x, x, addend=x, beta=3.4445)
    with pytest.raises(RuntimeError, match="multiples of 64"):
        syrk_out(x[:63], torch.empty((63, 63), device="cuda", dtype=x.dtype))


def test_zero_gradient_is_finite():
    gradient = torch.zeros((6912, 1536), device="cuda", dtype=torch.bfloat16)
    actual = newton_schulz(gradient, COEFFICIENTS, 5, 1e-7, backend="auto")
    assert torch.count_nonzero(actual).item() == 0


@pytest.mark.parametrize("steps", [0, 1, 2, 4, 5])
def test_layout_transition_preserves_normalized_input_for_each_step_count(steps):
    torch.manual_seed(41)
    gradient = torch.randn(320, 192, device="cuda", dtype=torch.bfloat16)
    reference_input = gradient.clone()
    expected = newton_schulz(reference_input, COEFFICIENTS, steps)
    normalized = gradient.clone()
    normalized.div_(gradient.T.norm().clamp_min(1e-7))
    cc = sum(a * b for a, b in zip(torch.cuda.get_device_capability(), (10, 1)))
    row = dict(
        operation="syrk",
        cc=cc,
        rows=192,
        cols=320,
        backend="cuda",
        tile="64x64x32_W16x32_S2",
    )
    with plan.override([row]):
        actual = newton_schulz(gradient, COEFFICIENTS, steps, backend="auto")
    assert actual.is_contiguous()
    assert torch.equal(gradient, normalized)
    relative = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative.item() <= 0.01
