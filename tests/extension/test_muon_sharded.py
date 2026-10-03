"""Parity checks for the local pieces of distributed Muon."""

import pytest
import torch

from astrai.extension.kernel.muon_sharded import is_available
from astrai.extension.loader import get_module

skip_no_muon_sharded = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="sharded Muon CUDA kernels are not available",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nesterov", [False, True])
@skip_no_muon_sharded
def test_prepare_and_normalize_match_torch(dtype, nesterov):
    torch.manual_seed(51)
    grad = torch.randn((64, 32), device="cuda", dtype=dtype)
    buffer = torch.randn_like(grad)
    reference_buffer = buffer.clone()
    reference_buffer.lerp_(grad, 0.05)
    reference_unscaled = (
        (grad.lerp(reference_buffer, 0.95) if nesterov else reference_buffer)
        .bfloat16()
        .clone()
    )
    reference = reference_unscaled.clone()
    reference.div_(reference.norm().clamp(min=1e-7))

    module = get_module("muon_ns")
    actual, partial_squares = module.prepare(grad, buffer, 0.95, nesterov)
    if dtype == torch.bfloat16:
        assert torch.equal(buffer, reference_buffer)
        assert torch.equal(actual, reference_unscaled)
    total = module.reduce_partials([partial_squares])
    torch.testing.assert_close(total[0], partial_squares.sum(), rtol=1e-6, atol=1e-5)
    module.normalize_(actual, partial_squares.sum(), 1e-7)
    torch.testing.assert_close(buffer, reference_buffer, rtol=0.01, atol=0.01)
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.01)


@skip_no_muon_sharded
def test_cublas_gram_and_ns_update_match_torch():
    torch.manual_seed(59)
    x = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16).T
    x.div_(x.norm())
    module = get_module("muon_ns")
    gram = torch.empty((32, 32), device="cuda", dtype=torch.bfloat16)
    module.gram_(x, gram)
    expected_gram = x @ x.T
    torch.testing.assert_close(gram, expected_gram, rtol=0.01, atol=1e-5)

    polynomial = torch.empty_like(gram)
    next_x = torch.empty_like(x)
    coefficients = (3.4445, -4.775, 2.0315)
    module.ns_update_(x, gram, polynomial, next_x, *coefficients)
    a, b, c = coefficients
    expected_polynomial = torch.addmm(gram, gram, gram, beta=b, alpha=c)
    expected_next_x = torch.addmm(x, expected_polynomial, x, beta=a, alpha=1)
    torch.testing.assert_close(polynomial, expected_polynomial, rtol=0.02, atol=0.005)
    torch.testing.assert_close(next_x, expected_next_x, rtol=0.02, atol=0.005)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@skip_no_muon_sharded
def test_finish_matches_torch(dtype):
    torch.manual_seed(53)
    param = torch.randn((64, 32), device="cuda", dtype=dtype)
    update = torch.randn((64, 32), device="cuda", dtype=torch.bfloat16)
    expected = param.clone()
    expected.mul_(1 - 1e-3 * 0.1)
    expected.add_(update, alpha=-5e-4)
    get_module("muon_ns").finish_(param, update, 1e-3, 0.1, 5e-4)
    torch.testing.assert_close(param, expected, rtol=0, atol=0)
