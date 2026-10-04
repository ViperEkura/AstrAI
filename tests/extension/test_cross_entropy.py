"""Independent guard: CE availability must not change other kernel tests."""

import pytest
import torch
import torch.nn.functional as F

from astrai.extension.kernel.cross_entropy import (
    cross_entropy,
    is_available,
    linear_cross_entropy,
)

skip_no_ce = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="CE CUDA kernel not built",
)


@skip_no_ce
@pytest.mark.parametrize("vocab", [33, 1000, 100000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("smoothing", [0.0, 0.1, 1.0])
def test_ce(vocab, dtype, smoothing):
    torch.manual_seed(31)
    logits = (torch.randn(11, vocab, device="cuda") * 3).to(dtype).requires_grad_()
    targets = torch.randint(vocab, (11,), device="cuda")
    targets[::3] = -100
    actual = cross_entropy(logits, targets, label_smoothing=smoothing)
    expected = F.cross_entropy(
        logits.float(), targets, reduction="sum", label_smoothing=smoothing
    )
    grad = torch.autograd.grad(actual / 7, logits)[0]
    ref = torch.autograd.grad(expected / 7, logits)[0]
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-6)
    torch.testing.assert_close(
        grad, ref, atol=2e-6 if dtype == torch.float32 else 3e-5, rtol=0.01
    )


@skip_no_ce
@pytest.mark.parametrize("chunk", [1, 16, 128, 256, 512, 1024])
@pytest.mark.parametrize("smoothing", [0.0, 0.1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_linear_gradients(chunk, smoothing, dtype):
    torch.manual_seed(8)
    x = torch.randn(137, 64, device="cuda", dtype=dtype, requires_grad=True)
    w = (torch.randn(257, 64, device="cuda", dtype=dtype) * 0.02).requires_grad_()
    y = torch.randint(257, (137,), device="cuda")
    y[::4] = -100
    loss = linear_cross_entropy(x, w, y, label_smoothing=smoothing, chunk_size=chunk)
    ref = F.cross_entropy(
        F.linear(x, w).float(), y, reduction="sum", label_smoothing=smoothing
    )
    grads = torch.autograd.grad(loss / 103, (x, w), retain_graph=True)
    again = torch.autograd.grad(loss / 103, (x, w))
    refs = torch.autograd.grad(ref / 103, (x, w))
    torch.testing.assert_close(loss, ref, rtol=3e-5, atol=1e-3)
    for actual, repeated, expected in zip(grads, again, refs):
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        relative_l2 = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm()
        assert relative_l2 < (0.008 if dtype == torch.bfloat16 else 1e-5)
        torch.testing.assert_close(
            actual, expected, rtol=0.03, atol=3e-4 if dtype == torch.bfloat16 else 1e-7
        )


@skip_no_ce
@pytest.mark.parametrize("linear", [False, True])
def test_all_masked(linear):
    x = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(99, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = torch.full((7,), -100, device="cuda")
    loss = linear_cross_entropy(x, w, y) if linear else cross_entropy(x @ w.T, y)
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(x.grad) == torch.count_nonzero(w.grad) == 0


@skip_no_ce
def test_noncontiguous_frozen_and_autocast():
    x = torch.randn(17, 32, device="cuda", requires_grad=True)
    w = torch.randn(32, 123, device="cuda").T
    y = torch.randint(123, (34,), device="cuda")[::2]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = linear_cross_entropy(x, w, y, chunk_size=8)
    loss.backward()
    assert x.grad.dtype == torch.float32 and torch.isfinite(x.grad).all()
    with torch.no_grad():
        assert torch.isfinite(linear_cross_entropy(x, w, y))


def test_cpu_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        cross_entropy(torch.ones(2, 3), torch.zeros(2, dtype=torch.long))


@skip_no_ce
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_multiple_vocabulary_tiles(dtype):
    torch.manual_seed(41)
    x = torch.randn(37, 64, device="cuda", dtype=dtype, requires_grad=True)
    w = (torch.randn(17003, 64, device="cuda", dtype=dtype) * 0.02).requires_grad_()
    y = torch.randint(17003, (37,), device="cuda")
    y[0], y[1], y[2] = 17002, 16384, -100
    actual = linear_cross_entropy(x, w, y, label_smoothing=0.1, chunk_size=16)
    expected = F.cross_entropy(
        F.linear(x, w).float(), y, reduction="sum", label_smoothing=0.1
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=1e-3)
    for a, b in zip(
        torch.autograd.grad(actual / 36, (x, w)),
        torch.autograd.grad(expected / 36, (x, w)),
    ):
        relative_l2 = (a.float() - b.float()).norm() / b.float().norm()
        assert relative_l2 < (0.008 if dtype == torch.bfloat16 else 1e-5)
