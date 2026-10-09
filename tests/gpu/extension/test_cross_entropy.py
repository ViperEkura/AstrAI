"""Independent guard: CE availability must not change other kernel tests."""

import pytest
import torch
import torch.nn.functional as F

from astrai.extension.kernel.cross_entropy import (
    is_available,
    linear_cross_entropy,
)

skip_no_ce = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="CE CUDA kernel not built",
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
def test_all_masked():
    x = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(99, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = torch.full((7,), -100, device="cuda")
    loss = linear_cross_entropy(x, w, y)
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


@skip_no_ce
@pytest.mark.parametrize(
    "input_grad,weight_grad", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_precomputed_gradients_preserve_retain_graph_and_scaling(
    input_grad, weight_grad, dtype
):
    torch.manual_seed(17)
    x = torch.randn(65, 64, device="cuda", dtype=dtype, requires_grad=input_grad)
    w = (torch.randn(257, 64, device="cuda", dtype=dtype) * 0.02).requires_grad_(
        weight_grad
    )
    y = torch.randint(257, (65,), device="cuda")
    y[::3] = -100
    loss = linear_cross_entropy(x, w, y, label_smoothing=0.1, chunk_size=32)
    reference = F.cross_entropy(
        F.linear(x, w).float(), y, reduction="sum", label_smoothing=0.1
    )
    active = tuple(t for t in (x, w) if t.requires_grad)
    repeated = None
    for scale in (1e-5, -0.25, 0.0, 1e-5):
        actual = torch.autograd.grad(loss * scale, active, retain_graph=True)
        expected = torch.autograd.grad(reference * scale, active, retain_graph=True)
        for grad, ref in zip(actual, expected):
            if scale == 0.0:
                assert torch.count_nonzero(grad) == 0
            else:
                relative_l2 = (grad.float() - ref.float()).norm() / ref.float().norm()
                assert relative_l2 < (0.008 if dtype == torch.bfloat16 else 1e-5)
        if scale == 1e-5:
            if repeated is not None:
                for grad, previous in zip(actual, repeated):
                    torch.testing.assert_close(grad, previous, rtol=0, atol=0)
            repeated = actual


@skip_no_ce
def test_precomputed_large_vocabulary_head():
    torch.manual_seed(19)
    x = torch.randn(513, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (
        torch.randn(100003, 1536, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    y = torch.randint(100003, (513,), device="cuda")
    y[::2] = -100
    loss = linear_cross_entropy(x, w, y, chunk_size=256)
    reference = F.cross_entropy(F.linear(x, w).float(), y, reduction="sum")
    torch.testing.assert_close(loss, reference, rtol=3e-5, atol=1e-3)
    for grad, ref in zip(
        torch.autograd.grad(loss / 257, (x, w)),
        torch.autograd.grad(reference / 257, (x, w)),
    ):
        error = (grad.float() - ref.float()).norm() / ref.float().norm()
        assert error < 0.008


@skip_no_ce
def test_precomputed_fp16_scales_before_casting_projected_gradients():
    x = torch.full(
        (2048, 16), 1000.0, device="cuda", dtype=torch.float16, requires_grad=True
    )
    w = torch.zeros(33, 16, device="cuda", dtype=torch.float16, requires_grad=True)
    y = torch.zeros(2048, device="cuda", dtype=torch.long)
    loss = linear_cross_entropy(x, w, y, chunk_size=256) / 2048
    reference = F.cross_entropy(F.linear(x.float(), w.float()), y)
    actual = torch.autograd.grad(loss, (x, w))
    expected = torch.autograd.grad(reference, (x, w))
    for grad, ref in zip(actual, expected):
        assert torch.isfinite(grad).all()
        torch.testing.assert_close(grad, ref, rtol=0.001, atol=0.01)
