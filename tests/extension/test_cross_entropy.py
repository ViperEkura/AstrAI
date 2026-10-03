import pytest
import torch
import torch.nn.functional as F

from astrai.extension.kernel.cross_entropy import cross_entropy, is_available

skip_no_cross_entropy = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="cross_entropy CUDA kernel unavailable",
)


@pytest.mark.parametrize("shape", [(7, 19), (33, 1031), (64, 100000)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("ignored", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@skip_no_cross_entropy
def test_cross_entropy_loss_and_backward(shape, dtype, ignored, reduction):
    torch.manual_seed(31)
    source = (torch.randn(shape, device="cuda") * 3).to(dtype)
    targets = torch.randint(shape[1], (shape[0],), device="cuda")
    if ignored:
        targets[::3] = -100
    baseline = source.clone().requires_grad_()
    candidate = source.clone().requires_grad_()
    expected = F.cross_entropy(baseline.float(), targets, reduction=reduction)
    actual = cross_entropy(candidate, targets, reduction=reduction)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)
    (expected * 0.37).backward()
    (actual * 0.37).backward()
    torch.testing.assert_close(
        candidate.grad,
        baseline.grad,
        rtol=0.01 if dtype == torch.bfloat16 else 0.002,
        atol=2e-7,
    )


@skip_no_cross_entropy
def test_cross_entropy_all_ignored_and_noncontiguous():
    source = torch.randn((19, 7), device="cuda", dtype=torch.bfloat16).T
    candidate = source.detach().requires_grad_()
    targets = torch.full((14,), -100, device="cuda", dtype=torch.int64)[::2]
    loss = cross_entropy(candidate, targets)
    assert torch.isnan(loss)
    loss.backward()
    assert torch.count_nonzero(candidate.grad).item() == 0


def test_cross_entropy_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA matrix"):
        cross_entropy(torch.randn(3, 4), torch.zeros(3, dtype=torch.int64))
