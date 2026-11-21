"""CE-capable LM head retains ordinary linear behavior and checkpoint keys."""

import pytest
import torch
import torch.nn.functional as F

from astrai.extension import is_available, op_backend
from astrai.model.components.linear import Linear, LinearCrossEntropyHead
from tests.support.capabilities import skip_no_cuda


@pytest.mark.parametrize("shape", [(6, 8), (2, 3, 8), (2, 3, 4, 8)])
def test_lm_head_cpu_ce_fallback_matches_projection(shape):
    torch.manual_seed(41)
    head = LinearCrossEntropyHead(8, 13)
    hidden = torch.randn(*shape, requires_grad=True)
    targets = torch.randint(0, 13, shape[:-1])
    actual = head(hidden, targets=targets, label_smoothing=0.1)
    expected = F.cross_entropy(
        head(hidden).reshape(-1, 13),
        targets.reshape(-1),
        label_smoothing=0.1,
        reduction="sum",
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert head.weight.grad is not None
    assert set(head.state_dict()) == {"weight"}
    with pytest.raises(ValueError, match="targets must match"):
        head(hidden, targets=targets[..., :1])


@pytest.mark.cuda
@skip_no_cuda
def test_native_bf16_projection_bias_and_gradients_match_fp32():
    if not is_available("gemm") or not torch.cuda.is_bf16_supported():
        pytest.skip("Native BF16 GEMM is required")

    torch.manual_seed(3407)
    layer = Linear(128, 256, bias=True).to("cuda", torch.bfloat16)
    with torch.no_grad():
        layer.bias.normal_()
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with op_backend(linear="cuda"):
        output = layer(x)
    upstream = torch.randn_like(output)
    grads = torch.autograd.grad(output, (x, layer.weight, layer.bias), upstream)

    operands = tuple(
        t.detach().float().requires_grad_() for t in (x, layer.weight, layer.bias)
    )
    reference = F.linear(*operands)
    reference_grads = torch.autograd.grad(reference, operands, upstream.float())

    torch.testing.assert_close(output, reference.bfloat16(), atol=4e-3, rtol=4e-3)
    for actual, expected in zip(grads, reference_grads):
        error = (actual.float() - expected).norm() / expected.norm()
        assert error < 0.005
