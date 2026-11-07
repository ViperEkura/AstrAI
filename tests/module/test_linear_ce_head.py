"""CE-capable LM head retains ordinary linear behavior and checkpoint keys."""

import pytest
import torch
import torch.nn.functional as F

from astrai.model.components.linear import LinearCrossEntropyHead


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
