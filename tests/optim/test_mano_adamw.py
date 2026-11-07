import math

import pytest
import torch

from astrai.optimization.optimizer import Mano


def test_mano_one_step_projects_to_tangent_space():
    original = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    param = torch.nn.Parameter(original.clone())
    grad = torch.tensor([[4.0, -3.0], [1.0, 1.0]])
    param.grad = grad.clone()

    optimizer = Mano(
        [param], lr=0.1, momentum=0.0, nesterov=False, eps=1e-8, weight_decay=0.0
    )
    optimizer.step()

    dim = 0
    tangent = grad - (torch.sum(grad * original, dim=dim, keepdim=True) * original)
    direction = tangent / (torch.norm(tangent, p=2, dim=dim, keepdim=True) + 1e-8)
    adjusted_lr = 0.1 * 0.2 * math.sqrt(direction.shape[dim])
    expected = original - adjusted_lr * direction
    torch.testing.assert_close(param, expected)


def test_mano_alternates_projection_axis():
    """Step 0 projects along dim 0, step 1 along dim 1 (steps % 2)."""
    original = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    param = torch.nn.Parameter(original.clone())
    eps = 1e-8
    optimizer = Mano(
        [param], lr=0.1, momentum=0.0, nesterov=False, eps=eps, weight_decay=0.0
    )

    param.grad = torch.ones(2, 2)
    optimizer.step()
    after_step0 = param.detach().clone()

    param.grad = torch.ones(2, 2)
    optimizer.step()

    grad = torch.ones(2, 2)

    def projected(after, dim):
        tangent = grad - (torch.sum(grad * after, dim=dim, keepdim=True) * after)
        direction = tangent / (torch.norm(tangent, p=2, dim=dim, keepdim=True) + eps)
        adjusted_lr = 0.1 * 0.2 * math.sqrt(direction.shape[dim])
        return after - adjusted_lr * direction

    torch.testing.assert_close(param.detach(), projected(after_step0, dim=1))
    # The dim=1 result is distinct, so the assertion above pins the axis.
    assert not torch.allclose(param.detach(), projected(after_step0, dim=0))


def test_mano_rejects_non_2d_parameters():
    param = torch.nn.Parameter(torch.randn(3, 4, 5))
    with pytest.raises(ValueError, match="2D"):
        Mano([param])
