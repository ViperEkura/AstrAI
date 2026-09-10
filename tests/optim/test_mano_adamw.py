import math
from copy import deepcopy

import pytest
import torch

from astrai.optim import Mano, ManoAdamW, OptimizerFactory
from tests.helpers import make_tiny_config


def _set_constant_grads(model, value):
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.full_like(param, value)


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


def test_factory_registers_mano():
    assert "mano_adamw" in OptimizerFactory.list_registered()
    from astrai.model import AutoRegressiveLM

    model = AutoRegressiveLM(make_tiny_config())
    optimizer = OptimizerFactory.create("mano_adamw", model, lr=3e-4)
    assert isinstance(optimizer, ManoAdamW)


def test_mano_adamw_runs_closure_once():
    from astrai.model import AutoRegressiveLM

    model = AutoRegressiveLM(make_tiny_config())
    optimizer = ManoAdamW(model)
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        return torch.tensor(1.0, requires_grad=True)

    loss = optimizer.step(closure)
    assert calls == 1
    assert loss.item() == 1.0


def test_mano_adamw_resume_matches_uninterrupted():
    from astrai.model import AutoRegressiveLM
    from astrai.trainer.schedule import SchedulerFactory

    torch.manual_seed(7)
    model_a = AutoRegressiveLM(make_tiny_config())
    optimizer_a = ManoAdamW(model_a, lr=3e-4)
    scheduler_a = SchedulerFactory.create(
        "cosine", optimizer_a, warmup_steps=2, lr_decay_steps=4, min_rate=0.1
    )

    _set_constant_grads(model_a, 0.125)
    optimizer_a.step()
    scheduler_a.step()
    model_state = {key: value.clone() for key, value in model_a.state_dict().items()}
    optimizer_state = deepcopy(optimizer_a.state_dict())
    scheduler_state = deepcopy(scheduler_a.state_dict())

    model_b = AutoRegressiveLM(make_tiny_config())
    model_b.load_state_dict(model_state)
    optimizer_b = ManoAdamW(model_b, lr=3e-4)
    scheduler_b = SchedulerFactory.create(
        "cosine", optimizer_b, warmup_steps=2, lr_decay_steps=4, min_rate=0.1
    )
    optimizer_b.load_state_dict(optimizer_state)
    scheduler_b.load_state_dict(scheduler_state)

    _set_constant_grads(model_a, -0.25)
    _set_constant_grads(model_b, -0.25)
    optimizer_a.step()
    optimizer_b.step()
    scheduler_a.step()
    scheduler_b.step()

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(param_a, param_b)
    assert scheduler_a.get_last_lr() == pytest.approx(scheduler_b.get_last_lr())
