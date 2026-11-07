import math
from copy import deepcopy

import pytest
import torch

from astrai.optimization.scheduler import (
    CosineScheduler,
    SchedulerFactory,
    SGDRScheduler,
)


def _stepped_lrs(scheduler, optimizer, n_steps):
    """Return the lr after construction plus each of *n_steps* steps."""
    lrs = list(scheduler.get_last_lr())
    for _ in range(n_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    return lrs


def test_cosine_scheduler_warms_up_then_decays_to_floor():
    """lr ramps linearly to base_lr during warmup, cosine-decays after it,
    and never drops below min_rate * base_lr."""
    base_lr = 0.001
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = SchedulerFactory.create(
        "cosine", optimizer, warmup_steps=2, lr_decay_steps=4, min_rate=0.1
    )

    assert isinstance(scheduler, CosineScheduler)
    lrs = _stepped_lrs(scheduler, optimizer, n_steps=7)

    assert lrs[0] == pytest.approx(0.1 * base_lr)  # warmup starts at the floor
    assert lrs[1] == pytest.approx(0.5 * base_lr)  # halfway through warmup
    assert lrs[2] == pytest.approx(base_lr)  # warmup complete
    expected_mid = base_lr * 0.5 * (1.0 + math.cos(math.pi * 0.25))
    assert lrs[3] == pytest.approx(expected_mid)  # quarter into decay
    assert lrs[5] > 0.1 * base_lr  # 3/4 into decay: not clamped yet
    assert lrs[6] == pytest.approx(0.1 * base_lr)  # clamped at min_rate floor
    assert lrs[7] == pytest.approx(0.1 * base_lr)  # stays at the floor
    assert all(lr >= 0.1 * base_lr - 1e-12 for lr in lrs)


def test_cosine_scheduler_decays_to_zero_with_min_rate_zero():
    """min_rate=0 must reach exactly 0.0 at the end of decay, not NaN."""
    base_lr = 0.001
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = SchedulerFactory.create(
        "cosine", optimizer, warmup_steps=1, lr_decay_steps=9, min_rate=0.0
    )

    lrs = _stepped_lrs(scheduler, optimizer, n_steps=11)

    assert lrs[10] == 0.0
    assert lrs[11] == 0.0
    assert all(math.isfinite(lr) for lr in lrs)


def test_sgdr_scheduler_restarts_each_cycle():
    """lr anneals within a cycle, then jumps back to base_lr on restart."""
    base_lr = 0.001
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = SchedulerFactory.create(
        "sgdr", optimizer, warmup_steps=2, cycle_length=4, t_mult=1, min_rate=0.1
    )

    assert isinstance(scheduler, SGDRScheduler)
    lrs = _stepped_lrs(scheduler, optimizer, n_steps=7)

    assert lrs[2] == pytest.approx(base_lr)  # cycle start
    expected_mid = base_lr * (0.1 + 0.9 * 0.5)  # halfway through the cycle
    assert lrs[4] == pytest.approx(expected_mid)
    assert lrs[5] < lrs[4]  # still annealing at the cycle end
    assert lrs[6] == pytest.approx(base_lr)  # restart: back to full lr


@pytest.mark.parametrize(
    "name, params",
    [
        ("cosine", {"warmup_steps": 2, "lr_decay_steps": 8, "min_rate": 0.1}),
        ("sgdr", {"warmup_steps": 2, "cycle_length": 4, "min_rate": 0.1, "t_mult": 2}),
        (
            "wsd",
            {"warmup_steps": 2, "stable_steps": 4, "decay_steps": 4, "min_rate": 0.1},
        ),
    ],
)
def test_schedule_factory_state_persistence(name, params):
    """Complete schedule state resumes without consuming the supplied dictionary."""
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = SchedulerFactory.create(name, optimizer, **params)
    _stepped_lrs(scheduler, optimizer, n_steps=5)
    state_dict = deepcopy(scheduler.state_dict())
    saved_state = deepcopy(state_dict)
    assert all(state_dict[key] == value for key, value in params.items())

    new_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    resumed = SchedulerFactory.create(name, new_optimizer, **{key: 1 for key in params})
    new_optimizer.load_state_dict(optimizer.state_dict())
    for _ in range(2):
        resumed.load_state_dict(state_dict)
        assert state_dict == saved_state
        assert resumed.state_dict() == saved_state

    for key in list(params) + (["total_steps"] if name != "sgdr" else []):
        incomplete = {k: v for k, v in saved_state.items() if k != key}
        with pytest.raises(KeyError, match=key):
            resumed.load_state_dict(incomplete)
        assert resumed.state_dict() == saved_state

    assert _stepped_lrs(resumed, new_optimizer, 8) == pytest.approx(
        _stepped_lrs(scheduler, optimizer, 8)
    )
