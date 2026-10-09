import math

import pytest
import torch

from astrai.trainer.schedule import CosineScheduler, SchedulerFactory, SGDRScheduler


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


def test_schedule_factory_state_persistence():
    """Test scheduler state persistence (save/load)"""

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Create scheduler directly with parameters
    warmup_steps = 100
    total_steps = 1000
    min_rate = 0.1
    lr_decay_steps = total_steps - warmup_steps
    scheduler = SchedulerFactory.create(
        "cosine",
        optimizer,
        warmup_steps=warmup_steps,
        lr_decay_steps=lr_decay_steps,
        min_rate=min_rate,
    )

    # Take a few steps
    for _ in range(5):
        optimizer.step()
        scheduler.step()

    # Save state
    state_dict = scheduler.state_dict()

    # Create new scheduler with same parameters
    new_scheduler = SchedulerFactory.create(
        "cosine",
        optimizer,
        warmup_steps=warmup_steps,
        lr_decay_steps=lr_decay_steps,
        min_rate=min_rate,
    )
    new_scheduler.load_state_dict(state_dict)

    # Verify states match
    assert scheduler.last_epoch == new_scheduler.last_epoch
    assert scheduler.get_last_lr() == new_scheduler.get_last_lr()
