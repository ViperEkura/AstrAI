"""Explicit windows match one global batch, including optimizer state and LR."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from astrai.parallel.executor import DDPExecutor, NoneExecutor
from astrai.parallel.setup import find_free_port


def _compare_windows(rank, world_size, port):
    if world_size > 1:
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=timedelta(seconds=30),
        )
    try:
        torch.manual_seed(17)
        reference = nn.Linear(3, 2)
        initial = {
            name: value.clone() for name, value in reference.state_dict().items()
        }

        def make_model():
            model = nn.Linear(3, 2)
            model.load_state_dict(initial)
            return model

        def make_optimizer(model):
            return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        def make_scheduler(optimizer):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

        executor = (DDPExecutor if world_size > 1 else NoneExecutor)(grad_accum_steps=3)
        model, optimizer, scheduler = executor.prepare(
            make_model, make_optimizer, make_scheduler
        )
        reference_optimizer = make_optimizer(reference)
        reference_scheduler = make_scheduler(reference_optimizer)
        inputs = torch.arange(world_size * 18, dtype=torch.float).reshape(-1, 3) / 50
        targets = inputs[:, :2].cos()
        local_inputs, local_targets = (
            inputs.chunk(world_size)[rank],
            targets.chunk(world_size)[rank],
        )
        for _ in range(3):
            for index, (x, y) in enumerate(
                zip(local_inputs.chunk(3), local_targets.chunk(3))
            ):
                with executor.accumulate_explicit(model, sync=index == 2):
                    executor.backward((model(x) - y).square().mean() / 3)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            (reference(inputs) - targets).square().mean().backward()
            reference_optimizer.step()
            reference_optimizer.zero_grad()
            reference_scheduler.step()

            for name, actual in executor.model_state_dict(model).items():
                torch.testing.assert_close(actual, reference.state_dict()[name])
            actual_state = optimizer.state_dict()
            expected_state = reference_optimizer.state_dict()
            assert actual_state["param_groups"] == expected_state["param_groups"]
            for key in expected_state["state"]:
                torch.testing.assert_close(
                    actual_state["state"][key]["momentum_buffer"],
                    expected_state["state"][key]["momentum_buffer"],
                )
            assert scheduler.state_dict() == reference_scheduler.state_dict()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_explicit_windows_match_global_batch(world_size):
    if world_size == 1:
        _compare_windows(0, world_size, 0)
    else:
        mp.spawn(
            _compare_windows, args=(world_size, find_free_port()), nprocs=world_size
        )
