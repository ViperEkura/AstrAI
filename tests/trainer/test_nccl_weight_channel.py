"""Real five-GPU checks for the rollout-only NCCL weight channel."""

import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

from astrai.trainer.rollout.nccl_transport import NCCLWeightChannel


class _MixedState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 4))
        self.register_buffer("counter", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("transposed", torch.zeros(2, 3).T)


def _state_values(model):
    return (
        model.weight.detach().cpu().tolist(),
        model.counter.item(),
        model.transposed.cpu().tolist(),
    )


def _read(conn):
    assert conn.poll(30), "rollout NCCL worker did not reply"
    return conn.recv()


def _receiver(rank, port, layout, conn):
    torch.cuda.set_device(rank)
    model = _MixedState().to(f"cuda:{rank}")
    channel = NCCLWeightChannel(model, 0, expected_layout=layout)
    try:
        conn.send("ready")
        channel.connect(rank=rank, world_size=5, port=port, timeout_s=30)
        channel.broadcast()
        conn.send(_state_values(model))
        assert _read(conn) == "next"
        channel.broadcast()
        conn.send(_state_values(model))
        assert not dist.is_initialized()
    finally:
        channel.close()
        conn.close()


@pytest.mark.integration
@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="five CUDA devices required")
def test_nccl_channel_broadcasts_mixed_and_noncontiguous_state():
    torch.cuda.set_device(0)
    model = _MixedState().to("cuda:0")
    with torch.no_grad():
        model.weight.fill_(1.25)
        model.counter.fill_(7)
        model.transposed.fill_(3.5)
    channel = NCCLWeightChannel(model, 0)
    port = channel.prepare_rendezvous(5, 30)
    context = mp.get_context("spawn")
    processes = []
    pipes = []
    failed = True
    try:
        for rank in range(1, 5):
            parent, child = context.Pipe()
            process = context.Process(
                target=_receiver, args=(rank, port, channel.layout, child)
            )
            process.start()
            child.close()
            pipes.append(parent)
            processes.append(process)
        assert [_read(conn) for conn in pipes] == ["ready"] * 4
        channel.connect(rank=0, world_size=5, port=port, timeout_s=30)
        channel.broadcast()
        assert [_read(conn) for conn in pipes] == [_state_values(model)] * 4

        with torch.no_grad():
            model.weight.add_(2)
            model.counter.add_(1)
            model.transposed.add_(4)
        channel.mark_committed(1)
        for conn in pipes:
            conn.send("next")
        channel.broadcast()
        assert [_read(conn) for conn in pipes] == [_state_values(model)] * 4
        assert not dist.is_initialized()
        failed = False
    finally:
        if failed:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        channel.close(force=failed)
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for conn in pipes:
            conn.close()
    assert all(process.exitcode == 0 for process in processes)
