"""Tests for :func:`broadcast_state_dict` and distributed ``create_ref_model``.

Uses ``spawn_parallel_fn`` with the ``gloo`` backend to simulate a
multi-rank environment without requiring multiple GPUs.
"""

from types import SimpleNamespace

import torch

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel import get_rank, spawn_parallel_fn
from astrai.parallel.executor import broadcast_state_dict
from astrai.trainer.components import create_ref_model
from astrai.trainer.strategy import GRPOStrategy
from tests.support.models import make_rollout_config


def _broadcast_worker():
    for reuse_storage in (False, True):
        expected = {
            "layer.weight": torch.arange(32, dtype=torch.float32).view(4, 8),
            "layer.bias": torch.full((4,), -0.5),
        }
        state = (
            {key: value.clone() for key, value in expected.items()}
            if get_rank() == 0
            else {key: torch.zeros_like(value) for key, value in expected.items()}
            if reuse_storage
            else None
        )
        pointers = {key: value.data_ptr() for key, value in (state or {}).items()}
        received = broadcast_state_dict(state, src=0)
        assert set(received) == set(expected)
        for key, reference in expected.items():
            torch.testing.assert_close(received[key], reference, atol=0, rtol=0)
            assert received[key].device.type == "cpu"
            if key in pointers:
                assert received[key].data_ptr() == pointers[key]


def test_broadcast_values_and_compatible_storage():
    spawn_parallel_fn(_broadcast_worker, world_size=2, backend="gloo")


def _rank_zero_state_dict(model):
    return model.state_dict() if get_rank() == 0 else None


def _create_ref_model_worker():
    """Verify create_ref_model works when model_state_dict returns None on non-rank-0."""

    rank = get_rank()
    config = make_rollout_config()
    model = AutoRegressiveLM(config).to("cpu")
    # Give each rank distinct weights so we can verify broadcast overwrites them
    with torch.no_grad():
        for p in model.parameters():
            p.add_(float(rank))

    executor = SimpleNamespace(
        use_distributed=True, model_state_dict=_rank_zero_state_dict
    )

    ref = create_ref_model(
        model_fn=lambda: AutoRegressiveLM(config),
        executor=executor,
        model=model,
        device="cpu",
    )

    assert ref is not None, f"rank {rank}: ref model is None"
    # Broadcast rank-0's original weights for comparison
    if rank == 0:
        expected_sd = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        expected_sd = None
    expected_sd = broadcast_state_dict(expected_sd, src=0)

    ref_sd = ref.state_dict()
    for k in ref_sd:
        assert torch.equal(ref_sd[k], expected_sd[k]), f"rank {rank}: mismatch at {k}"
    # ref model should be frozen and in eval mode
    assert not ref.training
    for p in ref.parameters():
        assert not p.requires_grad


def test_create_ref_model_distributed():
    spawn_parallel_fn(_create_ref_model_worker, world_size=2, backend="gloo")


def _sync_old_model_worker():
    """Verify that sync_old_model broadcasts weights to all ranks."""
    rank = get_rank()
    config = make_rollout_config()
    model = AutoRegressiveLM(config).to("cpu")
    old_model = AutoRegressiveLM(config).to("cpu")
    ref_model = AutoRegressiveLM(config).to("cpu")

    # Give model rank-distinct weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(float(rank) * 10)

    strategy = GRPOStrategy(
        model=model,
        device="cpu",
        old_model=old_model,
        ref_model=ref_model,
        executor=SimpleNamespace(
            use_distributed=True, model_state_dict=_rank_zero_state_dict
        ),
    )

    # Capture rank-0's policy weights for comparison
    if rank == 0:
        expected = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        expected = None
    expected = broadcast_state_dict(expected, src=0)

    strategy.sync_old_model()

    old_sd = strategy.old_model.state_dict()
    for k in old_sd:
        assert torch.equal(old_sd[k], expected[k]), f"rank {rank}: mismatch at {k}"


def test_sync_old_model_distributed():
    spawn_parallel_fn(_sync_old_model_worker, world_size=2, backend="gloo")


def test_create_ref_model_single_process():
    """create_ref_model still works without an executor (explicit state_dict)."""
    config = make_rollout_config()
    model = AutoRegressiveLM(config)
    sd = model.state_dict()

    ref = create_ref_model(
        model_fn=lambda: AutoRegressiveLM(config),
        state_dict=sd,
        device="cpu",
    )
    assert ref is not None
    assert not ref.training
    for p in ref.parameters():
        assert not p.requires_grad
    for k in sd:
        assert torch.equal(ref.state_dict()[k], sd[k])
