"""Tests for :func:`broadcast_state_dict` and distributed ``create_ref_model``.

Uses ``spawn_parallel_fn`` with the ``gloo`` backend to simulate a
multi-rank environment without requiring multiple GPUs.
"""

import torch

from astrai.model.transformer import AutoRegressiveLM
from astrai.parallel import get_rank, spawn_parallel_fn
from astrai.parallel.executor import broadcast_state_dict, create_ref_model
from astrai.trainer.strategy import GRPOStrategy
from tests.helpers import make_rollout_config


def _broadcast_worker():
    """Rank-0 builds a state_dict; all ranks verify they receive it."""
    rank = get_rank()

    if rank == 0:
        sd = {
            "layer.weight": torch.randn(4, 8),
            "layer.bias": torch.randn(4),
        }
        expected = {k: v.clone() for k, v in sd.items()}
    else:
        sd = None
        expected = None

    received = broadcast_state_dict(sd, src=0)

    assert received is not None, f"rank {rank}: received None"
    assert set(received.keys()) == {"layer.weight", "layer.bias"}
    if rank == 0:
        # rank-0 already had the data
        for k in received:
            assert torch.equal(received[k], expected[k])
    # tensors preserve the source device (cpu here since gloo test)
    for k, v in received.items():
        assert v.device.type == "cpu", f"rank {rank}: {k} on {v.device}"


def test_broadcast_state_dict():
    spawn_parallel_fn(_broadcast_worker, world_size=2, backend="gloo")


def _broadcast_into_local_worker():
    """Non-source ranks retain compatible rank-local tensor storage."""
    rank = get_rank()
    state_dict = {
        "layer.weight": torch.full((4, 8), float(rank)),
        "layer.bias": torch.full((4,), float(rank)),
    }
    local_pointers = {key: value.data_ptr() for key, value in state_dict.items()}

    received = broadcast_state_dict(state_dict, src=0)

    assert received is not None
    for key, value in received.items():
        assert value.data_ptr() == local_pointers[key]
        assert torch.equal(value, torch.zeros_like(value))


def test_broadcast_state_dict_reuses_compatible_local_tensors():
    spawn_parallel_fn(_broadcast_into_local_worker, world_size=2, backend="gloo")


def _create_ref_model_worker():
    """Verify create_ref_model works when unwrap_model returns None on non-rank-0."""

    class FakeFSDPExecutor:
        """Simulates FSDP: unwrap_model returns state_dict on rank-0, None elsewhere."""

        use_distributed = True

        def unwrap_model(self, model):
            if get_rank() == 0:
                return model.state_dict()
            return None

    rank = get_rank()
    config = make_rollout_config()
    model = AutoRegressiveLM(config).to("cpu")
    # Give each rank distinct weights so we can verify broadcast overwrites them
    with torch.no_grad():
        for p in model.parameters():
            p.add_(float(rank))

    executor = FakeFSDPExecutor()

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

    class _DistExecutor:
        use_distributed = True

        def unwrap_model(self, m):
            if get_rank() == 0:
                return m.state_dict()
            return None

    strategy = GRPOStrategy(
        model=model,
        device="cpu",
        old_model=old_model,
        ref_model=ref_model,
        executor=_DistExecutor(),
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


def test_broadcast_state_dict_single_process():
    """When dist is not initialized, broadcast_state_dict is a no-op."""
    sd = {"w": torch.randn(3, 3), "b": torch.randn(3)}
    result = broadcast_state_dict(sd)
    assert result is sd


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
