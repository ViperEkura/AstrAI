"""Context-parallel (sequence-parallel) training-path equivalence tests.

Spawns two ranks (nccl, one GPU each) and checks that the production CP
path — CPStrategy wrapping the real strategy, contiguous sequence
sharding + ring attention through the patched SDPA — reproduces the
single-device full-sequence forward and backward of the same tiny GQA
model.
"""

import pytest
import torch
import torch.distributed as dist

from astrai.config.train_config import TrainConfig
from astrai.dataset import RDSampler
from astrai.extension import ATTN_BACKEND, attn_backend
from astrai.parallel.cp import CPState, CPStrategy
from astrai.parallel.setup import spawn_parallel_fn
from astrai.parallel.topology import ParallelTopology
from astrai.trainer.strategy import SEQStrategy, SFTStrategy
from tests.support.capabilities import skip_lt2_cuda
from tests.support.models import make_seeded_gqa_model
from tests.support.rollout import RandomTokenDataset

SEQ_LEN = 64
BATCH = 2


def _build_gqa_model(device):
    """bf16 GQA model, head_dim 32: the ring path needs a flash/cuDNN
    SDPA kernel (CPState.shard restricts the selection), and flash
    requires a half-precision dtype."""
    return make_seeded_gqa_model(
        device, dtype=torch.bfloat16, max_position_embeddings=SEQ_LEN
    )


def _assert_shard_mechanics(cp_state, device):
    """cp.shard hands out the local slice and never restores it."""
    probe = torch.ones(BATCH, SEQ_LEN, device=device)
    with cp_state.shard([probe], [1]):
        local_len = probe.size(1)
        assert 0 < local_len < SEQ_LEN
    # no_restore_buffers: autograd's saved ids must not be resized back
    # under it, so the shard stays final after the context exits.
    assert probe.size(1) == local_len


def _assert_cp_matches_single_device(strategy, batch, topology):
    """The CPStrategy composition matches the stock strategy exactly.

    Forward: the reported request loss is the true global mean.  Backward:
    each rank holds grads of (local_sum * cp / global_count), so the sum
    over cp ranks is cp x the single-device mean-loss grads — normalize
    before comparing.  bf16 logits plus the ring's LSE merge both add
    noise; compare at percent-level rather than bitwise.
    """
    ref_out = strategy(batch)
    ref_loss = ref_out["metrics"]["task_loss"]
    strategy.model.zero_grad(set_to_none=True)
    ref_out["loss"].backward()
    ref_grads = {
        name: p.grad.detach().clone()
        for name, p in strategy.model.named_parameters()
        if p.grad is not None
    }
    strategy.model.zero_grad(set_to_none=True)

    cp_out = CPStrategy(strategy, CPState(topology))(batch)
    torch.testing.assert_close(
        cp_out["metrics"]["task_loss"], ref_loss, rtol=1e-2, atol=1e-2
    )

    (cp_out["loss"] / 2.0).backward()
    for name, ref_grad in ref_grads.items():
        grad = strategy.model.get_parameter(name).grad.detach().clone().float()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=topology.cp_group)
        torch.testing.assert_close(grad, ref_grad.float(), rtol=5e-2, atol=1e-2)


def _cp_equivalence_worker():
    device = torch.device("cuda", torch.cuda.current_device())
    topology = ParallelTopology(
        world_size=dist.get_world_size(), cp_size=2, device_type="cuda"
    )
    model = _build_gqa_model(device)

    torch.manual_seed(11)
    batch = {
        "input_ids": torch.randint(
            0, model.config.vocab_size, (BATCH, SEQ_LEN), device=device
        ),
        "target_ids": torch.randint(
            0, model.config.vocab_size, (BATCH, SEQ_LEN), device=device
        ),
    }

    with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
        _assert_shard_mechanics(CPState(topology), device)
        # Both ranks hold identical weights and inputs, so each verifies
        # the single-device reference locally.
        _assert_cp_matches_single_device(
            SEQStrategy(model, str(device)), batch, topology
        )


def _sft_cp_worker():
    device = torch.device("cuda", torch.cuda.current_device())
    topology = ParallelTopology(
        world_size=dist.get_world_size(), cp_size=2, device_type="cuda"
    )
    model = _build_gqa_model(device)

    torch.manual_seed(23)
    batch = {
        "input_ids": torch.randint(
            0, model.config.vocab_size, (BATCH, SEQ_LEN), device=device
        ),
        "target_ids": torch.randint(
            0, model.config.vocab_size, (BATCH, SEQ_LEN), device=device
        ),
        "position_ids": (
            torch.arange(SEQ_LEN, device=device).unsqueeze(0).expand(BATCH, -1)
        ),
        "loss_mask": torch.rand(BATCH, SEQ_LEN, device=device) > 0.3,
    }

    with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
        _assert_cp_matches_single_device(
            SFTStrategy(model, str(device)), batch, topology
        )


@skip_lt2_cuda
def test_cp2_matches_single_device():
    spawn_parallel_fn(_cp_equivalence_worker, world_size=2)


@skip_lt2_cuda
def test_cp2_sft_matches_single_device():
    spawn_parallel_fn(_sft_cp_worker, world_size=2)


def _cp_sampler_replication_worker():
    """dp=1 with cp=2: cp peers must draw identical batches.

    The dp dimension is a singleton group; a world-group fallback in the
    sampler would hand different batches to the two cp peers, whose
    gradients the reducer then silently averages across mismatched data.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    topology = ParallelTopology(
        world_size=dist.get_world_size(), cp_size=2, device_type="cuda"
    )
    # One real group per dimension once a mesh exists — singleton where
    # the dimension is inactive.
    assert dist.get_world_size(topology.dp_group) == 1
    assert dist.get_world_size(topology.cp_group) == 2
    assert dist.get_world_size(topology.tp_group) == 1

    sampler = RDSampler(list(range(10)), process_group=topology.dp_group)
    local = torch.tensor(list(sampler), dtype=torch.long, device=device)
    gathered = [torch.zeros_like(local) for _ in range(topology.world_size)]
    dist.all_gather(gathered, local)
    assert all(torch.equal(gathered[0], g) for g in gathered)


@skip_lt2_cuda
def test_cp_peers_sample_identical_batches():
    spawn_parallel_fn(_cp_sampler_replication_worker, world_size=2)


def _non_native_backend_worker():
    """shard refuses non-native backends before any sharding happens.

    flash_attn_func never passes through F.scaled_dot_product_attention,
    so the ring patch would silently never engage and every rank would
    attend only its local slice — wrong losses, no error.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    topology = ParallelTopology(
        world_size=dist.get_world_size(), cp_size=2, device_type="cuda"
    )
    cp_state = CPState(topology)
    probe = torch.ones(2, 8, device=device)
    with attn_backend(ATTN_BACKEND.FLASH):
        with pytest.raises(RuntimeError, match="torch-native"):
            with cp_state.shard([probe], [1]):
                pass


@skip_lt2_cuda
def test_shard_rejects_non_native_backend():
    spawn_parallel_fn(_non_native_backend_worker, world_size=2)


def _dp_reduce_worker():
    device = torch.device("cuda", torch.cuda.current_device())
    topology = ParallelTopology(world_size=dist.get_world_size(), device_type="cuda")
    # Trivial layout: no mesh, no subgroups — the world group plays the dp
    # role, so the reduce methods fall back to it.
    assert topology.dp_group is None

    values = torch.tensor([float(dist.get_rank()) + 1.0], device=device)
    summed = topology.reduce_sum(values.clone())
    assert summed.item() == pytest.approx(3.0)

    values = torch.tensor([float(dist.get_rank()) + 1.0], device=device)
    averaged = topology.reduce_mean(values.clone())
    assert averaged.item() == pytest.approx(1.5)


@skip_lt2_cuda
def test_dp_reduce_semantics():
    spawn_parallel_fn(_dp_reduce_worker, world_size=2)


def _minimal_train_config(**overrides):
    defaults = dict(
        strategy="seq",
        model_fn=lambda: torch.nn.Linear(2, 2),
        dataset=RandomTokenDataset(length=2),
        optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.0),
        scheduler_fn=lambda o: o,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)
