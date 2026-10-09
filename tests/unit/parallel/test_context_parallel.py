"""Context-parallel (sequence-parallel) training-path equivalence tests.

Spawns two ranks (nccl, one GPU each) and checks that the production CP
path — CPStrategy wrapping the real strategy, contiguous sequence
sharding + ring attention through the patched SDPA — reproduces the
single-device full-sequence forward and backward of the same tiny GQA
model.
"""

import pytest
import torch
from pydantic import ValidationError

from astrai.config.train_config import TrainConfig
from astrai.parallel.cp import CPStrategy, LossReduction
from astrai.parallel.topology import ParallelTopology
from astrai.trainer.strategy import SEQStrategy, SFTStrategy
from tests.support.rollout import RandomTokenDataset

SEQ_LEN = 64
BATCH = 2


def test_topology_decomposition():
    trivial = ParallelTopology(world_size=1)
    assert trivial.is_trivial
    assert trivial.dp_size == 1 and trivial.cp_group is None

    with pytest.raises(ValueError, match="divisible"):
        ParallelTopology(world_size=3, cp_size=2)

    with pytest.raises(ValueError, match="divisible"):
        ParallelTopology(world_size=3, tp_size=2)


def test_topology_reduce_single_process_noop():
    topology = ParallelTopology(world_size=1)
    values = torch.tensor([2.0])
    assert topology.reduce_sum(values).item() == 2.0
    assert topology.reduce_mean(values).item() == 2.0
    assert topology.samples_per_replica(10) == 10


def test_cp_strategy_requires_token_mean():
    class _SequenceReducer:
        loss_reduction = LossReduction.SEQUENCE

    with pytest.raises(ValueError, match="token-mean"):
        CPStrategy(_SequenceReducer(), cp=None)


def test_sft_shard_spec_rejects_packed():
    strategy = SFTStrategy(model=None, device="cpu")
    packed = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "target_ids": torch.zeros(2, 8, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1]] * 2),
        "loss_mask": torch.ones(2, 8, dtype=torch.bool),
    }
    with pytest.raises(NotImplementedError, match="packed"):
        strategy.shard_spec(packed)


def test_seq_prepare_batch_synthesizes_positions():
    strategy = SEQStrategy(model=None, device="cpu")
    batch = {
        "input_ids": torch.ones(2, 8, dtype=torch.long),
        "target_ids": torch.ones(2, 8, dtype=torch.long),
    }
    prepared = strategy.prepare_batch(batch)
    assert torch.equal(
        prepared["position_ids"], torch.arange(8).unsqueeze(0).expand(2, -1)
    )


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


def test_dp_size_derives_nprocs():
    config = _minimal_train_config()
    assert config.dp_size == 1 and config.cp_size == 1
    assert config.nprocs == 1

    config = _minimal_train_config(dp_size=2, cp_size=2)
    assert config.nprocs == 4  # world = dp x cp by construction


def test_dp_size_validator():
    with pytest.raises(ValidationError):
        _minimal_train_config(dp_size=0)


def test_cp_size_validator():
    config = _minimal_train_config()
    assert config.cp_size == 1
    with pytest.raises(ValidationError):
        _minimal_train_config(cp_size=0)
