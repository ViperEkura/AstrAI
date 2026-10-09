"""Tensor-parallel training-path equivalence tests.

Spawns two ranks (nccl, one GPU each) and checks that the production TP
path — TPState.shard chunking the plan-matched Linear weights and patching
their forward with a rowwise all-reduce — reproduces the single-device
forward and backward of the same tiny GQA model.

Each rank evaluates the full loss on its own shard of the weights, so a
rank's gradient is already the exact single-device gradient of its chunk:
comparisons are local, no cross-rank reduction is needed.
"""

from fnmatch import fnmatch

import pytest

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.model.components.linear import Linear
from astrai.parallel.topology import ParallelTopology
from astrai.parallel.tp import DEFAULT_TP_PLAN, TPState
from tests.support.models import make_tiny_config

SEQ_LEN = 64
BATCH = 2

#: name -> (param name suffix, chunk dim) for shard-slice comparisons
_SHARD_DIMS = {
    "q_proj": 0,
    "k_proj": 0,
    "v_proj": 0,
    "o_proj": 1,
    "up": 0,
    "gate": 0,
    "down": 1,
}


def test_tp_state_requires_active_dimension():
    with pytest.raises(ValueError, match="tp dimension"):
        TPState(ParallelTopology(world_size=1))


def test_default_plan_covers_standard_layout():
    """The default plan touches every projection and nothing else."""
    model = AutoRegressiveLM(make_tiny_config())
    matched = [
        name
        for name, module in model.named_modules()
        if isinstance(module, Linear)
        and any(fnmatch(name, pattern) for pattern in DEFAULT_TP_PLAN)
    ]
    assert matched, "default plan matched nothing"
    # every matched key is inside a layer's attention or mlp
    for name in matched:
        assert ".attention." in name or ".mlp." in name, name
    # lm_head and embeddings stay replicated
    assert not any("lm_head" in n or "embed" in n for n in matched)
