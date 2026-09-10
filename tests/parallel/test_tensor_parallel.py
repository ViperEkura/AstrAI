"""Tensor-parallel training-path equivalence tests.

Spawns two ranks (nccl, one GPU each) and checks that the production TP
path — TPState.shard chunking the plan-matched Linear weights and patching
their forward with a rowwise all-reduce — reproduces the single-device
forward and backward of the same tiny GQA model.

Each rank evaluates the full loss on its own shard of the weights, so a
rank's gradient is already the exact single-device gradient of its chunk:
comparisons are local, no cross-rank reduction is needed.
"""

import pytest
import torch
import torch.distributed as dist

from astrai.model.transformer import AutoRegressiveLM
from astrai.parallel.topology import ParallelTopology
from astrai.parallel.tp import DEFAULT_TP_PLAN, TPState
from tests.conftest import skip_lt2_cuda
from tests.helpers import make_tiny_config

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


def _build_model(device):
    torch.manual_seed(3407)
    cfg = make_tiny_config(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=SEQ_LEN,
    )
    model = AutoRegressiveLM(cfg).to(device=device)
    model.train()
    return model


def _reference(model, input_ids, target_ids):
    logits = model(input_ids=input_ids)["logits"]
    loss = (
        torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_ids.flatten(),
            reduction="sum",
        )
        / target_ids.numel()
    )
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads = {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return logits.detach().clone(), loss.detach(), grads


def _tp_equivalence_worker():
    device = torch.device("cuda", torch.cuda.current_device())
    topology = ParallelTopology(
        world_size=dist.get_world_size(), tp_size=2, device_type="cuda"
    )
    model = _build_model(device)

    torch.manual_seed(11)
    input_ids = torch.randint(
        0, model.config.vocab_size, (BATCH, SEQ_LEN), device=device
    )
    target_ids = torch.randint(
        0, model.config.vocab_size, (BATCH, SEQ_LEN), device=device
    )

    # Both ranks hold identical weights, so each verifies the single-device
    # reference locally before the same model object is sharded in place.
    ref_logits, ref_loss, ref_grads = _reference(model, input_ids, target_ids)

    tp = TPState(topology)
    with tp.shard(model):
        logits = model(input_ids=input_ids)["logits"]
        # Output precision: the tp forward reproduces the single-device
        # logits elementwise (only gemm-split accumulation-order noise;
        # fp32 tiny model keeps it at float-eps level).
        torch.testing.assert_close(logits, ref_logits, rtol=1e-5, atol=1e-5)
        loss = (
            torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), target_ids.flatten(), reduction="sum"
            )
            / target_ids.numel()
        )
        torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-5)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"no grad for {name}"
            ref_grad = ref_grads[name]
            param_base = name.rsplit(".", 1)[0]
            shard_key = next((k for k in _SHARD_DIMS if param_base.endswith(k)), None)
            if shard_key is None:
                # replicated parameter (embed/lm_head/norms): exact grad
                torch.testing.assert_close(param.grad, ref_grad, rtol=1e-5, atol=1e-6)
            else:
                local = param.grad
                expected = ref_grad.chunk(2, dim=_SHARD_DIMS[shard_key])[
                    topology.tp_rank
                ]
                torch.testing.assert_close(local, expected, rtol=1e-4, atol=1e-5)


@skip_lt2_cuda
def test_tp2_matches_single_device():
    from astrai.parallel.setup import spawn_parallel_fn

    spawn_parallel_fn(_tp_equivalence_worker, world_size=2)


def test_tp_state_requires_active_dimension():
    with pytest.raises(ValueError, match="tp dimension"):
        TPState(ParallelTopology(world_size=1))


def _plan_validation_worker():
    """A typo'd pattern fails loudly instead of leaving projections whole."""
    device = torch.device("cuda", torch.cuda.current_device())
    topology = ParallelTopology(
        world_size=dist.get_world_size(), tp_size=2, device_type="cuda"
    )
    model = _build_model(device)
    with pytest.raises(ValueError, match="matched no module"):
        with TPState(topology).shard(model, {"*.attention.q_projz": "colwise"}):
            pass
    with pytest.raises(ValueError, match="split styles"):
        with TPState(topology).shard(model, {"*.mlp.up": "diagonal"}):
            pass


@skip_lt2_cuda
def test_tp_plan_validation():
    from astrai.parallel.setup import spawn_parallel_fn

    spawn_parallel_fn(_plan_validation_worker, world_size=2)


def test_default_plan_covers_standard_layout():
    """The default plan touches every projection and nothing else."""
    from fnmatch import fnmatch

    from astrai.model.components.linear import Linear

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
