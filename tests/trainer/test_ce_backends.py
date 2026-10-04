import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.extension import ATTN_BACKEND, attn_backend
from astrai.extension.kernel.cross_entropy import is_available
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.cp import CPState, CPStrategy
from astrai.parallel.topology import ParallelTopology
from astrai.trainer.strategy import SEQStrategy, SFTStrategy

skip_no_ce = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="CE CUDA kernel not built",
)


def model(device, tied=False):
    config = AutoRegressiveLMConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
    )
    return AutoRegressiveLM(config).to(
        device=device,
        dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
    )


def batch(device, all_masked=False):
    return dict(
        input_ids=torch.randint(257, (2, 19), device=device),
        target_ids=torch.randint(257, (2, 19), device=device),
        position_ids=torch.arange(19, device=device).expand(2, -1),
        loss_mask=torch.zeros(2, 19, dtype=torch.bool, device=device)
        if all_masked
        else torch.rand(2, 19, device=device) > 0.3,
    )


@skip_no_ce
@pytest.mark.parametrize("backend", ["cuda_ce", "cuda_linear_ce"])
@pytest.mark.parametrize("kind", [SEQStrategy, SFTStrategy])
@pytest.mark.parametrize("tied", [False, True])
def test_multistep_accumulation(backend, kind, tied):
    torch.manual_seed(123)
    ref = model("cuda", tied)
    fused = copy.deepcopy(ref)
    a, b = (
        kind(ref, "cuda", label_smoothing=0.1),
        kind(
            fused, "cuda", label_smoothing=0.1, loss_backend=backend, loss_chunk_size=16
        ),
    )
    oa, ob = (
        torch.optim.SGD(ref.parameters(), lr=0.05),
        torch.optim.SGD(fused.parameters(), lr=0.05),
    )
    for _ in range(3):
        oa.zero_grad()
        ob.zero_grad()
        for _ in range(2):
            data = batch("cuda")
            la, lb = a.compute_loss(data), b.compute_loss(data)
            torch.testing.assert_close(la, lb, rtol=2e-4, atol=2e-4)
            (la / 2).backward()
            (lb / 2).backward()
        for x, y in zip(ref.parameters(), fused.parameters()):
            assert x.grad is not None and y.grad is not None
            err = (
                x.grad.float() - y.grad.float()
            ).norm() / x.grad.float().norm().clamp_min(1e-8)
            assert err < 0.04
        oa.step()
        ob.step()
    for x, y in zip(ref.parameters(), fused.parameters()):
        torch.testing.assert_close(x, y, rtol=0.04, atol=0.001)


@pytest.mark.parametrize("backend", ["cuda_ce", "cuda_linear_ce"])
def test_cpu_fallback_and_mask(backend):
    torch.manual_seed(2)
    m = model("cpu")
    for masked in (False, True):
        data = batch("cpu", masked)
        expected = SFTStrategy(m, "cpu").compute_loss(data)
        actual = SFTStrategy(m, "cpu", loss_backend=backend).compute_loss(data)
        torch.testing.assert_close(actual, expected)
        actual.backward()
    assert m(batch("cpu")["input_ids"])["logits"] is not None


@skip_no_ce
def test_missing_kernel_fallback(monkeypatch):
    monkeypatch.setattr("astrai.trainer.strategy.ce_available", lambda: False)
    m, data = model("cuda"), batch("cuda")
    ref = SEQStrategy(m, "cuda").compute_loss(data)
    for backend in ("cuda_ce", "cuda_linear_ce"):
        loss = SEQStrategy(m, "cuda", loss_backend=backend).compute_loss(data)
        torch.testing.assert_close(loss, ref, rtol=0, atol=0)


def test_model_has_no_loss_api():
    m = model("cpu")
    data = batch("cpu")
    with pytest.raises(TypeError, match="loss_targets"):
        m(data["input_ids"], loss_targets=data["target_ids"])
    outputs = m(data["input_ids"], skip_lm_head=True, return_lm_head_weight=True)
    assert outputs["logits"] is None
    assert outputs["lm_head_weight"].shape == m.lm_head.weight.shape


def _ddp_worker(rank, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="file://" + init_file, rank=rank, world_size=2
    )
    try:
        torch.manual_seed(91)
        initial = model("cuda:" + str(rank), tied=True)
        ddp = DDP(
            copy.deepcopy(initial), device_ids=[rank], find_unused_parameters=True
        )
        ref = DDP(
            copy.deepcopy(initial), device_ids=[rank], find_unused_parameters=True
        )
        torch.manual_seed(123 + rank)
        data = batch("cuda:" + str(rank), all_masked=rank == 1)
        for backend in ("cuda_ce", "cuda_linear_ce"):
            ddp.zero_grad()
            ref.zero_grad()
            for net, mode in ((ddp, backend), (ref, "torch")):
                strategy = SFTStrategy(
                    net, "cuda:" + str(rank), loss_backend=mode, loss_chunk_size=16
                )
                result = strategy.forward_tokens(data)
                tokens = strategy.reduce_loss(result, data)
                count = tokens.token_count.clone()
                dist.all_reduce(count)
                # DDP averages gradients; global token mean needs world/count.
                (tokens.loss_sum * (2 / count.clamp_min(1))).backward()
            for x, y in zip(ddp.parameters(), ref.parameters()):
                torch.testing.assert_close(x.grad, y.grad, rtol=0.04, atol=0.001)
        # Exercise the real context-parallel sharding/normalization protocol.
        topology = ParallelTopology(world_size=2, cp_size=2, device_type="cuda")
        torch.manual_seed(123)
        data = {
            k: v[:, :18].contiguous() for k, v in batch("cuda:" + str(rank)).items()
        }
        with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
            single = copy.deepcopy(initial)
            reference = SFTStrategy(
                single, "cuda:" + str(rank), label_smoothing=0.1
            ).compute_loss(data)
            reference.backward()
            for backend in ("cuda_ce", "cuda_linear_ce"):
                sharded = copy.deepcopy(initial)
                strategy = SFTStrategy(
                    sharded,
                    "cuda:" + str(rank),
                    label_smoothing=0.1,
                    loss_backend=backend,
                    loss_chunk_size=8,
                )
                result = CPStrategy(strategy, CPState(topology))(copy.deepcopy(data))
                torch.testing.assert_close(
                    result["metrics"]["task_loss"],
                    reference.item(),
                    rtol=0.01,
                    atol=0.01,
                )
                (result["loss"] / 2).backward()
                for x, y in zip(sharded.parameters(), single.parameters()):
                    grad = x.grad.float()
                    dist.all_reduce(grad, group=topology.cp_group)
                    torch.testing.assert_close(
                        grad, y.grad.float(), rtol=0.05, atol=0.01
                    )
    finally:
        dist.destroy_process_group()


@skip_no_ce
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_ddp_global_count(tmp_path):
    mp.spawn(_ddp_worker, args=(os.fspath(tmp_path / "init"),), nprocs=2, join=True)


def _fsdp_worker(rank, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method="file://" + init_file, rank=rank, world_size=2
    )
    try:
        torch.manual_seed(50)
        sharded = model("cuda:" + str(rank))
        reference = copy.deepcopy(sharded)
        fully_shard(sharded.lm_head, reshard_after_forward=False)
        data = batch("cuda:" + str(rank))
        actual = SEQStrategy(
            sharded, "cuda:" + str(rank), loss_backend="cuda_linear_ce"
        ).compute_loss(data)
        expected = SEQStrategy(reference, "cuda:" + str(rank)).compute_loss(data)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        actual.backward()
        assert sharded.lm_head.weight.grad is not None
    finally:
        dist.destroy_process_group()


@skip_no_ce
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_fsdp_head_fallback(tmp_path):
    mp.spawn(
        _fsdp_worker, args=(os.fspath(tmp_path / "init_fsdp"),), nprocs=2, join=True
    )
