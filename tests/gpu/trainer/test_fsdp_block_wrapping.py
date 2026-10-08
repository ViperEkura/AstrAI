"""Two-GPU FSDP forward/update and canonical weight checkpoint oracle."""

from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.executor import FSDPExecutor
from astrai.serialization import Checkpoint
from tests.support.models import make_rollout_config


def _worker(rank, rendezvous, output, tied):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(3407)
        config = make_rollout_config(
            vocab_size=32, num_hidden_layers=2, tie_word_embeddings=tied
        )
        full = AutoRegressiveLM(config).to(device)
        expected = deepcopy(full)
        executor = FSDPExecutor(mesh=init_device_mesh("cuda", (2,)))
        wrapped, optimizer, _ = executor.prepare(
            lambda: full, lambda model: torch.optim.AdamW(model.parameters(), lr=0.001)
        )
        assert sum(isinstance(module, FSDPModule) for module in wrapped.modules()) == 3
        assert (wrapped.lm_head.weight is wrapped.model.embed_tokens.weight) == tied
        wanted_optimizer = torch.optim.AdamW(expected.parameters(), lr=0.001)
        ids = torch.arange(24, device=device).reshape(4, 6) % 32
        labels = (ids + 1) % 32
        for _ in range(2):
            wanted = F.cross_entropy(
                expected(ids)["logits"].reshape(-1, 32), labels.reshape(-1)
            )
            local_ids, local_labels = ids.chunk(2)[rank], labels.chunk(2)[rank]
            got = F.cross_entropy(
                wrapped(local_ids)["logits"].reshape(-1, 32), local_labels.reshape(-1)
            )
            wanted.backward()
            got.backward()
            wanted_optimizer.step()
            optimizer.step()
            wanted_optimizer.zero_grad()
            optimizer.zero_grad()
        with executor.checkpoint_context(wrapped) as state:
            if rank == 0:
                for key, value in expected.state_dict().items():
                    torch.testing.assert_close(state[key], value, rtol=3e-4, atol=3e-5)
                Checkpoint(state_dict=state, config=config.to_dict()).save(output)
        dist.barrier()
        checkpoint = Checkpoint.load_any(output)
        restored = AutoRegressiveLM(config).to(device)
        restored.load_state_dict(checkpoint.state_dict)
        torch.testing.assert_close(
            restored(ids)["logits"], expected(ids)["logits"], rtol=3e-4, atol=3e-5
        )
        assert set(checkpoint.state_dict) == set(expected.state_dict())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires two assigned CUDA test GPUs"
)
@pytest.mark.parametrize("tied", [False, True])
def test_fsdp_blocks_two_gpu_update_and_checkpoint(tmp_path, tied):
    mp.spawn(
        _worker,
        args=(
            "file://" + str(tmp_path / "rendezvous"),
            str(tmp_path / "weights"),
            tied,
        ),
        nprocs=2,
        join=True,
    )
