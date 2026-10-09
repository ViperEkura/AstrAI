import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

from astrai.config import TrainConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel import get_rank, spawn_parallel_fn
from astrai.parallel.executor import DDPExecutor
from tests.support.inference import make_cpu_scheduler
from tests.support.models import make_rollout_config
from tests.support.tokenizers import FakeTokenizer

_DDP_TEST_WORLD_SIZE = int(os.environ.get("ASTRAI_DDP_TEST_WORLD_SIZE", "2"))


class _ConfigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.config = SimpleNamespace(max_position_embeddings=32)

    def forward(self, value):
        return self.projection(value)


class _OnlineStrategy:
    def __init__(self):
        self.runner = None

    def supports_online(self):
        return True

    def set_rollout_runner(self, runner):
        self.runner = runner


class _ConfigDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"index": index}


def _optimizer_fn(model):
    return torch.optim.SGD(model.parameters(), lr=1e-3)


def _scheduler_fn(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)


def _online_train_config(dp_mode):
    return TrainConfig(
        strategy="online_grpo",
        model_fn=_ConfigModel,
        dataset=_ConfigDataset(),
        optimizer_fn=_optimizer_fn,
        scheduler_fn=_scheduler_fn,
        dp_size=2,
        dp_mode=dp_mode,
        reward_model_fn=object,
    )


def _init_single_rank_process_group(tmp_path, backend):
    if not dist.is_available() or dist.is_initialized():
        pytest.skip("test requires ownership of one temporary process group")
    rendezvous = tmp_path / f"{backend}-rendezvous"
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )


def _rollout_config(*, compile_mode=None):
    return SimpleNamespace(
        strategy="online_grpo",
        compile_mode=compile_mode,
        batch_per_device=1,
        rollout_max_tokens=4,
        rollout_temperature=0.0,
        rollout_top_k=0,
        rollout_top_p=1.0,
        rollout_interval=1,
        rollout_max_policy_lag=None,
        rollout_pool_seq_len=None,
        rollout_device=None,
        rollout_val_device=None,
        rollout_val_overrides=lambda: {},
        cp_size=1,
        tp_size=1,
        device_type="cpu",
        reward_model_fn=object,
    )


def _rollout_context(model, executor):
    return SimpleNamespace(
        model=model,
        executor=executor,
        strategy=_OnlineStrategy(),
        checkpoint=None,
        optimizer_step=0,
    )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_l20_ddp_inference_view_matches_greedy_generation(tmp_path):
    _init_single_rank_process_group(tmp_path, "nccl")
    try:
        torch.manual_seed(47)
        model = AutoRegressiveLM(make_rollout_config()).to(
            device="cuda", dtype=torch.bfloat16
        )
        model.eval()
        wrapped = DDP(model, device_ids=[0], output_device=0)
        inference_model = DDPExecutor().model_for_inference(wrapped)
        tokenizer = FakeTokenizer()

        baseline = make_cpu_scheduler(
            model, tokenizer, max_batch_size=2, max_seq_len=64
        )
        ddp_view = make_cpu_scheduler(
            inference_model, tokenizer, max_batch_size=2, max_seq_len=64
        )
        prompts = [[5, 6, 7], [8, 9, 10, 11]]

        expected = baseline.run_batch(prompts, max_tokens=4, temperature=0)
        actual = ddp_view.run_batch(prompts, max_tokens=4, temperature=0)

        assert actual == expected
    finally:
        dist.destroy_process_group()


def _multi_rank_rollout_then_train_worker():
    rank = get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    config = make_rollout_config()
    torch.manual_seed(47 + rank)

    executor = DDPExecutor()
    wrapped, optimizer, _ = executor.prepare(
        lambda: AutoRegressiveLM(config),
        _optimizer_fn,
        before_wrap=lambda model: model.to(device=device, dtype=torch.float32),
    )
    assert isinstance(wrapped, DDP)
    inference_model = executor.model_for_inference(wrapped)
    assert inference_model is wrapped.module
    initial_parameters = [
        parameter.detach().clone() for parameter in inference_model.parameters()
    ]

    # Deliberately issue a different number of collective-free inference
    # forwards on each rank. This is the deadlock pattern that is unsafe when
    # the DDP wrapper itself is passed to the scheduler.
    inference_model.eval()
    scheduler = make_cpu_scheduler(
        inference_model, FakeTokenizer(), max_batch_size=1, max_seq_len=64
    )
    result = scheduler.run_batch(
        [[5 + rank, 6 + rank, 7 + rank]],
        max_tokens=2 + rank,
        temperature=0,
    )
    assert len(result) == 1

    # Training stays wrapped and performs one synchronized update per rank.
    wrapped.train()
    optimizer.zero_grad()
    input_ids = torch.tensor([[5 + rank, 6 + rank, 7 + rank, 8 + rank]], device=device)
    target_ids = torch.tensor([[6 + rank, 7 + rank, 8 + rank, 9 + rank]], device=device)
    logits = wrapped(input_ids)["logits"]
    loss = F.cross_entropy(logits.flatten(0, 1), target_ids.flatten())
    executor.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, inference_model.parameters())
    )

    # DDP replicas must remain bitwise-consistent after the update.
    for parameter in inference_model.parameters():
        rank_zero_parameter = parameter.detach().clone()
        dist.broadcast(rank_zero_parameter, src=0)
        torch.testing.assert_close(parameter, rank_zero_parameter, rtol=0, atol=0)

    # The shared inference view sees the new weights, and deterministic
    # generation agrees across replicas after invalidating stale KV state.
    scheduler.update_weights(1)
    inference_model.eval()
    post_update = scheduler.run_batch([[11, 12, 13]], max_tokens=3, temperature=0)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, post_update)
    assert all(tokens == gathered[0] for tokens in gathered)


@pytest.mark.integration
@pytest.mark.skipif(
    torch.cuda.device_count() < _DDP_TEST_WORLD_SIZE,
    reason=f"{_DDP_TEST_WORLD_SIZE} CUDA devices are required",
)
def test_l20_multi_rank_rollout_can_diverge_before_ddp_training():
    spawn_parallel_fn(
        _multi_rank_rollout_then_train_worker,
        world_size=_DDP_TEST_WORLD_SIZE,
        backend="nccl",
    )
