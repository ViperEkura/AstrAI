import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset

import astrai.parallel.executor as executor_module
from astrai.config import TrainConfig
from astrai.inference.scheduler import InferenceScheduler
from astrai.model.transformer import AutoRegressiveLM
from astrai.parallel import get_rank, spawn_parallel_fn
from astrai.parallel.executor import DDPExecutor, FSDPExecutor, NoneExecutor
from astrai.trainer import train_context
from astrai.trainer.train_context import TrainContextBuilder
from tests.helpers import FakeTokenizer, make_rollout_config

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


def _online_train_config(parallel_mode):
    return TrainConfig(
        strategy="online_grpo",
        model_fn=_ConfigModel,
        dataset=_ConfigDataset(),
        optimizer_fn=_optimizer_fn,
        scheduler_fn=_scheduler_fn,
        nprocs=2,
        parallel_mode=parallel_mode,
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


def test_ddp_executor_returns_the_public_underlying_module(tmp_path):
    _init_single_rank_process_group(tmp_path, "gloo")
    try:
        model = _ConfigModel()
        wrapped = DDP(model)

        assert not hasattr(wrapped, "config")
        assert DDPExecutor().model_for_inference(wrapped) is model
    finally:
        dist.destroy_process_group()


def test_train_context_passes_ddp_inference_view_to_rollout(tmp_path, monkeypatch):
    _init_single_rank_process_group(tmp_path, "gloo")
    captured = {}

    class _Scheduler:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train_context, "InferenceScheduler", _Scheduler)
    monkeypatch.setattr(
        train_context.AutoTokenizer,
        "from_pretrained",
        lambda _param_path: FakeTokenizer(),
    )
    try:
        model = _ConfigModel()
        wrapped = DDP(model)
        context = _rollout_context(wrapped, DDPExecutor())
        builder = TrainContextBuilder(_rollout_config())

        builder._configure_rollout(context, {"group_size": 2})

        assert captured["model"] is model
        assert captured["max_seq_len"] == 32
        assert captured["max_batch_size"] == 2
        assert context.strategy.runner.generator.scheduler.__class__ is _Scheduler
    finally:
        dist.destroy_process_group()


def test_online_rollout_rejects_torch_compile_before_scheduler(monkeypatch):
    monkeypatch.setattr(
        train_context.AutoTokenizer,
        "from_pretrained",
        lambda _param_path: pytest.fail("tokenizer should not be loaded"),
    )
    context = _rollout_context(_ConfigModel(), NoneExecutor())
    builder = TrainContextBuilder(_rollout_config(compile_mode="default"))

    with pytest.raises(ValueError, match="does not support torch.compile"):
        builder._configure_rollout(context, {"group_size": 1})


def test_train_config_accepts_multi_process_ddp_online_rollout():
    config = _online_train_config("ddp")

    assert config.nprocs == 2
    assert config.parallel_mode == "ddp"


@pytest.mark.parametrize("parallel_mode", ["none", "fsdp"])
def test_train_config_rejects_multi_process_online_rollout_without_ddp(
    parallel_mode,
):
    with pytest.raises(ValueError, match="requires parallel_mode='ddp'"):
        _online_train_config(parallel_mode)


def test_distributed_fsdp_rollout_fails_before_model_access(monkeypatch):
    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    executor = FSDPExecutor()
    capabilities = executor.rollout_capabilities()

    assert not capabilities.supports_in_process
    assert "replicated model view" in capabilities.reason
    with pytest.raises(RuntimeError, match="parameters are sharded"):
        executor.model_for_inference(_ConfigModel())


def test_train_context_rejects_distributed_fsdp_during_early_validation(monkeypatch):
    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    builder = TrainContextBuilder(_rollout_config())

    with pytest.raises(ValueError, match="replicated model view"):
        builder._validate_rollout_configuration(FSDPExecutor())


def test_build_rejects_distributed_fsdp_before_model_construction(monkeypatch):
    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    builder = TrainContextBuilder(_rollout_config())
    monkeypatch.setattr(builder, "_load_preloaded_state", lambda: object())
    monkeypatch.setattr(builder, "_create_executor", FSDPExecutor)
    monkeypatch.setattr(
        builder,
        "_create_context",
        lambda *_args: pytest.fail("model context should not be constructed"),
    )

    with pytest.raises(ValueError, match="replicated model view"):
        builder.build()


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

        baseline = InferenceScheduler(
            model=model,
            tokenizer=tokenizer,
            max_batch_size=2,
            max_seq_len=64,
            enable_cuda_graph=False,
            backend="torch_native",
        )
        ddp_view = InferenceScheduler(
            model=inference_model,
            tokenizer=tokenizer,
            max_batch_size=2,
            max_seq_len=64,
            enable_cuda_graph=False,
            backend="torch_native",
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
    scheduler = InferenceScheduler(
        model=inference_model,
        tokenizer=FakeTokenizer(),
        max_batch_size=1,
        max_seq_len=64,
        enable_cuda_graph=False,
        backend="torch_native",
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
