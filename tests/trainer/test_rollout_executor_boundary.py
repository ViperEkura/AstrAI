import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import astrai.parallel.executor.base as executor_module
from astrai.config import ComponentSpec, TrainConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel import get_rank, spawn_parallel_fn
from astrai.parallel.executor import DDPExecutor, FSDPExecutor, NoneExecutor
from astrai.trainer.backend import run_batch
from astrai.trainer.context import TrainContextBuilder
from astrai.trainer.context import builder as train_context_builder
from astrai.trainer.rollout.setup import resolve_async_rollout
from astrai.trainer.rollout.types import SamplingParams
from tests.support.inference import make_cpu_model, make_cpu_scheduler
from tests.support.models import make_rollout_config
from tests.support.tokenizers import FakeTokenizer

_DDP_TEST_WORLD_SIZE = int(os.environ.get("ASTRAI_DDP_TEST_WORLD_SIZE", "2"))


def _optimizer_fn(model):
    return torch.optim.SGD(model.parameters(), lr=1e-3)


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


def _online_train_config(dp_mode):
    return TrainConfig(
        strategy="online_grpo",
        model=ComponentSpec(
            type="torch.nn:Linear", params={"in_features": 2, "out_features": 2}
        ),
        dataset=ComponentSpec(
            type="tests.support.rollout:RandomTokenDataset",
            params={"length": 2},
        ),
        optimizer=ComponentSpec(type="torch.optim:SGD", params={"lr": 1e-3}),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR",
            params={"step_size": 1, "gamma": 1.0},
        ),
        dp_size=2,
        dp_mode=dp_mode,
        reward_model_fn=object,
    )


@pytest.mark.parametrize("enable_overlap", [False, True])
def test_rollout_enable_overlap_reaches_engine_core(monkeypatch, enable_overlap):
    captured = {}

    class _EngineCore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train_context_builder, "EngineCore", _EngineCore)
    monkeypatch.setattr(
        train_context_builder.AutoTokenizer,
        "from_pretrained",
        lambda _param_path: FakeTokenizer(),
    )
    context = SimpleNamespace(
        model=make_cpu_model(),
        executor=NoneExecutor(),
        checkpoint=None,
        optimizer_step=0,
        strategy=SimpleNamespace(supports_online=lambda: True),
    )
    builder = TrainContextBuilder(
        _config(
            rollout_mode="sync",
            rollout_devices=[],
            rollout_enable_overlap=enable_overlap,
        )
    )

    builder._configure_rollout(context, {"group_size": 2})

    assert captured["enable_overlap"] is enable_overlap


def test_online_rollout_rejects_torch_compile_before_scheduler(monkeypatch):
    monkeypatch.setattr(
        train_context_builder.AutoTokenizer,
        "from_pretrained",
        lambda _param_path: pytest.fail("tokenizer should not be loaded"),
    )
    context = SimpleNamespace(
        model=nn.Linear(4, 4),
        executor=NoneExecutor(),
        checkpoint=None,
        optimizer_step=0,
        strategy=SimpleNamespace(supports_online=lambda: True),
    )
    builder = TrainContextBuilder(
        _config(rollout_mode="sync", rollout_devices=[], compile_mode="default")
    )

    with pytest.raises(ValueError, match="does not support torch.compile"):
        builder._configure_rollout(context, {"group_size": 1})


@pytest.mark.parametrize("dp_mode", ["none", "fsdp"])
def test_train_config_rejects_multi_process_online_rollout_without_ddp(
    dp_mode,
):
    with pytest.raises(ValueError, match="requires dp_mode='ddp'"):
        _online_train_config(dp_mode)


def test_distributed_fsdp_rollout_fails_before_model_access(monkeypatch):
    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    executor = FSDPExecutor()
    capabilities = executor.rollout_capabilities()

    assert not capabilities.supports_in_process
    assert "replicated model view" in capabilities.reason
    with pytest.raises(RuntimeError, match="parameters are sharded"):
        executor.model_for_inference(nn.Linear(4, 4))


def test_build_rejects_distributed_fsdp_before_model_construction(monkeypatch):
    monkeypatch.setattr(executor_module, "get_world_size", lambda: 2)
    builder = TrainContextBuilder(_config(rollout_mode="sync", rollout_devices=[]))
    monkeypatch.setattr(builder, "_load_preloaded_state", lambda: object())
    monkeypatch.setattr(builder, "_create_executor", FSDPExecutor)
    monkeypatch.setattr(
        builder,
        "_create_context",
        lambda *_args: pytest.fail("model context should not be constructed"),
    )

    with pytest.raises(ValueError, match="replicated model view"):
        builder.build()


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
    result = run_batch(
        scheduler,
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
    post_update = run_batch(scheduler, [[11, 12, 13]], max_tokens=3, temperature=0)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, post_update)
    assert all(tokens == gathered[0] for tokens in gathered)


def _config(**overrides):
    values = dict(
        model=ComponentSpec(
            type="torch.nn:Linear", params={"in_features": 2, "out_features": 2}
        ),
        dataset=ComponentSpec(
            type="tests.support.rollout:RandomTokenDataset", params={"length": 5}
        ),
        optimizer=ComponentSpec(type="torch.optim:SGD", params={"lr": 0.0}),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR", params={"step_size": 1}
        ),
        strategy="online_grpo",
        reward_model_fn=Mock(),
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2"],
        rollout_interval=1,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_invalid_grpo_group_size_fails_before_startup():
    with pytest.raises(ValueError, match="group_size"):
        _config(strategy_kwargs={"group_size": 1})


@pytest.mark.parametrize(
    "name",
    [
        "rollout_temperature",
        "rollout_worker_timeout_s",
        "rollout_weight_timeout_s",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_sampling_and_timeout_fields_fail(name, value):
    with pytest.raises(ValueError):
        _config(**{name: value})


def test_runtime_world_size_rejected_before_loading_models(monkeypatch):
    builder = TrainContextBuilder(_config())
    load = Mock()
    builder._load_preloaded_state = load
    monkeypatch.setattr("astrai.trainer.context.builder.get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="world_size must match configured nprocs"):
        builder.build()
    load.assert_not_called()


@pytest.mark.parametrize("devices", [["cuda:0"], ["cuda:3"]])
def test_device_conflicts_fail_before_startup(monkeypatch, devices):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 3)
    with pytest.raises(ValueError, match="exclude|exceed"):
        resolve_async_rollout(
            _config(rollout_devices=devices),
            torch.device("cuda:0"),
            SamplingParams(group_size=4),
            256,
            0,
        )


def test_ddp_rollout_devices_are_partitioned_by_learner_rank(monkeypatch):
    monkeypatch.setattr("torch.cuda.device_count", lambda: 8)
    config = _config(
        dp_size=2,
        dp_mode="ddp",
        rollout_devices=["cuda:2", "cuda:3", "cuda:4", "cuda:5"],
        batch_per_device=5,
    )
    resolved = [
        resolve_async_rollout(
            config,
            torch.device(f"cuda:{rank}"),
            SamplingParams(group_size=4),
            256,
            12,
            rank=rank,
            world_size=2,
        )
        for rank in range(2)
    ]
    assert resolved[0].devices == ("cuda:2", "cuda:4")
    assert resolved[1].devices == ("cuda:3", "cuda:5")
    assert all(item.sample_cursor == 6 for item in resolved)
    assert all(item.max_prompts_per_worker == 3 for item in resolved)
    assert [item.learner_rank for item in resolved] == [0, 1]


def test_ddp_rollout_rejects_shared_or_missing_worker_devices():
    with pytest.raises(ValueError, match="exclude all learner devices"):
        _config(dp_size=2, dp_mode="ddp", rollout_devices=["cuda:1", "cuda:2"])
    with pytest.raises(ValueError, match="at least one rollout device per learner"):
        _config(dp_size=2, dp_mode="ddp", rollout_devices=["cuda:2"])


@pytest.mark.parametrize(
    "devices, message",
    [
        ([], "at least one rollout device"),
        (["cpu"], "indexed CUDA"),
        (["cuda"], "indexed CUDA"),
        (["cuda:-1"], "indexed CUDA"),
        (["cuda:1", "cuda:01"], "distinct"),
        (["cuda:0"], "exclude all learner"),
    ],
)
def test_async_layout_rules_match_at_config_and_runtime(devices, message):
    with pytest.raises(ValueError, match=message):
        _config(rollout_devices=devices)
    # Setup may be invoked directly, so it enforces the same pure rules.
    with pytest.raises(ValueError, match=message):
        resolve_async_rollout(
            SimpleNamespace(rollout_devices=devices),
            torch.device("cuda:0"),
            SamplingParams(group_size=4),
            256,
            0,
        )


def test_async_runtime_rechecks_available_devices(monkeypatch):
    config = _config(rollout_devices=["cuda:1", "cuda:2"])
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    with pytest.raises(ValueError, match="exceed available CUDA devices"):
        resolve_async_rollout(
            config, torch.device("cuda:0"), SamplingParams(group_size=4), 256, 0
        )


@pytest.mark.distributed
@pytest.mark.cuda
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

        expected = run_batch(baseline, prompts, max_tokens=4, temperature=0)
        actual = run_batch(ddp_view, prompts, max_tokens=4, temperature=0)

        assert actual == expected
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.cuda
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
