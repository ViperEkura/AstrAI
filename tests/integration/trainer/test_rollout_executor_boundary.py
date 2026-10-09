import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, TensorDataset

import astrai.parallel.executor as executor_module
from astrai.config import TrainConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel import get_rank
from astrai.parallel.executor import DDPExecutor, FSDPExecutor, NoneExecutor
from astrai.trainer import train_context
from astrai.trainer.rollout.setup import configure_rollout, resolve_async_rollout
from astrai.trainer.rollout.types import SamplingParams
from astrai.trainer.train_context import TrainContextBuilder
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
        rollout_mode="sync",
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

    monkeypatch.setattr(train_context, "Scheduler", _Scheduler)
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
        assert (
            context.strategy.runner.generator.backend.scheduler.__class__ is _Scheduler
        )
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
    assert config.dp_mode == "ddp"


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


def _config(**overrides):
    values = dict(
        model_fn=Mock(),
        dataset=TensorDataset(torch.ones(5)),
        optimizer_fn=Mock(),
        scheduler_fn=Mock(),
        strategy="online_grpo",
        reward_model_fn=Mock(),
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2"],
        rollout_interval=1,
    )
    values.update(overrides)
    return TrainConfig(**values)


@pytest.mark.parametrize("name", ["learner_device", "group_size", "rollout_devies"])
def test_unknown_training_fields_fail(name):
    with pytest.raises(ValueError, match=name):
        _config(**{name: 2})


@pytest.mark.parametrize(
    "name,value",
    [
        ("rl_update_epochs", 3),
        ("rl_minibatch_prompts", 2),
        ("gradient_chunked_logprobs", True),
        ("moe_aux_loss_coef", 0.5),
    ],
)
def test_strategy_kwargs_cannot_override_owned_fields(name, value):
    with pytest.raises(ValueError, match="not strategy_kwargs"):
        _config(strategy_kwargs={name: value})


@pytest.mark.parametrize("value", [0, 1, True, 1.5, "4"])
def test_invalid_grpo_group_size_fails_before_startup(value):
    with pytest.raises(ValueError, match="group_size"):
        _config(strategy_kwargs={"group_size": value})


@pytest.mark.parametrize(
    "name",
    [
        "rollout_temperature",
        "rollout_val_temperature",
        "rollout_worker_timeout_s",
        "rollout_startup_timeout_s",
        "rollout_weight_timeout_s",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_sampling_and_timeout_fields_fail(name, value):
    with pytest.raises(ValueError):
        _config(**{name: value})


def test_runtime_world_size_rejected_before_loading_models(monkeypatch):
    builder = TrainContextBuilder(_config())
    load = Mock()
    builder._load_preloaded_state = load
    monkeypatch.setattr("astrai.trainer.train_context.get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="actual world_size=1"):
        builder.build()
    load.assert_not_called()


def test_effective_grpo_defaults_and_all_worker_parameters(monkeypatch):
    cfg = _config(
        batch_per_device=5,
        rollout_startup_timeout_s=73,
        rollout_worker_timeout_s=91,
        rollout_weight_timeout_s=123,
        random_seed=42,
        rollout_temperature=0.8,
        rollout_top_k=7,
        rollout_top_p=0.75,
        rollout_max_tokens=16,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(ffn_type="mlp", max_position_embeddings=256),
        parameters=lambda: iter(
            [SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.bfloat16)]
        ),
    )
    context = SimpleNamespace(
        model=model,
        executor=SimpleNamespace(model_for_inference=lambda m: m),
        strategy=SimpleNamespace(
            group_size=4, supports_online=lambda: True, set_rollout_runner=Mock()
        ),
        consumed_samples=10,
        checkpoint=None,
        optimizer_step=2,
    )
    coordinator = Mock()
    monkeypatch.setattr(
        "astrai.trainer.rollout.setup.AsyncRoundCoordinator", coordinator
    )
    monkeypatch.setattr("torch.cuda.device_count", lambda: 3)
    configure_rollout(
        context, cfg, "tokenizer", {}, Mock(), lambda _: None, Mock(), Mock()
    )
    values = coordinator.call_args.kwargs
    assert values["params"] == SamplingParams(
        group_size=4, max_tokens=16, temperature=0.8, top_k=7, top_p=0.75
    )
    assert values["max_batch_size"] == 12
    assert values["max_prompts_per_worker"] == 3
    assert values["startup_timeout_s"] == 73
    assert values["worker_timeout_s"] == 91
    assert values["weight_timeout_s"] == 123
    assert values["random_seed"] == 42
    assert values["sample_cursor"] == 10
    assert cfg.rollout_max_policy_lag == values["max_policy_lag"] == 1
    assert values["model_dtype"] == torch.bfloat16
    assert values["devices"] == ["cuda:1", "cuda:2"]


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
