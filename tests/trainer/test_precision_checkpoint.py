"""Export formats must never change resumable training tensors."""

from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
from torch import nn

from astrai.config import ComponentSpec, PrecisionConfig, TrainConfig
from astrai.optimization.optimizer.precision import PrecisionOptimizer
from astrai.parallel.executor import NoneExecutor
from astrai.serialization import Checkpoint
from astrai.trainer.context import TrainContextBuilder
from astrai.trainer.context.resume import PreloadedState, ResumeLoader
from astrai.trainer.trainer import Trainer
from tests.support.precision import assert_precision_state
from tests.support.rollout import LengthRewardModel
from tests.support.tokenizers import CHAT_TEMPLATE
from tests.support.trainer import make_online_value_model
from tests.trainer.test_unified_resume import _StopAfterOne, make_resume_config


@pytest.mark.parametrize("compute", ["float32", "bfloat16"])
def test_export_roundtrip_preserves_training_state(tmp_path, compute):
    policy = PrecisionConfig(compute_dtype=compute, export_dtype="float16")
    baseline_dir, source_dir, resumed_dir = [
        tmp_path / n for n in ("baseline", "source", "resumed")
    ]
    Trainer(make_resume_config(baseline_dir, precision=policy)).train()
    Trainer(
        make_resume_config(source_dir, precision=policy), callbacks=[_StopAfterOne()]
    ).train()
    source = source_dir / "epoch_0_step_1"
    saved = Checkpoint.load(str(source))
    assert all(
        t.dtype == torch.float16
        for t in saved.state_dict.values()
        if t.is_floating_point()
    )
    assert all(
        t.dtype == getattr(torch, compute)
        for t in saved.extra["training_model"].values()
        if t.is_floating_point()
    )
    warm = PreloadedState()
    loader = ResumeLoader(
        make_resume_config(tmp_path / "warm", precision=policy),
        str(source),
        False,
        None,
        lambda _: None,
    )
    loader._load_from_param_path(warm)
    for name, value in saved.state_dict.items():
        torch.testing.assert_close(warm.state_dict[name], value, atol=0, rtol=0)
    # Changing only export format must preserve the full resume trajectory.
    Trainer(
        make_resume_config(
            resumed_dir, precision=replace(policy, export_dtype="float32")
        )
    ).train(param_path=str(source), resume=True)
    baseline = Checkpoint.load(str(baseline_dir / "epoch_0_step_4"))
    resumed = Checkpoint.load(str(resumed_dir / "epoch_0_step_4"))
    for name, value in baseline.extra["training_model"].items():
        torch.testing.assert_close(
            value, resumed.extra["training_model"][name], atol=0, rtol=0
        )
    if compute == "bfloat16":
        for a, b in zip(
            baseline.extra["optimizer"]["master_params"],
            resumed.extra["optimizer"]["master_params"],
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize(
    "name", ["compute_dtype", "gradient_dtype", "master_dtype", "optimizer_state_dtype"]
)
def test_resume_rejects_training_precision_change(tmp_path, name):
    policy = PrecisionConfig(compute_dtype="float32")
    source_dir = tmp_path / "source"
    Trainer(
        make_resume_config(source_dir, precision=policy), callbacks=[_StopAfterOne()]
    ).train()
    changed = replace(policy, **{name: "bfloat16"})
    with pytest.raises(ValueError, match="precision." + name):
        TrainContextBuilder(
            make_resume_config(tmp_path / "resumed", precision=changed)
        ).with_param_path(str(source_dir / "epoch_0_step_1"), resume=True).build()


def test_export_uses_master_without_mutating_training_state():
    model = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    optimizer = PrecisionOptimizer(torch.optim.SGD(model.parameters(), lr=1e-5))
    for _ in range(20):
        model.weight.main_grad = torch.ones_like(model.weight, dtype=torch.float32)
        optimizer.step()
        optimizer.zero_grad()
    master = optimizer.master_state_dict(model.named_parameters())["weight"]
    assert model.weight.item() == 1
    assert master.item() < 1
    executor = NoneExecutor()
    with executor.checkpoint_context(
        model, optimizer=optimizer, export_dtype="float32"
    ) as (fp32, extra):
        torch.testing.assert_close(fp32["weight"], master, atol=0, rtol=0)
        torch.testing.assert_close(
            extra["training_model"]["weight"], model.weight, atol=0, rtol=0
        )
    with executor.checkpoint_context(
        model, optimizer=optimizer, export_dtype="bfloat16"
    ) as (bf16, _):
        assert bf16["weight"].item() == 1
    with executor.checkpoint_context(model, optimizer=optimizer, export_dtype=None) as (
        native,
        extra,
    ):
        assert native["weight"].dtype == torch.bfloat16
        assert native["weight"].item() == 1
        assert extra == {}
    assert master.dtype == torch.float32 and model.weight.dtype == torch.bfloat16
    assert executor.model_state_dict(model)["weight"].item() == 1


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("use_masters", [False, True])
def test_export_handles_tied_names_unmanaged_weights_and_buffers(
    compiled, use_masters, tmp_path
):
    model = nn.Sequential(
        nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    ).bfloat16()
    model[1].weight = model[0].weight
    model.register_parameter(
        "unmanaged", nn.Parameter(torch.full((2,), 0.375, dtype=torch.bfloat16))
    )
    model.register_buffer("count", torch.tensor(7, dtype=torch.int64))
    model.register_buffer("average", torch.tensor(0.625, dtype=torch.bfloat16))
    optimizer = torch.optim.SGD([model[0].weight], lr=1e-5)
    if use_masters:
        optimizer = PrecisionOptimizer(optimizer)
        master = optimizer.master_state_dict(model.named_parameters())["0.weight"]
        with torch.no_grad():
            master.fill_(0.9998)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    if compiled:
        model = torch.compile(model, backend="eager")
    executor = NoneExecutor()
    with executor.checkpoint_context(
        model, optimizer=optimizer, export_dtype="float32"
    ) as (exported, extra):
        assert set(exported) == set(expected)
        assert exported["count"].dtype == torch.int64
        assert exported["average"].dtype == torch.float32
        assert exported["count"].item() == 7
        assert exported["average"].item() == 0.625
        torch.testing.assert_close(
            exported["unmanaged"], expected["unmanaged"].float(), atol=0, rtol=0
        )
        for name in ("0.weight", "1.weight"):
            torch.testing.assert_close(
                exported[name],
                master if use_masters else expected[name].float(),
                atol=0,
                rtol=0,
            )
        # Both tied names must be independently serializable.
        saved_path = tmp_path / "export"
        Checkpoint(state_dict=exported).save(str(saved_path))
        saved = Checkpoint.load(str(saved_path))
        torch.testing.assert_close(
            saved.state_dict["0.weight"], saved.state_dict["1.weight"], atol=0, rtol=0
        )
        exported["0.weight"].zero_()
        assert torch.count_nonzero(exported["1.weight"]) > 0
        extra["training_model"]["unmanaged"].zero_()
    actual = executor.model_state_dict(model)
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, atol=0, rtol=0)


def _sharded_config(path, policy, shard_state=True):
    config = make_resume_config(
        path,
        precision=policy,
        dp_mode="ddp",
        dp_size=2,
        batch_per_device=1,
        backend="gloo",
    )
    return replace(
        config,
        optimizer=replace(
            config.optimizer,
            params={**config.optimizer.params, "shard_state": shard_state},
        ),
    )


@pytest.mark.parametrize(
    "compute,source_sharded,target_sharded",
    [
        ("bfloat16", True, True),
        ("float32", True, False),
        ("float32", False, True),
    ],
)
def test_sharded_trainer_export_resume_matches_continuous_training(
    tmp_path, compute, source_sharded, target_sharded
):
    policy = PrecisionConfig(compute_dtype=compute, export_dtype="float16")
    baseline_dir, source_dir, resumed_dir = [
        tmp_path / name for name in ("baseline", "source", "resumed")
    ]
    Trainer(_sharded_config(baseline_dir, policy, source_sharded)).train()
    Trainer(
        _sharded_config(source_dir, policy, source_sharded), callbacks=[_StopAfterOne()]
    ).train()
    source = source_dir / "epoch_0_step_1"
    saved = Checkpoint.load(str(source))
    assert all(
        value.dtype == torch.float16
        for value in saved.state_dict.values()
        if value.is_floating_point()
    )
    assert all(
        value.dtype == getattr(torch, compute)
        for value in saved.extra["training_model"].values()
        if value.is_floating_point()
    )
    warm = PreloadedState()
    loader = ResumeLoader(
        _sharded_config(tmp_path / "warm", policy),
        str(source),
        False,
        None,
        lambda _: None,
    )
    loader._load_from_param_path(warm)
    torch.testing.assert_close(warm.state_dict, saved.state_dict, rtol=0, atol=0)
    Trainer(
        _sharded_config(
            resumed_dir, replace(policy, export_dtype="float32"), target_sharded
        )
    ).train(param_path=str(source), resume=True)
    baseline = Checkpoint.load(str(baseline_dir / "epoch_0_step_4"))
    resumed = Checkpoint.load(str(resumed_dir / "epoch_0_step_4"))
    torch.testing.assert_close(
        baseline.extra["training_model"],
        resumed.extra["training_model"],
        rtol=0,
        atol=0,
    )
    assert_precision_state(baseline.extra["optimizer"], resumed.extra["optimizer"])


def test_sharded_ppo_critic_checkpoint_matches_continuous_training(base_test_env):
    root = Path(base_test_env["test_dir"])
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(CHAT_TEMPLATE)
    tokenizer.save_pretrained(root)
    model_config = base_test_env["transformer_config"]
    config = TrainConfig(
        strategy="online_ppo",
        model=ComponentSpec(
            type="autoregressive_lm", params={"config": model_config.to_dict()}
        ),
        precision=PrecisionConfig(compute_dtype="float32", export_dtype="bfloat16"),
        dataset=ComponentSpec(
            type="tests.support.rollout:InstructionDataset",
            params={"collate": "tests.support.rollout:instruction_collate_fn"},
        ),
        optimizer=ComponentSpec(
            type="torch.optim:AdamW", params={"lr": 1e-4, "shard_state": True}
        ),
        scheduler=ComponentSpec(
            type="torch.optim.lr_scheduler:StepLR",
            params={"step_size": 1, "gamma": 1.0},
        ),
        n_epoch=1,
        batch_per_device=1,
        grad_accum_steps=1,
        ckpt_interval=1,
        random_seed=42,
        device_type="cpu",
        backend="gloo",
        dp_mode="ddp",
        dp_size=2,
        strategy_kwargs={"group_size": 2, "kl_coef": 0.01},
        rollout_interval=1,
        rollout_max_policy_lag=0,
        rollout_temperature=1.0,
        rollout_top_k=1,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        critic_model_fn=partial(make_online_value_model, model_config),
    )
    Trainer(replace(config, ckpt_dir=str(root / "continuous"))).train(
        param_path=str(root)
    )
    Trainer(
        replace(config, ckpt_dir=str(root / "source")), callbacks=[_StopAfterOne()]
    ).train(param_path=str(root))
    source = root / "source/epoch_0_step_1"
    saved = Checkpoint.load(str(source))
    assert saved.extra["value_optimizer"]["master_params"]
    assert all(
        t.dtype == torch.float32
        for t in saved.extra["value_optimizer"]["master_params"]
    )
    Trainer(replace(config, ckpt_dir=str(root / "resumed"))).train(
        param_path=str(source), resume=True
    )
    expected = Checkpoint.load(str(root / "continuous/epoch_0_step_2"))
    actual = Checkpoint.load(str(root / "resumed/epoch_0_step_2"))
    for key in ("training_model", "optimizer", "value_model", "value_optimizer"):
        assert_precision_state(actual.extra[key], expected.extra[key])
