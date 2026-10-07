"""End-to-end integration tests for online GRPO/DPO rollout."""

import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, Subset

from astrai.config import TrainConfig
from astrai.serialization import Checkpoint
from astrai.trainer import train_context
from astrai.trainer.rollout import BaseRewardModel
from astrai.trainer.rollout.async_round import (
    AsyncRoundCoordinator,
    WeightSnapshotError,
)
from astrai.trainer.rollout.setup import configure_rollout
from astrai.trainer.rollout.weight_transport import SharedWeightBuffer
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.trainer import Trainer
from tests.helpers import CHAT_TEMPLATE
from tests.trainer.conftest import (
    make_online_lr_scheduler,
    make_online_model,
    make_online_optimizer,
    make_online_value_model,
)


class InstructionDataset(Dataset):
    """Toy instruction/input dataset for online RL rollout.

    Each sample has an ``instruction`` and an optional ``input``; the
    RolloutGenerator renders both through the tokenizer's chat template
    so the prompt matches the SFT-trained format.
    """

    _SAMPLES = (
        {"instruction": "Hello", "input": ""},
        {"instruction": "Tell me a story", "input": "about dragons"},
        {"instruction": "Summarize", "input": "the article"},
        {"instruction": "Translate", "input": "to French: hi"},
    )

    def __init__(self, repeats=1):
        self.repeats = repeats

    def __len__(self):
        return len(self._SAMPLES) * self.repeats

    def __getitem__(self, idx):
        return dict(self._SAMPLES[idx % len(self._SAMPLES)])


class LengthRewardModel(BaseRewardModel):
    """Rewards each response by its (non-pad) token count.

    Gives the group-normalized advantage a non-degenerate signal.
    """

    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        rewards = torch.zeros(B, G)
        for i in range(B):
            for g in range(G):
                rewards[i, g] = float(len(responses[i][g]))
        return rewards


def instruction_collate_fn(batch):
    """Stack a list of instruction/input dicts into a batch dict of lists."""
    return {
        "instruction": [b["instruction"] for b in batch],
        "input": [b.get("input", "") for b in batch],
    }


_ONLINE_STRATEGIES = [
    pytest.param(
        "online_grpo",
        {"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        None,
        id="grpo",
    ),
    pytest.param("online_dpo", {"beta": 0.1, "group_size": 2}, None, id="dpo"),
    pytest.param(
        "online_ppo",
        {
            "clip_eps": 0.2,
            "kl_coef": 0.01,
            "group_size": 2,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
        },
        True,
        id="ppo",
    ),
]

_DDP_TEST_WORLD_SIZE = int(os.environ.get("ASTRAI_DDP_TEST_WORLD_SIZE", "2"))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("strategy", "strategy_kwargs", "with_critic"), _ONLINE_STRATEGIES
)
def test_online_rollout_end_to_end(
    base_test_env, strategy, strategy_kwargs, with_critic, monkeypatch
):
    """Run one epoch of online RL rollout with KV-cache-backed generation."""
    created_reference_models = []
    create_ref_model = train_context.create_ref_model

    def track_reference_model(*args, **kwargs):
        created_reference_models.append(strategy)
        return create_ref_model(*args, **kwargs)

    monkeypatch.setattr(train_context, "create_ref_model", track_reference_model)

    test_dir = base_test_env["test_dir"]
    device = base_test_env["device"]
    tokenizer = base_test_env["tokenizer"]
    model_config = base_test_env["transformer_config"]

    tokenizer.set_chat_template(CHAT_TEMPLATE)
    tokenizer.save_pretrained(test_dir)

    config_kwargs = dict(
        strategy=strategy,
        model_fn=partial(make_online_model, model_config),
        dataset=InstructionDataset(),
        optimizer_fn=make_online_optimizer,
        scheduler_fn=make_online_lr_scheduler,
        ckpt_dir=os.path.join(test_dir, "ckpt"),
        n_epoch=1,
        batch_per_device=2,
        ckpt_interval=100,
        grad_accum_steps=1,
        random_seed=42,
        device_type=device,
        dp_mode="none",
        strategy_kwargs=strategy_kwargs,
        rollout_interval=1,
        rollout_max_policy_lag=0,
        rollout_temperature=1.0,
        rollout_top_k=0,
        rollout_top_p=1.0,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        collate_fn=instruction_collate_fn,
    )
    if with_critic:
        config_kwargs["critic_model_fn"] = partial(
            make_online_value_model, model_config
        )
        config_kwargs["critic_optimizer_fn"] = make_online_optimizer
    train_config = TrainConfig(**config_kwargs)

    trainer = Trainer(train_config)
    trainer.train(param_path=test_dir)

    checkpoint_dir = os.path.join(test_dir, "ckpt", "epoch_0_step_2")
    assert os.path.isdir(checkpoint_dir)
    checkpoint = Checkpoint.load(checkpoint_dir)
    assert checkpoint.meta["policy_version"] == 2
    assert len(created_reference_models) == 1
    if with_critic:
        assert "value_model" in checkpoint.extra
        assert "value_optimizer" in checkpoint.extra
    else:
        assert "value_model" not in checkpoint.extra


def _minimal_online_config(**overrides):
    """A TrainConfig for online GRPO that only needs field overrides."""
    defaults = dict(
        strategy="online_grpo",
        model_fn=lambda: torch.nn.Linear(2, 2),
        dataset=InstructionDataset(),
        optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.0),
        scheduler_fn=lambda o: SchedulerFactory.create(
            "cosine", o, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
        ),
        reward_model_fn=LengthRewardModel,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_online_config_rejects_contradictory_policy_lag():
    """rollout_max_policy_lag below rollout_interval - 1 guarantees a fatal
    RolloutVersionError mid-training; it must fail at config time instead."""
    with pytest.raises(ValueError, match="rollout_max_policy_lag=0"):
        _minimal_online_config(rollout_interval=3, rollout_max_policy_lag=0)

    # lag == interval - 1 (including the derived default) stays valid.
    config = _minimal_online_config(rollout_interval=3, rollout_max_policy_lag=2)
    assert config.rollout_max_policy_lag == 2
    config = _minimal_online_config(rollout_interval=3)
    assert config.rollout_max_policy_lag is None

    # Offline strategies never consult the rollout window.
    config = _minimal_online_config(
        strategy="sft", rollout_interval=3, rollout_max_policy_lag=0
    )
    assert config.rollout_max_policy_lag == 0


def test_async_round_rejects_moe_before_starting_workers():
    config = _minimal_online_config(
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4"],
        rollout_interval=1,
        rollout_max_policy_lag=1,
        device_type="cuda",
        dp_mode="none",
        grad_accum_steps=1,
    )
    model = torch.nn.Linear(2, 2)
    inference_model = SimpleNamespace(
        config=SimpleNamespace(ffn_type="moe", max_position_embeddings=8)
    )
    context = SimpleNamespace(
        strategy=SimpleNamespace(supports_online=lambda: True),
        executor=SimpleNamespace(model_for_inference=lambda _model: inference_model),
        model=model,
        checkpoint=None,
        optimizer_step=0,
    )
    tokenizer_cls = SimpleNamespace(from_pretrained=lambda _path: object())
    with pytest.raises(ValueError, match="dense models only"):
        configure_rollout(
            context,
            config,
            param_path="unused",
            strategy_kwargs={"group_size": 2},
            create_ref_model=lambda **_kwargs: None,
            validate=lambda _executor: None,
            scheduler_cls=object,
            tokenizer_cls=tokenizer_cls,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    torch.cuda.device_count() < _DDP_TEST_WORLD_SIZE,
    reason=f"{_DDP_TEST_WORLD_SIZE} CUDA devices are required",
)
def test_ddp_online_grpo_end_to_end(base_test_env):
    """Run real rollout and optimizer steps on the requested DDP replicas."""
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    model_config = base_test_env["transformer_config"]

    tokenizer.set_chat_template(CHAT_TEMPLATE)
    tokenizer.save_pretrained(test_dir)

    dataset = InstructionDataset(repeats=10)
    train_config = TrainConfig(
        strategy="online_grpo",
        model_fn=partial(make_online_model, model_config),
        dataset=dataset,
        optimizer_fn=make_online_optimizer,
        scheduler_fn=make_online_lr_scheduler,
        ckpt_dir=os.path.join(test_dir, "ckpt"),
        n_epoch=1,
        batch_per_device=1,
        ckpt_interval=100,
        grad_accum_steps=1,
        random_seed=42,
        device_type="cuda",
        dp_size=_DDP_TEST_WORLD_SIZE,
        dp_mode="ddp",
        strategy_kwargs={"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        rollout_interval=1,
        rollout_temperature=1.0,
        rollout_top_k=0,
        rollout_top_p=1.0,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        collate_fn=instruction_collate_fn,
    )

    Trainer(train_config).train(param_path=test_dir)

    expected_steps = (len(dataset) + _DDP_TEST_WORLD_SIZE - 1) // _DDP_TEST_WORLD_SIZE
    assert Path(test_dir, "ckpt", f"epoch_0_step_{expected_steps}").is_dir()


@pytest.mark.integration
@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="five CUDA devices required")
def test_async_round_online_grpo_five_gpus(base_test_env, monkeypatch):
    """Four simultaneous replicas feed one optimizer update per round."""
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    model_config = base_test_env["transformer_config"]
    tokenizer.set_chat_template(CHAT_TEMPLATE)
    tokenizer.save_pretrained(test_dir)

    seen = []
    pools = []
    original_init = AsyncRoundCoordinator.__init__
    original_collect = AsyncRoundCoordinator.collect_round

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        pools.append(self)

    def tracked_collect(self, handle):
        result = original_collect(self, handle)
        seen.append(result.policy_version)
        return result

    monkeypatch.setattr(AsyncRoundCoordinator, "__init__", tracked_init)
    monkeypatch.setattr(AsyncRoundCoordinator, "collect_round", tracked_collect)
    config = TrainConfig(
        strategy="online_grpo",
        model_fn=partial(make_online_model, model_config),
        dataset=Subset(InstructionDataset(repeats=3), range(10)),
        optimizer_fn=make_online_optimizer,
        scheduler_fn=make_online_lr_scheduler,
        ckpt_dir=os.path.join(test_dir, "ckpt"),
        n_epoch=1,
        batch_per_device=4,
        grad_accum_steps=1,
        device_type="cuda",
        dp_mode="none",
        strategy_kwargs={"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4"],
        rollout_interval=1,
        rollout_max_policy_lag=1,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        collate_fn=instruction_collate_fn,
    )
    Trainer(config).train(param_path=test_dir)

    checkpoint = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_0_step_3"))
    assert checkpoint.meta["policy_version"] == 3
    assert checkpoint.consumed_samples == 10
    assert seen == [0, 0, 1]
    assert len({process.pid for process in pools[0]._processes}) == 4
    assert set(pools[0].worker_cuda_graph_enabled) == {
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:4",
    }
    assert all(
        isinstance(enabled, bool)
        for enabled in pools[0].worker_cuda_graph_enabled.values()
    )

    resume_config = replace(config, n_epoch=2)
    Trainer(resume_config).train(
        param_path=os.path.join(test_dir, "ckpt", "epoch_0_step_3"), resume=True
    )
    resumed = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_1_step_6"))
    assert resumed.meta["policy_version"] == 6
    assert resumed.consumed_samples == 20


@pytest.mark.integration
@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="five CUDA devices required")
def test_async_round_snapshot_failure_checkpoint_resumes_once(
    base_test_env, monkeypatch
):
    test_dir = base_test_env["test_dir"]
    tokenizer = base_test_env["tokenizer"]
    model_config = base_test_env["transformer_config"]
    tokenizer.set_chat_template(CHAT_TEMPLATE)
    tokenizer.save_pretrained(test_dir)
    config = TrainConfig(
        strategy="online_grpo",
        model_fn=partial(make_online_model, model_config),
        dataset=Subset(InstructionDataset(repeats=2), range(8)),
        optimizer_fn=make_online_optimizer,
        scheduler_fn=make_online_lr_scheduler,
        ckpt_dir=os.path.join(test_dir, "ckpt"),
        n_epoch=1,
        batch_per_device=4,
        grad_accum_steps=1,
        device_type="cuda",
        dp_mode="none",
        strategy_kwargs={"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        rollout_mode="async_round",
        rollout_devices=["cuda:1", "cuda:2", "cuda:3", "cuda:4"],
        rollout_interval=1,
        rollout_max_policy_lag=1,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        collate_fn=instruction_collate_fn,
    )
    original_snapshot = SharedWeightBuffer.snapshot

    def fail_after_commit(self, version):
        if version == 1:
            raise RuntimeError("injected staging failure")
        return original_snapshot(self, version)

    monkeypatch.setattr(SharedWeightBuffer, "snapshot", fail_after_commit)
    with pytest.raises(WeightSnapshotError, match="optimizer committed"):
        Trainer(config).train(param_path=test_dir)
    checkpoint_path = os.path.join(test_dir, "ckpt", "epoch_0_step_1")
    checkpoint = Checkpoint.load(checkpoint_path)
    assert checkpoint.meta["policy_version"] == 1
    assert checkpoint.meta["optimizer_step"] == 1
    assert checkpoint.consumed_samples == 4

    monkeypatch.setattr(SharedWeightBuffer, "snapshot", original_snapshot)
    Trainer(config).train(param_path=checkpoint_path, resume=True)
    resumed = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_0_step_2"))
    assert resumed.meta["policy_version"] == 2
    assert resumed.meta["optimizer_step"] == 2
    assert resumed.consumed_samples == 8


def test_rollout_val_overrides_drop_unset_fields():
    """Only explicitly set validation fields enter the replace() dict."""
    config = _minimal_online_config(rollout_val_temperature=0.0)
    assert config.rollout_val_overrides() == {"temperature": 0.0}

    config = _minimal_online_config(
        rollout_val_temperature=0.0,
        rollout_val_top_p=0.95,
        rollout_val_top_k=0,
        rollout_val_max_tokens=256,
        rollout_val_group_size=1,
    )
    assert config.rollout_val_overrides() == {
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 0,
        "max_tokens": 256,
        "group_size": 1,
    }

    # Unset fields stay legal (inherit) while out-of-range values are
    # rejected at config time; temperature=0 (greedy) is val-only legal.
    with pytest.raises(ValueError, match="rollout_val_top_p"):
        _minimal_online_config(rollout_val_top_p=0.0)
    with pytest.raises(ValueError, match="rollout_val_group_size"):
        _minimal_online_config(rollout_val_group_size=0)
