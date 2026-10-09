"""End-to-end integration tests for online GRPO/DPO rollout."""

import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset, Subset

from astrai.config import TrainConfig
from astrai.serialization import Checkpoint
from astrai.trainer.rollout import BaseRewardModel
from astrai.trainer.rollout.async_round import AsyncRoundCoordinator
from astrai.trainer.rollout.nccl_transport import NCCLWeightChannel
from astrai.trainer.rollout.types import SamplingParams
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.trainer import Trainer
from tests.support.tokenizers import CHAT_TEMPLATE
from tests.support.trainer import (
    make_online_lr_scheduler,
    make_online_model,
    make_online_optimizer,
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
def test_async_round_version_commit_failure_checkpoint_resumes_once(
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
    original_publication = NCCLWeightChannel.mark_committed

    def fail_after_commit(self, version):
        if version == 1:
            raise RuntimeError("injected version commit failure")
        return original_publication(self, version)

    monkeypatch.setattr(NCCLWeightChannel, "mark_committed", fail_after_commit)
    with pytest.raises(RuntimeError, match="injected version commit failure"):
        Trainer(config).train(param_path=test_dir)
    checkpoint_path = os.path.join(test_dir, "ckpt", "epoch_0_step_1")
    checkpoint = Checkpoint.load(checkpoint_path)
    assert checkpoint.meta["policy_version"] == 1
    assert checkpoint.meta["optimizer_step"] == 1
    assert checkpoint.consumed_samples == 4

    monkeypatch.setattr(NCCLWeightChannel, "mark_committed", original_publication)
    Trainer(config).train(param_path=checkpoint_path, resume=True)
    resumed = Checkpoint.load(os.path.join(test_dir, "ckpt", "epoch_0_step_2"))
    assert resumed.meta["policy_version"] == 2
    assert resumed.meta["optimizer_step"] == 2
    assert resumed.consumed_samples == 8


def make_graph_rollout_model(config):
    return make_online_model(config).to(dtype=torch.bfloat16)


@pytest.mark.integration
@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="five CUDA devices required")
def test_async_seeds_match_after_worker_repartition_and_resume(base_test_env):
    tokenizer = base_test_env["tokenizer"]
    tokenizer.set_chat_template(CHAT_TEMPLATE)
    tokenizer.save_pretrained(base_test_env["test_dir"])
    config = replace(
        base_test_env["transformer_config"], hidden_size=128, intermediate_size=256
    )
    model_fn = partial(make_graph_rollout_model, config)
    source = model_fn().to("cuda:0").eval()
    batch = instruction_collate_fn([InstructionDataset()[i % 4] for i in range(8)])
    results = []
    for devices in (
        ["cuda:1", "cuda:2"],
        ["cuda:1", "cuda:2", "cuda:3", "cuda:4"],
        ["cuda:1", "cuda:2"],
    ):
        per_worker = 8 // len(devices)
        pool = AsyncRoundCoordinator(
            source=source,
            model_fn=model_fn,
            param_path=base_test_env["test_dir"],
            devices=devices,
            params=SamplingParams(
                group_size=2, max_tokens=6, temperature=0.7, top_k=0, top_p=1.0
            ),
            reward_model=LengthRewardModel(),
            policy_version=0,
            max_batch_size=per_worker * 2,
            max_seq_len=64,
            model_dtype=torch.bfloat16,
            max_prompts_per_worker=per_worker,
            random_seed=42,
            sample_cursor=10,
        )
        try:
            results.append(pool.collect_round(pool.submit_round(batch)))
            assert all(pool.worker_cuda_graph_enabled.values())
        finally:
            pool.close()
        assert all(not p.is_alive() for p in pool._processes)
    for other in results[1:]:
        assert torch.equal(results[0].responses, other.responses)
        assert torch.equal(results[0].response_mask, other.response_mask)
        torch.testing.assert_close(
            results[0].logprobs_old, other.logprobs_old, atol=1e-5, rtol=1e-5
        )
