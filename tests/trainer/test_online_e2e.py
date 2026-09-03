"""End-to-end integration tests for online GRPO/DPO rollout."""

import os
from functools import partial
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from astrai.config import TrainConfig
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer import train_context
from astrai.trainer.rollout import BaseRewardModel
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.trainer import Trainer
from tests.helpers import CHAT_TEMPLATE


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


def _model_fn(model_config):
    return AutoRegressiveLM(model_config).to(dtype=torch.float32)


def _optimizer_fn(m):
    return torch.optim.AdamW(m.parameters(), lr=1e-4)


def _scheduler_fn(optim):
    return SchedulerFactory.create(
        "cosine", optim, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
    )


_ONLINE_STRATEGIES = [
    pytest.param(
        "online_grpo",
        {"clip_eps": 0.2, "kl_coef": 0.01, "group_size": 2},
        id="grpo",
    ),
    pytest.param("online_dpo", {"beta": 0.1, "group_size": 2}, id="dpo"),
]

_DDP_TEST_WORLD_SIZE = int(os.environ.get("ASTRAI_DDP_TEST_WORLD_SIZE", "2"))


@pytest.mark.integration
@pytest.mark.parametrize(("strategy", "strategy_kwargs"), _ONLINE_STRATEGIES)
def test_online_rollout_end_to_end(
    base_test_env, strategy, strategy_kwargs, monkeypatch
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

    train_config = TrainConfig(
        strategy=strategy,
        model_fn=partial(_model_fn, model_config),
        dataset=InstructionDataset(),
        optimizer_fn=_optimizer_fn,
        scheduler_fn=_scheduler_fn,
        ckpt_dir=os.path.join(test_dir, "ckpt"),
        n_epoch=1,
        batch_per_device=2,
        ckpt_interval=100,
        grad_accum_steps=1,
        random_seed=42,
        device_type=device,
        nprocs=1,
        parallel_mode="none",
        strategy_kwargs=strategy_kwargs,
        rollout_interval=1,
        rollout_temperature=1.0,
        rollout_top_k=0,
        rollout_top_p=1.0,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        collate_fn=instruction_collate_fn,
    )

    trainer = Trainer(train_config)
    trainer.train(param_path=test_dir)

    assert os.path.isdir(os.path.join(test_dir, "ckpt"))
    assert len(created_reference_models) == 1


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
        model_fn=partial(_model_fn, model_config),
        dataset=dataset,
        optimizer_fn=_optimizer_fn,
        scheduler_fn=_scheduler_fn,
        ckpt_dir=os.path.join(test_dir, "ckpt"),
        n_epoch=1,
        batch_per_device=1,
        ckpt_interval=100,
        grad_accum_steps=1,
        random_seed=42,
        device_type="cuda",
        nprocs=_DDP_TEST_WORLD_SIZE,
        parallel_mode="ddp",
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
