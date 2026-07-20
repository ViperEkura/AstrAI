"""End-to-end integration test for online DPO rollout."""

import os
from functools import partial

import pytest
import torch
from torch.utils.data import Dataset

from astrai.config import TrainConfig
from astrai.model.transformer import AutoRegressiveLM
from astrai.trainer.rollout import BaseRewardModel
from astrai.trainer.schedule import SchedulerFactory
from astrai.trainer.trainer import Trainer


class PromptDataset(Dataset):
    """Toy prompt-only dataset for online RL rollout."""

    def __init__(self, n=4, seq_len=8, vocab_size=1000):
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(3, self.vocab_size, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.bool),
        }


class LengthRewardModel(BaseRewardModel):
    """Rewards each response by its (non-pad) token count.

    Enough for DPO to distinguish chosen/rejected from the rollout group.
    """

    def score(self, prompts, responses):
        B = len(prompts)
        G = len(responses[0]) if B else 0
        rewards = torch.zeros(B, G)
        for i in range(B):
            for g in range(G):
                rewards[i, g] = float(len(responses[i][g]))
        return rewards


def _model_fn(model_config):
    return AutoRegressiveLM(model_config).to(dtype=torch.float32)


def _optimizer_fn(m):
    return torch.optim.AdamW(m.parameters(), lr=1e-4)


def _scheduler_fn(optim):
    return SchedulerFactory.create(
        "cosine", optim, warmup_steps=1, lr_decay_steps=4, min_rate=0.05
    )


@pytest.mark.integration
def test_online_dpo_end_to_end(base_test_env):
    """Run one epoch of online DPO with KV-cache-backed rollout."""
    test_dir = base_test_env["test_dir"]
    device = base_test_env["device"]
    tokenizer = base_test_env["tokenizer"]
    model_config = base_test_env["transformer_config"]

    # base_test_env already wrote config.json into test_dir; we only need
    # to drop the tokenizer files so AutoTokenizer.from_pretrained works.
    tokenizer.save_pretrained(test_dir)

    model_fn = partial(_model_fn, model_config)
    optimizer_fn = _optimizer_fn
    scheduler_fn = _scheduler_fn

    dataset = PromptDataset(n=4, seq_len=8, vocab_size=model_config.vocab_size)

    train_config = TrainConfig(
        strategy="online_dpo",
        model_fn=model_fn,
        dataset=dataset,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        ckpt_dir=os.path.join(test_dir, "ckpt"),
        log_dir=os.path.join(test_dir, "logs"),
        n_epoch=1,
        batch_per_device=2,
        ckpt_interval=100,
        grad_accum_steps=1,
        random_seed=42,
        device_type=device,
        nprocs=1,
        parallel_mode="none",
        extra_kwargs={"beta": 0.1, "group_size": 2},
        rollout_interval=1,
        rollout_temperature=1.0,
        rollout_top_k=0,
        rollout_top_p=1.0,
        rollout_max_tokens=4,
        reward_model_fn=LengthRewardModel,
        collate_fn=None,
    )

    trainer = Trainer(train_config)
    trainer.train(param_path=test_dir)

    assert os.path.isdir(os.path.join(test_dir, "ckpt"))
