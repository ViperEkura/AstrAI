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

_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "SYSTEM: {{ message['content'] }}\n"
    "{% elif message['role'] == 'user' %}"
    "USER: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "ASSISTANT: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT: {% endif %}"
)


class InstructionDataset(Dataset):
    """Toy instruction/input dataset for online RL rollout.

    Each sample has an ``instruction`` and an optional ``input``; the
    RolloutGenerator renders both through the tokenizer's chat template
    so the prompt matches the SFT-trained format.
    """

    _SAMPLES = [
        {"instruction": "Hello", "input": ""},
        {"instruction": "Tell me a story", "input": "about dragons"},
        {"instruction": "Summarize", "input": "the article"},
        {"instruction": "Translate", "input": "to French: hi"},
    ]

    def __len__(self):
        return len(self._SAMPLES)

    def __getitem__(self, idx):
        return dict(self._SAMPLES[idx])


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


@pytest.mark.integration
def test_online_dpo_end_to_end(base_test_env):
    """Run one epoch of online DPO with KV-cache-backed rollout."""
    test_dir = base_test_env["test_dir"]
    device = base_test_env["device"]
    tokenizer = base_test_env["tokenizer"]
    model_config = base_test_env["transformer_config"]

    # Equip tokenizer with a chat template so RolloutGenerator can
    # render instruction/input via apply_chat_template.
    tokenizer.set_chat_template(_CHAT_TEMPLATE)
    tokenizer.save_pretrained(test_dir)

    model_fn = partial(_model_fn, model_config)
    optimizer_fn = _optimizer_fn
    scheduler_fn = _scheduler_fn

    dataset = InstructionDataset()

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
        collate_fn=instruction_collate_fn,
    )

    trainer = Trainer(train_config)
    trainer.train(param_path=test_dir)

    assert os.path.isdir(os.path.join(test_dir, "ckpt"))
