from unittest.mock import MagicMock

import torch

from astrai.inference import EngineCore
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import make_rollout_config
from tests.support.tokenizers import FakeTokenizer


def make_contract_scheduler():
    scheduler, _, _ = make_real_scheduler("cpu")
    kv = MagicMock()
    scheduler.scheduler._kv_manager = kv
    scheduler.scheduler._planner.kv_manager = kv
    kv.cached_tokens.return_value = 0
    kv.extend_slots_batch.side_effect = lambda ids, positions: [True] * len(ids)
    return scheduler


def make_real_scheduler(device):
    """Build a scheduler backed by a tiny real model for run_batch tests."""
    cfg = make_rollout_config(max_position_embeddings=64)
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    tokenizer = FakeTokenizer()
    model.generation_config = {"eos_token_id": tokenizer.eos_token_id}
    scheduler = EngineCore(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_seq_len=64,
    )
    return scheduler, tokenizer, model
