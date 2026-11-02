"""Shared fixtures for extension tests."""

import pytest
import torch

import astrai.extension.runtime.context as context
from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.extension import kernel
from astrai.extension.runtime.loader import is_available
from astrai.model.autoregressive_lm import AutoRegressiveLM


@pytest.fixture(autouse=True)
def _reset_dispatch_state():
    """Isolate the process-level dispatch/planner state per test.

    set_op selections and the gemm planner
    configuration are process state; without this a test that touches
    them leaks into the next one.
    """
    previous_selection = context._process_choices
    try:
        yield
    finally:
        context._process_choices = previous_selection
        context.invalidate_selection_cache()
        if is_available("gemm"):
            kernel.gemm.set_table("")
            kernel.gemm.set_planner("")  # back to the shipped default
            # staging too: the planner prices per staging variant (gemm.cuh
            # cost_of branches on q.tma), so a test leaving tma disabled would
            # silently move every later probe to the cp.async cost form
            kernel.gemm.set_staging(tma=True, mx=True)


CFG = dict(
    vocab_size=1000,
    hidden_size=128,
    num_attention_heads=2,
    num_key_value_heads=1,
    intermediate_size=256,
    max_position_embeddings=64,
    num_hidden_layers=2,
    rms_norm_eps=1e-5,
    attn_type="gqa",
    ffn_type="mlp",
)


@pytest.fixture
def cuda_model():
    config = AutoRegressiveLMConfig(**CFG)
    model = AutoRegressiveLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    return model, config
