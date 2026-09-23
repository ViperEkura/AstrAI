"""Shared fixtures for extension tests."""

import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM

D = 64
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


@pytest.fixture(autouse=True)
def _reset_dispatch_state():
    """Isolate the process-level dispatch/planner state per test.

    set_op selections, cached record lists and the gemm planner
    configuration are process state; without this a test that touches
    them leaks into the next one.
    """
    yield
    import astrai.extension.dispatch as dispatch

    dispatch._selection = None
    dispatch.invalidate()
    from astrai.extension import ops
    from astrai.extension.loader import is_available

    if is_available("gemm"):
        ops.gemm.set_table("")
        ops.gemm.set_planner("")  # back to the shipped default
        # staging too: the planner prices per staging variant (gemm.cuh
        # cost_of branches on q.tma), so a test leaving tma disabled would
        # silently move every later probe to the cp.async cost form
        ops.gemm.set_staging(tma=True, mx=True)
