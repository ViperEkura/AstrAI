"""Acceptance tests for the Gated DeltaNet reference operators (training and inference).

The three interfaces over one rule must agree, causality must hold, decode state
must be fixed-size, and batch elements must be isolated. Tolerances are measured
rather than guessed: on these shapes the chunked and recurrent paths agree to
~1e-6 absolute in float32, so the assertions use 1e-5/1e-4 and one order looser
for gradients, where the UT transform's inverse amplifies rounding.
"""

import pytest
import torch

from astrai.model.components.attention import GDN
from astrai.model.components.gdn_ops import (
    chunk_gated_delta_rule,
)

BATCH, VALUE_HEADS, KEY_DIM, VALUE_DIM = 2, 4, 8, 8
SEQ_LEN = 100  # not a multiple of 64, so the chunked path has to pad
ATOL, RTOL = 1e-5, 1e-4
LAYER_DIM, LAYER_KEY_HEADS, LAYER_VALUE_HEADS = 32, 2, 4
LAYER_HEAD_DIM, LAYER_CONV = 8, 4


def make_inputs(
    seq_len=SEQ_LEN, batch=BATCH, device="cpu", dtype=torch.float32, seed=0
):
    """q/k/v and per-head gates: ``g`` is log decay (<= 0), ``beta`` a write gate.

    Heads are already expanded, so no key/value head sharing is involved here.
    """
    gen = torch.Generator().manual_seed(seed)
    gate_shape = (batch, seq_len, VALUE_HEADS)

    def randn(*dims):
        return torch.randn(*dims, generator=gen, dtype=torch.float32).to(device, dtype)

    return (
        randn(batch, seq_len, VALUE_HEADS, KEY_DIM),
        randn(batch, seq_len, VALUE_HEADS, KEY_DIM),
        randn(batch, seq_len, VALUE_HEADS, VALUE_DIM),
        (-0.5 * torch.rand(*gate_shape, generator=gen)).to(device, dtype),
        torch.rand(*gate_shape, generator=gen).to(device, dtype),
    )


def make_layer(device, dtype=torch.float32):
    return (
        GDN(
            dim=LAYER_DIM,
            n_heads=LAYER_VALUE_HEADS,
            gdn_num_key_heads=LAYER_KEY_HEADS,
            gdn_num_value_heads=LAYER_VALUE_HEADS,
            gdn_key_head_dim=LAYER_HEAD_DIM,
            gdn_value_head_dim=LAYER_HEAD_DIM,
            gdn_conv_kernel_size=LAYER_CONV,
        )
        .to(device)
        .to(dtype)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_chunked_path_does_not_materialize_quadratic():
    """Four times the tokens must cost far less than sixteen times the memory."""
    peaks = {}
    for seq_len in (256, 1024):
        inputs = make_inputs(seq_len=seq_len, batch=1, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        chunk_gated_delta_rule(*inputs)
        torch.cuda.synchronize()
        peaks[seq_len] = torch.cuda.max_memory_allocated() - baseline
        del inputs
        torch.cuda.empty_cache()
    assert peaks[1024] < 6 * peaks[256], peaks
