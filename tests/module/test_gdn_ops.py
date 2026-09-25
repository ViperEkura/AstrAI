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
    recurrent_gated_delta_rule,
    recurrent_gated_delta_rule_step,
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


@pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
def test_chunked_matches_recurrent(chunk_size, device):
    """Chunked training path and the per-token reference agree on output and state.

    chunk_size 128 exceeds the sequence length, which forces heavy padding.
    """
    inputs = make_inputs(device=device)
    reference, reference_state = recurrent_gated_delta_rule(
        *inputs, output_final_state=True
    )
    output, state = chunk_gated_delta_rule(
        *inputs, chunk_size=chunk_size, output_final_state=True
    )
    torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(state, reference_state, atol=ATOL, rtol=RTOL)


def test_decode_steps_match_full_sequence(device):
    """Threading the state through single-token steps reproduces the full pass."""
    inputs = make_inputs(device=device)
    reference, reference_state = recurrent_gated_delta_rule(
        *inputs, output_final_state=True
    )
    state = torch.zeros(BATCH, VALUE_HEADS, KEY_DIM, VALUE_DIM, device=device)
    outputs = []
    for index in range(SEQ_LEN):
        out, state = recurrent_gated_delta_rule_step(
            inputs[0][:, index],
            inputs[1][:, index],
            inputs[2][:, index],
            inputs[3][:, index],
            inputs[4][:, index],
            state,
        )
        outputs.append(out)
    stepped = torch.stack(outputs, dim=1)
    torch.testing.assert_close(stepped, reference, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(state, reference_state, atol=ATOL, rtol=RTOL)


def test_decode_state_is_fixed_size(device):
    """Decode state must not grow with history, and must be resettable."""
    inputs = make_inputs(device=device)
    expected = (BATCH, VALUE_HEADS, KEY_DIM, VALUE_DIM)

    def run(steps):
        state = torch.zeros(*expected, device=device)
        shapes = set()
        for index in range(steps):
            _, state = recurrent_gated_delta_rule_step(
                *(tensor[:, index] for tensor in inputs), state
            )
            shapes.add(tuple(state.shape))
        return state, shapes

    after_one, shapes = run(1)
    after_all, longer_shapes = run(SEQ_LEN)
    assert shapes == {expected}
    assert longer_shapes == {expected}
    # A single step cannot already reach the 100-step state, so the state is
    # actually being carried rather than reset or ignored.
    assert not torch.allclose(after_one, after_all)
    # Re-running from the same zeros reproduces the same state: no hidden state.
    repeated, _ = run(SEQ_LEN)
    torch.testing.assert_close(after_all, repeated)


def test_batch_elements_are_isolated(device):
    inputs = make_inputs(device=device)
    batched, _ = chunk_gated_delta_rule(*inputs)
    single, _ = chunk_gated_delta_rule(*(x[:1] for x in inputs))
    torch.testing.assert_close(batched[:1], single, atol=ATOL, rtol=RTOL)

    perturbed = [x.clone() for x in inputs]
    perturbed[0][1] += 5.0
    other, _ = chunk_gated_delta_rule(*perturbed)
    torch.testing.assert_close(other[:1], batched[:1], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("boundary", [32, 64, 96])
def test_prefix_does_not_depend_on_suffix(boundary, device):
    """Changing a suffix must leave everything before it untouched."""
    inputs = make_inputs(device=device)
    reference, _ = chunk_gated_delta_rule(*inputs)
    changed = [x.clone() for x in inputs]
    for tensor in changed:
        tensor[:, boundary:] = torch.randn_like(tensor[:, boundary:])
    output, _ = chunk_gated_delta_rule(*changed)
    torch.testing.assert_close(
        output[:, :boundary], reference[:, :boundary], atol=ATOL, rtol=RTOL
    )


def test_gradients_match_between_interfaces(device):
    """The chunked path's autograd must agree with the per-token path's."""
    inputs = make_inputs(seq_len=64, device=device)
    for tensor in inputs:
        tensor.requires_grad_(True)
    recurrent_gated_delta_rule(*inputs)[0].square().sum().backward()
    expected = [tensor.grad.clone() for tensor in inputs]

    chunked_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    chunk_gated_delta_rule(*chunked_inputs)[0].square().sum().backward()
    for actual, want in zip(chunked_inputs, expected):
        torch.testing.assert_close(actual.grad, want, atol=1e-4, rtol=1e-3)


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


def test_layer_prefill_decode_matches_training_forward(device):
    """The headline inference check: prefill + decode reproduces the training pass."""
    layer = make_layer(device)
    layer.eval()
    x = torch.randn(2, 48, LAYER_DIM, device=device)
    with torch.no_grad():
        trained = layer(x)
        output, state = layer.prefill(x[:, :16])
        pieces = [output]
        for index in range(16, 48):
            step, state = layer.decode_step(x[:, index : index + 1], state)
            pieces.append(step)
    torch.testing.assert_close(torch.cat(pieces, dim=1), trained, atol=ATOL, rtol=RTOL)


def test_layer_decode_from_scratch_matches_prefill(device):
    """Decoding with no prior state must equal prefilling the same tokens."""
    layer = make_layer(device)
    layer.eval()
    x = torch.randn(1, 12, LAYER_DIM, device=device)
    with torch.no_grad():
        prefilled, _ = layer.prefill(x)
        state = None
        pieces = []
        for index in range(x.shape[1]):
            step, state = layer.decode_step(x[:, index : index + 1], state)
            pieces.append(step)
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), prefilled, atol=ATOL, rtol=RTOL
    )


def test_layer_prefill_continues_existing_state(device):
    """A second prefill must continue the first one's state, not restart it."""
    layer = make_layer(device)
    layer.eval()
    x = torch.randn(1, 24, LAYER_DIM, device=device)
    with torch.no_grad():
        whole, _ = layer.prefill(x)
        first, state = layer.prefill(x[:, :12])
        second, _ = layer.prefill(x[:, 12:], state)
    torch.testing.assert_close(
        torch.cat((first, second), dim=1), whole, atol=ATOL, rtol=RTOL
    )


def test_layer_state_is_fixed_size_after_long_decode(device):
    layer = make_layer(device)
    layer.eval()
    x = torch.randn(1, 64, LAYER_DIM, device=device)
    with torch.no_grad():
        _, state = layer.prefill(x[:, :1])
        for index in range(1, 64):
            _, state = layer.decode_step(x[:, index : index + 1], state)
    assert state.recurrent.shape == (
        1,
        LAYER_VALUE_HEADS,
        LAYER_HEAD_DIM,
        LAYER_HEAD_DIM,
    )
    assert state.conv.shape == (1, LAYER_CONV - 1, layer.conv_channels)
    assert state.recurrent.dtype == torch.float32


def test_layer_forward_rejects_non_right_padding(device):
    layer = make_layer(device)
    x = torch.randn(2, 8, LAYER_DIM, device=device)
    mask = torch.ones(2, 1, 1, 8, dtype=torch.bool, device=device)
    mask[0, 0, 0, 0] = False
    with pytest.raises(ValueError, match="right padding only"):
        layer(x, attn_mask=mask)


def test_layer_rejects_multi_token_decode(device):
    layer = make_layer(device)
    with pytest.raises(ValueError, match="one token per sequence"):
        layer.decode_step(torch.randn(1, 2, LAYER_DIM, device=device))


def test_chunked_rejects_non_positive_chunk_size(device):
    inputs = make_inputs(seq_len=8, device=device)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_gated_delta_rule(*inputs, chunk_size=0)
