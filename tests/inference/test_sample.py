"""Behavioral tests for the sampling entry point."""

from unittest.mock import patch

import pytest
import torch

from astrai.inference.worker.sample import SamplingMeta, sample


def test_module_sample_shortcut():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    tokens = sample(logits, temperature=0.8, top_k=3, top_p=0.95)
    assert tokens.shape == (1,)
    assert 0 <= tokens[0] < logits.size(-1)


def test_module_sample_batch():
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    tokens = sample(logits, temperature=0.8, top_k=3, top_p=0.95)
    assert tokens.shape == (2,)
    for t in tokens:
        assert 0 <= t < logits.size(-1)


def test_sample_with_frequency_penalty():
    logits = torch.tensor([[5.0, 1.0, 2.0, 3.0]])
    input_ids = torch.tensor([[0, 2, 0]])
    input_mask = torch.tensor([[True, True, True]])
    tokens = sample(
        logits,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        frequency_penalty=0.5,
        input_ids=input_ids,
        input_mask=input_mask,
    )
    assert tokens.shape == (1,)
    assert 0 <= tokens[0] < logits.size(-1)


def test_sample_return_logprobs_shape():
    """``return_logprobs=True`` returns ``[batch]`` logprobs aligned to tokens."""
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    out = sample(logits, temperature=1.0, return_logprobs=True)
    tokens, logprobs = out
    assert tokens.shape == (2,)
    assert logprobs.shape == (2,)


def test_sample_return_logprobs_nonpositive():
    """Probabilities never exceed 1, so logprobs are always ≤ 0."""
    torch.manual_seed(0)
    logits = torch.randn(4, 50)
    _, logprobs = sample(
        logits, temperature=0.8, top_k=20, top_p=0.9, return_logprobs=True
    )
    assert torch.all(logprobs <= 1e-5)


def test_sample_return_logprobs_greedy_path():
    """Greedy decode (temperature 0) also returns logprobs."""
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    tokens, logprobs = sample(logits, temperature=0.0, return_logprobs=True)
    assert tokens[0].item() == 1
    # log p(token=1) should equal log_softmax(logits)[1]
    expected = torch.log_softmax(logits.float(), dim=-1)[0, 1]
    assert torch.allclose(logprobs[0], expected, atol=1e-5)


def test_sample_return_logprobs_matches_manual_computation():
    """Returned logprob equals log_softmax(raw_logits)[token].

    Logprobs live in the raw (pre-strategy) model distribution so they
    line up with training-side policy logprobs for RL importance ratios.
    """
    torch.manual_seed(1)
    logits = torch.randn(2, 30)
    tokens, logprobs = sample(logits, temperature=0.7, top_p=0.95, return_logprobs=True)
    expected = torch.gather(
        torch.log_softmax(logits.float(), dim=-1),
        -1,
        tokens.unsqueeze(-1),
    ).squeeze(-1)
    assert torch.allclose(logprobs, expected, atol=1e-5)


def test_greedy_respects_frequency_penalty():
    """temperature=0 must not silently skip the frequency penalty."""
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 4.0, 3.0]])

    plain = sample(logits.clone(), temperature=0.0)
    assert plain.tolist() == [0]

    penalized = sample(
        logits.clone(),
        temperature=0.0,
        frequency_penalty=2.0,
        input_ids=torch.tensor([[0, 0, 0, 0]]),
    )
    # Token 0 saw four occurrences: 5 - 2*4 < 4, so the argmax flips.
    assert penalized.tolist() == [1]


def test_seeded_sampling_is_independent_of_global_rng_and_order():
    logits = torch.randn(4, 128)
    seeds = [11, 22, 33, 44]
    before = torch.get_rng_state()
    tokens, logprobs = sample(logits.clone(), seeds=seeds, return_logprobs=True)
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(999)
    order = [3, 1, 0, 2]
    other, other_logprobs = sample(
        logits[order].clone(), seeds=[seeds[i] for i in order], return_logprobs=True
    )
    assert torch.equal(other, tokens[order])
    torch.testing.assert_close(other_logprobs, logprobs[order])
    single = sample(logits[2:3].clone(), seeds=[33])
    assert single.item() == tokens[2].item()


@pytest.mark.parametrize("meta_cached", [False, True])
def test_per_row_filters_penalties_and_raw_logprobs(meta_cached):
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
    ids = torch.tensor([[0, 0, 1], [3, 3, 0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    captured = []

    def draw(probabilities, num_samples):
        captured.append(probabilities.clone())
        return probabilities.argmax(-1, keepdim=True)

    with patch("torch.multinomial", side_effect=draw):
        tokens, logprobs = sample(
            logits.clone(),
            temperature=torch.tensor([0.5, 1.0]),
            top_k=torch.tensor([2, 0]),
            top_p=torch.tensor([1.0, 0.0]),
            frequency_penalty=torch.tensor([1.0, -1.0]),
            input_ids=ids,
            input_mask=mask,
            return_logprobs=True,
            meta=SamplingMeta(False, True, 2, True, True) if meta_cached else None,
        )
    # Top-k preserves ties at the cutoff: all three first-row candidates remain.
    expected = torch.softmax(
        torch.tensor(
            [
                [6.0, 8.0, 6.0, -float("inf")],
                [-float("inf"), -float("inf"), -float("inf"), 5.0],
            ]
        ),
        -1,
    )
    torch.testing.assert_close(captured[0], expected)
    torch.testing.assert_close(
        logprobs, logits.log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
    )


def test_seed_count_must_match_batch():
    with pytest.raises(ValueError, match="seeds"):
        sample(torch.zeros(2, 4), seeds=[1])
