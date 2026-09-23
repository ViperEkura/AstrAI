"""Tests for InferenceEngine.score (teacher-forced log-likelihood).

The scorer is the engine's entry point for every log-likelihood metric, so the
properties that matter are: it agrees with an independent causal forward, the
scored span does not depend on what follows it, per-token sums to the total,
and the argument validation holds.  The causality property is the one whose
absence let a 2-D attention mask silently make MMLU/HellaSwag bidirectional.
"""

import pytest
import torch

from astrai.inference import build_engine
from tests.helpers import FakeTokenizer, make_model


@pytest.fixture
def engine():
    model, _ = make_model("cpu", max_position_embeddings=64, vocab_size=128)
    eng = build_engine(
        model=model,
        tokenizer=FakeTokenizer(),
        device=None,
        dtype=None,
        max_batch_size=4,
        max_seq_len=64,
    )
    try:
        yield eng
    finally:
        eng.shutdown()


def _span_logprob(model, ids, cont_start, cont_len, offset=0):
    """Summed log-prob of ``ids[cont_start : cont_start + cont_len]``.

    ``offset`` shifts the read position on purpose, so a caller can show that an
    off-by-one would give a different answer.
    """
    with torch.inference_mode():
        logits = model(torch.tensor([ids]), input_mask=None)["logits"]
    lp = torch.nn.functional.log_softmax(logits[0].float(), dim=-1)
    return sum(
        lp[cont_start - 1 + offset + j, ids[cont_start + j]].item()
        for j in range(cont_len)
    )


def test_score_matches_an_independent_causal_forward(engine):
    prompt, cont = [3, 5, 7, 11, 13], [17, 19, 23]
    assert engine.score(prompt, cont) == pytest.approx(
        _span_logprob(engine.model, prompt + cont, len(prompt), len(cont)), abs=2e-3
    )


def test_scored_span_does_not_depend_on_later_tokens(engine):
    """Read the same span from a sequence that continues past it.

    A causal model gives the span the same log-probability there, so this pins
    both causality and the position arithmetic.  The off-by-one control shows
    the assertion can actually fail.
    """
    prompt, cont = [3, 5, 7, 11], [17, 19]
    longer = prompt + cont + [29, 31, 37]
    assert engine.score(prompt, cont) == pytest.approx(
        _span_logprob(engine.model, longer, len(prompt), len(cont)), abs=2e-3
    )
    assert engine.score(prompt, cont) != pytest.approx(
        _span_logprob(engine.model, longer, len(prompt), len(cont), offset=-1), abs=2e-3
    )


def test_per_token_sums_to_the_total(engine):
    prompt, cont = [3, 5, 7, 11, 13], [17, 19, 23]
    per_token = engine.score(prompt, cont, per_token=True)
    assert len(per_token) == len(cont)
    assert sum(per_token) == pytest.approx(engine.score(prompt, cont), abs=1e-6)


def test_adding_continuation_tokens_can_only_lower_the_score(engine):
    prompt, cont = [3, 5, 7, 11, 13], [17, 19, 23]
    assert engine.score(prompt, cont) < engine.score(prompt, cont[:1])


def test_batch_matches_single_pair_calls(engine):
    pairs = [([3, 5, 7, 11], [13]), ([3, 5, 7, 11], [17]), ([2, 4, 6, 8], [10])]
    batched = engine.score([p for p, _ in pairs], [c for _, c in pairs])
    assert len(batched) == len(pairs)
    for (prompt, cont), got in zip(pairs, batched):
        assert got == pytest.approx(engine.score(prompt, cont), abs=2e-3)


def test_ids_and_strings_agree(engine):
    """A string pair must score the same as its tokenized equivalent."""
    text_ids = engine.tokenizer.encode("ab")
    cont_ids = engine.tokenizer.encode("c", add_special_tokens=False)
    text_ids = text_ids[0] if isinstance(text_ids[0], list) else text_ids
    cont_ids = cont_ids[0] if isinstance(cont_ids[0], list) else cont_ids
    assert engine.score("ab", "c") == pytest.approx(
        engine.score(text_ids, cont_ids), abs=2e-3
    )


def test_rejects_bad_arguments(engine):
    with pytest.raises(ValueError):
        engine.score([], [1])
    with pytest.raises(ValueError):
        engine.score([1, 2], [])
    with pytest.raises(ValueError):
        engine.score([[1, 2], [3, 4]], [[5]])  # batch length mismatch


def test_unscorable_pair_returns_none(engine):
    assert engine.score(list(range(3, 70)), [2]) is None
