"""Unit tests for inference sampling strategies."""

import torch

from astrai.inference.sample import (
    FrequencyPenaltyStrategy,
    SamplingPipeline,
    TemperatureStrategy,
    TopKStrategy,
    TopPStrategy,
    sample,
)


def test_temperature_scalar():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = TemperatureStrategy(0.5)
    result = s.apply(logits.clone())
    assert torch.allclose(result, logits / 0.5)


def test_temperature_skip_when_one():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = TemperatureStrategy(1.0)
    result = s.apply(logits.clone())
    assert torch.equal(result, logits)


def test_temperature_per_sample_tensor():
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    s = TemperatureStrategy(torch.tensor([0.5, 0.5]))
    result = s.apply(logits.clone())
    assert torch.allclose(result, logits / 0.5)


def test_top_k_keeps_top():
    logits = torch.tensor([[0.1, 0.5, 0.3, 0.9, 0.2]])
    s = TopKStrategy(top_k=2)
    result = s.apply(logits.clone(), filter_value=-1e9)
    kept = (result > -1e9).sum().item()
    assert kept == 2


def test_top_k_skip_when_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = TopKStrategy(top_k=0)
    result = s.apply(logits.clone())
    assert torch.equal(result, logits)


def test_top_k_batch_tensor():
    """Each row respects its own top_k."""
    logits = torch.tensor([[0.1, 0.5, 0.3], [0.9, 0.2, 0.1]])
    s = TopKStrategy(top_k=torch.tensor([2, 1]))
    result = s.apply(logits.clone(), filter_value=-1e9)
    assert (result[0] > -1e9).sum() == 2
    assert (result[1] > -1e9).sum() == 1


def test_top_p_nucleus_filtering():
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
    s = TopPStrategy(top_p=0.5)
    result = s.apply(logits.clone(), filter_value=-1e9)
    kept = (result > -1e9).sum().item()
    assert kept >= 1


def test_top_p_skip_when_one():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = TopPStrategy(top_p=1.0)
    result = s.apply(logits.clone())
    assert torch.equal(result, logits)


def test_top_p_filter_all_except_max_when_zero():
    logits = torch.tensor([[0.1, 0.5, 0.3, 0.9, 0.2]])
    s = TopPStrategy(top_p=0.0)
    result = s.apply(logits.clone(), filter_value=-1e9)
    kept = (result > -1e9).sum().item()
    assert kept == 1


def test_sampling_pipeline_composes_strategies():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pipeline = SamplingPipeline(
        [
            TemperatureStrategy(0.8),
            TopKStrategy(3),
            TopPStrategy(0.95),
        ]
    )
    result = pipeline.apply(logits.clone(), filter_value=-1e9)
    kept = (result > -1e9).sum().item()
    assert 1 <= kept <= 3


def test_sampling_pipeline_sample_returns_valid_token():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pipeline = SamplingPipeline(
        [
            TemperatureStrategy(0.8),
            TopKStrategy(3),
            TopPStrategy(0.95),
        ]
    )
    tokens = pipeline.sample(logits)
    assert tokens.shape == (1,)
    assert 0 <= tokens[0] < logits.size(-1)


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


def test_frequency_penalty_noop_when_zero():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    input_ids = torch.tensor([[0, 2]])
    s = FrequencyPenaltyStrategy(penalty=0.0)
    result = s.apply(logits.clone(), input_ids=input_ids)
    assert torch.equal(result, logits)


def test_frequency_penalty_noop_when_no_input_ids():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    s = FrequencyPenaltyStrategy(penalty=0.5)
    result = s.apply(logits.clone())
    assert torch.equal(result, logits)


def test_frequency_penalty_single_occurrence():
    logits = torch.tensor([[4.0, 1.0, 2.0]])
    input_ids = torch.tensor([[0, 2]])
    input_mask = torch.tensor([[True, True]])
    s = FrequencyPenaltyStrategy(penalty=0.5)
    result = s.apply(logits.clone(), input_ids=input_ids, input_mask=input_mask)
    assert result[0, 0] == 3.5
    assert result[0, 1] == 1.0
    assert result[0, 2] == 1.5


def test_frequency_penalty_multiple_occurrences():
    logits = torch.tensor([[4.0, 1.0, 2.0]])
    input_ids = torch.tensor([[0, 2, 0]])
    input_mask = torch.tensor([[True, True, True]])
    s = FrequencyPenaltyStrategy(penalty=0.5)
    result = s.apply(logits.clone(), input_ids=input_ids, input_mask=input_mask)
    assert result[0, 0] == 3.0
    assert result[0, 1] == 1.0
    assert result[0, 2] == 1.5


def test_frequency_penalty_respects_padding_mask():
    logits = torch.tensor([[4.0, 1.0, 2.0]])
    input_ids = torch.tensor([[0, 2, 0]])
    input_mask = torch.tensor([[True, True, False]])
    s = FrequencyPenaltyStrategy(penalty=0.5)
    result = s.apply(logits.clone(), input_ids=input_ids, input_mask=input_mask)
    assert result[0, 0] == 3.5
    assert result[0, 1] == 1.0
    assert result[0, 2] == 1.5


def test_frequency_penalty_batch_tensor():
    logits = torch.tensor(
        [
            [4.0, 1.0, 2.0],
            [3.0, 5.0, 1.0],
        ]
    )
    input_ids = torch.tensor([[0, 2, 0], [1, 1, 0]])
    input_mask = torch.tensor([[True, True, True], [True, True, False]])
    s = FrequencyPenaltyStrategy(penalty=torch.tensor([0.5, 1.0]))
    result = s.apply(logits.clone(), input_ids=input_ids, input_mask=input_mask)
    assert result[0, 0] == 3.0
    assert result[0, 2] == 1.5
    assert result[1, 1] == 3.0


def test_frequency_penalty_negative_penalty_boosts_repeats():
    logits = torch.tensor([[4.0, 1.0, 2.0]])
    input_ids = torch.tensor([[0, 0]])
    input_mask = torch.tensor([[True, True]])
    s = FrequencyPenaltyStrategy(penalty=-0.5)
    result = s.apply(logits.clone(), input_ids=input_ids, input_mask=input_mask)
    assert result[0, 0] == 5.0


def test_frequency_penalty_in_pipeline():
    logits = torch.tensor([[5.0, 1.0, 2.0, 3.0]])
    input_ids = torch.tensor([[0, 2, 0]])
    input_mask = torch.tensor([[True, True, True]])
    pipeline = SamplingPipeline(
        [
            TemperatureStrategy(1.0),
            FrequencyPenaltyStrategy(0.5),
        ]
    )
    result = pipeline.apply(logits.clone(), input_ids=input_ids, input_mask=input_mask)
    assert result[0, 0] == 4.0
    assert result[0, 2] == 1.5


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
