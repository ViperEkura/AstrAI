"""Request sampling streams remain stable across ordering and batch changes."""

import pytest
import torch

from astrai.inference.worker.sample import sample
from tests.support.scheduler import _make_real_scheduler


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


def test_scheduler_seeds_survive_batching_and_request_completion():
    scheduler, _, _ = _make_real_scheduler("cpu")
    prompts = [[1, 2], [3, 4, 5]]
    first = scheduler.run_batch(
        prompts, seeds=[11, 22], max_tokens=6, top_k=0, return_details=True
    )
    torch.manual_seed(999)
    reordered = scheduler.run_batch(
        list(reversed(prompts)),
        seeds=[22, 11],
        max_tokens=6,
        top_k=0,
        return_details=True,
    )
    individual = scheduler.run_batch(
        [prompts[1]], seeds=[22], max_tokens=6, top_k=0, return_details=True
    )
    assert [r.token_ids for r in first] == [r.token_ids for r in reversed(reordered)]
    assert individual[0].token_ids == first[1].token_ids
    assert all(r.error_reason is None for r in first + reordered + individual)


@pytest.mark.parametrize("seeds", [[], [True], [-1], [2**63]])
def test_bad_request_seeds_fail_before_allocating_cache(seeds):
    scheduler, _, _ = _make_real_scheduler("cpu")
    with pytest.raises(ValueError, match="seeds"):
        scheduler.run_batch([[1, 2]], seeds=seeds)
    assert not scheduler._planned
