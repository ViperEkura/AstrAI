"""Qualification must reject constant-length, dropped or invalid episodes."""

from copy import deepcopy

import pytest
import torch

from astrai.trainer.rollout import RawRollout
from examples.rl_reward.qualify_collector import (
    assert_drained,
    compare_pairs,
    validate_episode,
)


def _episode():
    lengths = (torch.arange(32).reshape(4, 8) % 6) + 1
    mask = torch.arange(6) < lengths.unsqueeze(-1)
    tokens = torch.full((4, 8, 6), 7, dtype=torch.long)
    reasons = []
    for row in range(4):
        reasons.append([])
        for response in range(8):
            length = int(lengths[row, response])
            if length < 6:
                tokens[row, response, length - 1] = 9
            reasons[row].append("stop" if length < 6 else "length")
    return RawRollout(
        prompts=torch.ones(4, 2, dtype=torch.long),
        prompt_mask=torch.ones(4, 2, dtype=torch.bool),
        responses=tokens,
        response_mask=mask,
        logprobs_old=torch.full((4, 8, 6), -2.0),
        finish_reasons=reasons,
        policy_version=0,
    )


def test_complete_variable_eos_groups_keep_all_zero_reward_episodes():
    raw = _episode()
    metrics = validate_episode(raw, 32, 8, 6, {9})
    assert metrics["responses"] == 32
    assert metrics["stop_responses"] + metrics["length_responses"] == 32
    assert metrics["min_response_tokens"] == 1
    assert metrics["max_response_tokens"] == 6
    compare_pairs(raw, deepcopy(raw))


@pytest.mark.parametrize(
    "damage", ["length_only", "hole", "lost_eos", "nonfinite", "missing_group"]
)
def test_official_qualification_cannot_pass_incomplete_or_synthetic_length_only(damage):
    raw = _episode()
    if damage == "length_only":
        raw.response_mask[:] = True
        raw.finish_reasons = [["length"] * 8 for _ in range(4)]
    elif damage == "hole":
        raw.response_mask[0, 4, 1] = False
    elif damage == "lost_eos":
        raw.responses[0, 0, 0] = 7
    elif damage == "nonfinite":
        raw.logprobs_old[0, 0, 0] = float("nan")
    else:
        raw.responses = raw.responses[:-1]
    with pytest.raises(ValueError):
        validate_episode(raw, 32, 8, 6, {9})


def test_collector_drain_check_tracks_current_engine_owners():
    from astrai.trainer.backend import run_batch
    from tests.support.inference import make_cpu_model, make_cpu_scheduler

    with torch.random.fork_rng(devices=[]):
        engine = make_cpu_scheduler(make_cpu_model())
        try:
            assert_drained(engine)
            assert len(run_batch(engine, [[10, 20]], max_tokens=2)) == 1
            assert_drained(engine)
            engine.add_request("pending", prompt_ids=[10, 20], max_tokens=2)
            engine.scheduler.admit_requests()
            with pytest.raises(RuntimeError, match="did not drain"):
                assert_drained(engine)
        finally:
            engine.stop()
        assert_drained(engine)
