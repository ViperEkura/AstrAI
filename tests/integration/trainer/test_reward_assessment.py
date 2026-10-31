"""Curve decisions cannot discard failed or unreached seeds to claim success."""

from copy import deepcopy

import pytest

from examples.rl_reward.assessment import (
    decide,
    independent_target,
    paired_quality_interval,
)


def _pairs():
    points = [
        {"policy_version": 0, "reward_mean": 0.1, "elapsed_seconds": 1.0},
        {"policy_version": 400, "reward_mean": 0.3, "elapsed_seconds": 100.0},
    ]
    pairs = []
    for seed in (3407, 3408, 3409):
        baseline = {
            "budget_completed": True,
            "exit_code": 0,
            "allocation_cost_verified": True,
            "end_to_end_timing_verified": True,
            "dev_points": deepcopy(points),
        }
        candidate = deepcopy(baseline)
        candidate["dev_points"][-1]["elapsed_seconds"] = 75.0
        pairs.append({"seed": seed, "baseline": baseline, "candidate": candidate})
    return pairs


def test_pilot_freeze_requires_quality_gain_and_nonzero_advantages():
    points = _pairs()[0]["baseline"]["dev_points"]
    assert independent_target(points, [0.1])["reward_target"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="learning signal"):
        independent_target(points, [0.0])


@pytest.mark.parametrize(
    "failure",
    [
        "missing_seed",
        "failed",
        "unreached",
        "unverified_cost",
        "quality_loss",
        "too_slow",
    ],
)
def test_failed_or_missing_pairs_never_pass_the_formal_gate(failure):
    pairs = _pairs()
    quality = {"paired_seeds": 3, "lower_95": 0.0}
    if failure == "missing_seed":
        with pytest.raises(ValueError, match="three"):
            decide(pairs[:-1], target=0.2, quality_interval=quality)
        return
    if failure == "failed":
        pairs[2]["candidate"]["exit_code"] = 1
    elif failure == "unreached":
        pairs[2]["candidate"]["dev_points"][-1]["reward_mean"] = 0.1
    elif failure == "unverified_cost":
        pairs[2]["baseline"]["allocation_cost_verified"] = False
    elif failure == "quality_loss":
        quality["lower_95"] = -0.03
    else:
        for pair in pairs:
            pair["candidate"]["dev_points"][-1]["elapsed_seconds"] = 90.0
    result = decide(pairs, target=0.2, quality_interval=quality)
    assert result["state"] == "FAIL_OR_INCOMPLETE"
    assert len(result["seed_pairs"]) == 3


def test_quality_interval_pairs_same_prompts_and_keeps_zero_outcomes():
    baseline = [{"a": 1, "b": 0, "c": 0}] * 3
    candidate = [{"a": 1, "b": 0, "c": 0}] * 3
    interval = paired_quality_interval(baseline, candidate, repetitions=1000)
    assert (
        interval["mean_difference"] == interval["lower_95"] == interval["upper_95"] == 0
    )
    with pytest.raises(ValueError, match="identical held-out"):
        paired_quality_interval(baseline, [{"a": 1}] * 3)


def test_initially_reached_target_and_nonfinite_quality_cannot_pass():
    pairs = _pairs()
    result = decide(
        pairs, target=0.05, quality_interval={"paired_seeds": 3, "lower_95": 0.0}
    )
    assert result["state"] == "FAIL_OR_INCOMPLETE"
    with pytest.raises(ValueError, match="finite"):
        decide(
            pairs,
            target=0.2,
            quality_interval={"paired_seeds": 3, "lower_95": float("nan")},
        )
