"""Frozen-target, paired-seed quality and time-to-reward decisions."""

import json
import math
from pathlib import Path

import numpy as np


def read_rows(root, kind):
    return [
        json.loads(line)
        for path in sorted(Path(root).glob(f"{kind}.*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def scalar_points(rows, split):
    points = {}
    for row in rows:
        if row["split"] != split:
            continue
        step = row["policy_version"]
        reward = row["reward_mean"]
        elapsed = row["elapsed_seconds"]
        if (
            not math.isfinite(reward)
            or not 0 <= reward <= 1
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("invalid evaluation reward or elapsed time")
        previous = points.get(step)
        if previous is not None and not math.isclose(
            previous["reward_mean"], reward, abs_tol=1e-7
        ):
            raise ValueError("repeated evaluation changed at the same policy version")
        if previous is None or elapsed < previous["elapsed_seconds"]:
            points[step] = row
    return [points[step] for step in sorted(points)]


def first_crossing(points, target):
    return next(
        (row["elapsed_seconds"] for row in points if row["reward_mean"] >= target), None
    )


def independent_target(points, advantage_fractions, *, minimum_gain=0.02):
    """Choose a target from the independent control pilot, before any pair."""
    if not points or points[0]["policy_version"] != 0:
        raise ValueError("pilot must include the initial dev evaluation")
    initial = points[0]["reward_mean"]
    best = max(row["reward_mean"] for row in points)
    if (
        best - initial < minimum_gain
        or not advantage_fractions
        or max(advantage_fractions) <= 0
    ):
        raise ValueError("pilot has not established a learning signal")
    target = initial + (best - initial) / 2
    if target <= initial:
        raise ValueError("target must exceed initial policy quality")
    return {
        "reward_target": target,
        "initial_dev_reward": initial,
        "best_pilot_dev_reward": best,
        "rule": "initial + half of independent pilot dev gain",
        "minimum_pilot_gain": minimum_gain,
    }


def final_prompt_scores(rows, final_step):
    scores = {}
    for row in rows:
        if row["phase"] != "test" or row["policy_version"] != final_step:
            continue
        for group in row["groups"]:
            responses = group["responses"]
            if len(responses) != 1:
                raise ValueError(
                    "formal held-out evaluation must be greedy group size 1"
                )
            key, reward = group["prompt_id"], responses[0]["reward"]
            if not math.isfinite(reward) or reward not in (0, 1):
                raise ValueError(
                    "held-out task reward must retain every binary outcome"
                )
            if key in scores and scores[key] != reward:
                raise ValueError("duplicate held-out evaluation changed its reward")
            scores[key] = reward
    if not scores:
        raise ValueError("final held-out outcomes are missing")
    return scores


def paired_quality_interval(baselines, candidates, *, repetitions=10000, seed=731):
    """Resample matched seeds and matched held-out prompts jointly."""
    if len(baselines) != 3 or len(candidates) != 3:
        raise ValueError("quality requires all three matched seed outcomes")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    ids = sorted(baselines[0])
    if not ids or any(set(scores) != set(ids) for scores in baselines + candidates):
        raise ValueError("all arms and seeds must retain identical held-out prompts")
    if any(
        not math.isfinite(v) or v not in (0, 1)
        for scores in baselines + candidates
        for v in scores.values()
    ):
        raise ValueError("every held-out binary outcome must be finite")
    differences = np.array(
        [
            [candidate[key] - baseline[key] for key in ids]
            for baseline, candidate in zip(baselines, candidates)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = []
    for begin in range(0, repetitions, 100):
        size = min(100, repetitions - begin)
        seeds = rng.integers(0, 3, (size, 3, 1))
        prompts = rng.integers(0, len(ids), (size, 1, len(ids)))
        estimates.extend(differences[seeds, prompts].mean(axis=(1, 2)).tolist())
    lower, upper = np.quantile(estimates, [0.025, 0.975]).tolist()
    return {
        "mean_difference": differences.mean().item(),
        "lower_95": lower,
        "upper_95": upper,
        "heldout_prompts": len(ids),
        "paired_seeds": 3,
        "method": "hierarchical paired seed/prompt percentile bootstrap",
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def decide(pairs, *, target, quality_interval, margin=0.02, time_ratio_limit=0.8):
    """Missing, failed and unreached seeds remain failures of the full gate."""
    seeds = [pair["seed"] for pair in pairs]
    if not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError("reward target must be finite and in (0, 1]")
    if not math.isfinite(quality_interval["lower_95"]):
        raise ValueError("quality confidence bound must be finite")
    if len(pairs) != 3 or len(set(seeds)) != 3:
        raise ValueError(
            "decision requires exactly three unique predeclared seed pairs"
        )
    reasons = []
    times = {"baseline": [], "candidate": []}
    for pair in pairs:
        for arm in times:
            run = pair[arm]
            points = run["dev_points"]
            if not points or points[0]["policy_version"] != 0:
                reasons.append(f"{arm}:seed={pair['seed']}:initial_dev_missing")
            elif points[0]["reward_mean"] >= target:
                reasons.append(
                    f"{arm}:seed={pair['seed']}:target_already_met_at_initialization"
                )
            if not run["budget_completed"] or run["exit_code"] != 0:
                reasons.append(f"{arm}:seed={pair['seed']}:failed_or_incomplete")
            crossing = first_crossing(run["dev_points"], target)
            if crossing is None:
                reasons.append(f"{arm}:seed={pair['seed']}:target_unreached")
            else:
                times[arm].append(crossing)
            if not run.get("allocation_cost_verified", False):
                reasons.append(
                    f"{arm}:seed={pair['seed']}:allocation_accounting_unverified"
                )
    ratio = None
    if len(times["baseline"]) == len(times["candidate"]) == 3:
        baseline = float(np.mean(times["baseline"]))
        candidate = float(np.mean(times["candidate"]))
        if baseline <= 0:
            raise ValueError("reward target must not be crossed at initialization")
        ratio = candidate / baseline
        if ratio > time_ratio_limit:
            reasons.append("time_to_reward_ratio_above_frozen_limit")
    for pair in pairs:
        for arm in ("baseline", "candidate"):
            if not pair[arm].get("end_to_end_timing_verified", False):
                reasons.append(f"{arm}:seed={pair['seed']}:launcher_timing_unverified")
    if quality_interval["paired_seeds"] != 3 or quality_interval["lower_95"] < -margin:
        reasons.append("heldout_quality_noninferiority_not_proven")
    return {
        "state": "PASS_FORMAL_PAIRED_REWARD_GATE"
        if not reasons
        else "FAIL_OR_INCOMPLETE",
        "reasons": reasons,
        "seed_pairs": pairs,
        "reward_target": target,
        "time_to_reward_ratio": ratio,
        "time_ratio_limit": time_ratio_limit,
        "quality_interval": quality_interval,
        "noninferiority_margin": margin,
    }
