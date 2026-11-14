"""Assigned eight-H100 ten-update smoke and fresh-process resume parity."""

import argparse
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open

from examples.rl_reward.experiment import launch, require_run_identity, write_json
from examples.rl_reward.run import Recipe, resolve_optimizer


def assert_state_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=False)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError("resume state keys differ")
        for key in expected:
            assert_state_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError("resume state sequence differs")
        for left, right in zip(expected, actual):
            assert_state_equal(left, right)
    elif type(expected) is not type(actual) or expected != actual:
        raise ValueError("resume scalar state differs")


def compare_checkpoints(original, resumed, *, updates, consumed_samples):
    original, resumed = Path(original), Path(resumed)
    with safe_open(
        original / "model.safetensors", framework="pt", device="cpu"
    ) as left:
        with safe_open(
            resumed / "model.safetensors", framework="pt", device="cpu"
        ) as right:
            if set(left.keys()) != set(right.keys()):
                raise ValueError("resume actor tensor keys differ")
            tensors = len(left.keys())
            for key in left.keys():
                assert_state_equal(left.get_tensor(key), right.get_tensor(key))
    for root in (original, resumed):
        meta = json.loads((root / "meta.json").read_text())
        if (
            meta.get("optimizer_steps") != updates
            or meta.get("policy_version") != updates
            or meta.get("consumed_samples") != consumed_samples
        ):
            raise ValueError(
                "resume optimizer/version/data cursor differs from full smoke budget"
            )
    for name in ("optimizer", "scheduler", "reference_model", "rng_state"):
        left = torch.load(
            original / f"{name}.pt", map_location="cpu", weights_only=False
        )
        right = torch.load(
            resumed / f"{name}.pt", map_location="cpu", weights_only=False
        )
        assert_state_equal(left, right)
        del left, right
    left = torch.load(
        original / "reward_runner.pt", map_location="cpu", weights_only=False
    )
    right = torch.load(
        resumed / "reward_runner.pt", map_location="cpu", weights_only=False
    )
    for key in ("recipe", "data_hashes", "verifier_sha256", "rng_by_rank"):
        assert_state_equal(left[key], right[key])
    if len(left["rng_by_rank"]) != 8:
        raise ValueError("resume must retain all eight learner RNG snapshots")
    return tensors


def training_traces(root, *, after_version):
    traces = {}
    for path in Path(root).glob("train.*.pt"):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["policy_version"] < after_version:
            continue
        for row, group in enumerate(value["groups"]):
            if group in traces:
                raise ValueError("duplicate training group in one smoke attempt")
            traces[group] = {
                key: value[key][row]
                for key in (
                    "prompts",
                    "prompt_mask",
                    "tokens",
                    "response_mask",
                    "policy_raw_logp",
                    "rewards",
                )
            }
    if not traces:
        raise ValueError("smoke training token traces are missing")
    return traces


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    if (
        settings["allocation_owner_authorized"] is not True
        or settings["allocated_h100_count"] != 8
    ):
        raise ValueError("smoke requires an actual assigned eight-H100 resource")
    template = Recipe(**yaml.safe_load(Path(settings["template_recipe"]).read_text()))
    if template.model_repo != "Qwen/Qwen3-1.7B" or not template.request_seeded_sampling:
        raise ValueError(
            "smoke requires the official formal model and request-local sampling"
        )
    if (
        template.model_revision != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
        or template.device_type != "cuda"
        or template.dtype != "bfloat16"
    ):
        raise ValueError(
            "smoke checkpoint, dtype and device must match formal qualification"
        )
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("smoke requires an assigned H100 test resource")
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual_head != settings["source_head"]:
        raise ValueError("smoke checkout does not match the frozen source")
    root = Path(settings["output_dir"])
    if not root.is_absolute() or root.exists():
        raise ValueError("smoke output must be a new absolute private directory")
    root.mkdir(parents=True, mode=0o700)
    recipe = replace(
        template,
        updates=10,
        checkpoint_interval=2,
        test_file=None,
        eval_interval=5,
        seed=7013,
        batch_per_device=8,
        group_size=8,
        learner_microbatch_prompts=1,
        overlap_collection=True,
        output_dir=str(root / "run"),
    )
    recipe.validate()
    resolve_optimizer(recipe)
    for attempt in ("uninterrupted", "resumed"):
        resume = (
            None
            if attempt == "uninterrupted"
            else root / "run/checkpoints/epoch_0_step_2"
        )
        receipt = launch(settings, recipe, attempt, resume=resume)
        if not receipt["budget_completed"]:
            write_json(
                root / "smoke-resume.json",
                {"state": "FAIL_OR_INCOMPLETE", "attempt": attempt},
            )
            raise RuntimeError("eight-H100 smoke attempt did not complete")
        require_run_identity(recipe.output_dir, recipe, settings["source_head"])
        checkpoint = root / "run/checkpoints/epoch_0_step_10"
        if attempt == "uninterrupted":
            checkpoint.rename(root / "original-final-checkpoint")
            (root / "run/token_traces").rename(root / "original-token-traces")
    original = training_traces(root / "original-token-traces", after_version=2)
    resumed = training_traces(root / "run/token_traces", after_version=2)
    assert_state_equal(original, resumed)
    if len(original) != 8 * 64:
        raise ValueError("smoke did not retain every post-resume global prompt group")
    tensors = compare_checkpoints(
        root / "original-final-checkpoint", checkpoint, updates=10, consumed_samples=640
    )
    result = {
        "state": "PASS",
        "gate": "smoke_resume_8_h100",
        "source_head": settings["source_head"],
        "model_repo": recipe.model_repo,
        "model_revision": recipe.model_revision,
        "h100_count": 8,
        "updates": 10,
        "resume_step": 2,
        "compared_prompt_groups": len(original),
        "compared_actor_tensors": tensors,
        "actor_optimizer_reference_rng_and_token_parity": True,
        "skipped": 0,
        "full_reward_curves": "NOT_TESTED_BY_THIS_CHECK",
    }
    write_json(root / "smoke-resume.json", result)
    print(
        json.dumps({"state": result["state"], "gate": result["gate"], "h100_count": 8}),
        flush=True,
    )


if __name__ == "__main__":
    main()
