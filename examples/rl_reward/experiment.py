"""Assigned-resource pilot and complete paired reward matrix controller.

Launch arguments and allocation receipts come from the test-resource owner.
Raw launch commands, process identifiers, stderr and checkpoints stay private.
"""

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from examples.rl_reward.assessment import (
    decide,
    final_prompt_scores,
    independent_target,
    paired_quality_interval,
    read_rows,
    scalar_points,
)
from examples.rl_reward.data import sha256_file
from examples.rl_reward.publication import public_recipe
from examples.rl_reward.run import Recipe, resolve_optimizer

SEEDS = (3407, 3408, 3409)
PILOT_SEED = 9413


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def manifest(root):
    return json.loads((Path(root) / "run_manifest.rank0.json").read_text())


def require_run_identity(root, recipe, source_head):
    actual = manifest(root)
    if actual["git_head"] != source_head or actual["recipe"] != public_recipe(recipe):
        raise ValueError("run source or frozen recipe differs from the controller")
    runtimes = read_rows(root, "runtime")
    if (
        actual["learner"]["dp_size"] != 8
        or not runtimes
        or any(row["h100_count"] != 8 for row in runtimes)
    ):
        raise ValueError("formal run did not execute the declared eight-H100 workload")
    return actual


def completed(root, updates):
    rows = read_rows(root, "run_end")
    return bool(rows) and all(
        row["budget_completed"] and row["optimizer_step"] == updates for row in rows
    )


def launch(settings, recipe, case, *, resume=None):
    root = Path(settings["output_dir"])
    private = root / "private"
    private.mkdir(exist_ok=True, mode=0o700)
    config_file = private / f"{case}.yaml"
    with config_file.open("x") as stream:
        yaml.safe_dump(asdict(recipe), stream)
    runner = Path(__file__).with_name("run.py")
    substitutions = {"{script}": str(runner), "{recipe}": str(config_file)}
    template = settings["launcher_argv"]
    if not isinstance(template, list) or not all(
        isinstance(item, str) and item for item in template
    ):
        raise ValueError("launch owner must provide an argument array")
    if not all(key in template for key in substitutions):
        raise ValueError("launcher must contain {script} and {recipe} arguments")
    argv = [substitutions.get(item, item) for item in template]
    if resume is not None:
        argv.extend(["--resume", str(resume)])
    started = time.perf_counter()
    startup_seconds = None
    with (private / f"{case}.log").open("x") as log:
        process = subprocess.Popen(
            argv,
            cwd=Path(__file__).resolve().parents[2],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "ASTRAI_RECORD_CONTROLLER_TIMING": "1",
                "PYTHONUNBUFFERED": "1",
            },
        )
        write_json(
            private / f"{case}.process.json",
            {"pid": process.pid, "argv": argv, "case": case},
        )
        for line in process.stdout:
            log.write(line)
            if line.strip() == "ASTRAI_RUNNER_MAIN_STARTED" and startup_seconds is None:
                startup_seconds = time.perf_counter() - started
        exit_code = process.wait()
    result = {
        "case": case,
        "exit_code": exit_code,
        "controller_seconds": time.perf_counter() - started,
        "launcher_startup_seconds": startup_seconds,
        "budget_completed": exit_code == 0
        and completed(recipe.output_dir, recipe.updates),
    }
    write_json(root / f"{case}.receipt.json", result)
    print(
        json.dumps(
            {
                "case": case,
                "state": "BUDGET_COMPLETED"
                if result["budget_completed"]
                else "FAILED_OR_INCOMPLETE",
            }
        ),
        flush=True,
    )
    return result


def verify_gpu_gates(settings, recipe):
    required = {
        "official_loader",
        "native_cuda_protocol",
        "official_collector",
        "smoke_resume_8_h100",
    }
    receipts = settings["gpu_gate_receipts"]
    if set(receipts) != required:
        raise ValueError("formal learning requires the complete predeclared GPU gates")
    for gate, path in receipts.items():
        record = json.loads(Path(path).read_text())
        if (
            record.get("state") != "PASS"
            or record.get("source_head") != settings["source_head"]
        ):
            raise ValueError(f"GPU gate incomplete or from a different source: {gate}")
        if record.get("skipped", 0) != 0:
            raise ValueError("skipped GPU cases do not establish qualification")
        if gate in {"official_loader", "official_collector"} and (
            record.get("model_repo") != recipe.model_repo
            or record.get("model_revision") != recipe.model_revision
        ):
            raise ValueError("official-model gates must cover the formal checkpoint")


def validate_template(settings):
    recipe = Recipe(**yaml.safe_load(Path(settings["template_recipe"]).read_text()))
    recipe.validate()
    resolve_optimizer(recipe)
    if (
        recipe.device_type != "cuda"
        or recipe.group_size != 8
        or not recipe.request_seeded_sampling
    ):
        raise ValueError(
            "formal matrix requires CUDA, complete groups of eight and request-local sampling"
        )
    if (
        recipe.model_repo != "Qwen/Qwen3-1.7B"
        or recipe.model_revision != "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    ):
        raise ValueError(
            "formal curves require the frozen preferred official checkpoint"
        )
    if (
        recipe.updates != {"countdown": 400, "gsm8k": 1000}[recipe.task]
        or recipe.test_file is None
    ):
        raise ValueError(
            "formal matrix must retain the full task budget and held-out test"
        )
    if settings["allocated_h100_count"] != 8 or recipe.batch_per_device != 8:
        raise ValueError("formal workload requires eight H100 and global 64 prompts")
    if settings["allocation_owner_authorized"] is not True:
        raise ValueError("actual assigned-resource authorization is required")
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual_head != settings["source_head"]:
        raise ValueError("assigned checkout is not the frozen source")
    verify_gpu_gates(settings, recipe)
    return recipe


def pilot(settings, template):
    root = Path(settings["output_dir"])
    pilot_recipe = replace(
        template,
        seed=PILOT_SEED,
        updates=25,
        test_file=None,
        eval_interval=5,
        checkpoint_interval=25,
        overlap_collection=False,
        learner_microbatch_prompts=settings["baseline_microbatch_prompts"],
        reward_target=None,
        output_dir=str(root / "pilot"),
    )
    receipt = launch(settings, pilot_recipe, "pilot")
    if not receipt["budget_completed"]:
        raise RuntimeError("independent pilot did not complete its budget")
    actual = require_run_identity(
        pilot_recipe.output_dir, pilot_recipe, settings["source_head"]
    )
    points = scalar_points(read_rows(pilot_recipe.output_dir, "eval_metrics"), "dev")
    advantages = [
        row["nonzero_advantage_group_fraction"]
        for row in read_rows(pilot_recipe.output_dir, "round_metrics")
    ]
    target = independent_target(points, advantages)
    frozen = {
        "schema_version": 1,
        "source_head": settings["source_head"],
        "task": template.task,
        "model_repo": template.model_repo,
        "model_revision": template.model_revision,
        "dataset_sha256": {
            **actual["dataset_sha256"],
            "test": sha256_file(template.test_file),
        },
        "model_files": actual["model_files"],
        "pilot_seed": PILOT_SEED,
        "pilot_manifest_sha256": sha256_file(
            Path(pilot_recipe.output_dir) / "run_manifest.rank0.json"
        ),
        "paired_seeds": list(SEEDS),
        "full_updates": template.updates,
        "quality_margin": 0.02,
        "time_ratio_limit": 0.8,
        "frozen_unix": time.time(),
        **target,
    }
    write_json(root / "frozen-protocol.json", frozen)
    print(
        json.dumps(
            {
                "state": "INDEPENDENT_PILOT_PROTOCOL_FROZEN",
                "task": template.task,
                "reward_target": frozen["reward_target"],
            }
        ),
        flush=True,
    )


def paired(settings, template, *, launch_runs=True):
    root = Path(settings["output_dir"])
    protocol_path = root / "frozen-protocol.json"
    frozen = json.loads(protocol_path.read_text())
    if (
        frozen["source_head"] != settings["source_head"]
        or frozen["paired_seeds"] != list(SEEDS)
        or frozen["task"] != template.task
    ):
        raise ValueError("frozen pilot protocol does not match the complete matrix")
    results, scores = [], {"baseline": [], "candidate": []}
    for index, seed in enumerate(SEEDS):
        pair = {"seed": seed}
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        for arm in order:
            case = f"{arm}.seed{seed}"
            recipe = replace(
                template,
                seed=seed,
                reward_target=frozen["reward_target"],
                output_dir=str(root / case),
                overlap_collection=arm == "candidate",
                learner_microbatch_prompts=settings[f"{arm}_microbatch_prompts"],
            )
            receipt = (
                launch(settings, recipe, case)
                if launch_runs
                else json.loads((root / f"{case}.receipt.json").read_text())
            )
            if not (Path(recipe.output_dir) / "run_manifest.rank0.json").is_file():
                pair[arm] = {
                    **receipt,
                    "budget_completed": False,
                    "dev_points": [],
                    "allocation_cost_verified": False,
                    "failure": "run_identity_missing",
                }
                continue
            actual = require_run_identity(
                recipe.output_dir, recipe, settings["source_head"]
            )
            if (
                actual["created_unix"] <= frozen["frozen_unix"]
                or actual["model_files"] != frozen["model_files"]
            ):
                raise ValueError(
                    "pair started before protocol freeze or changed its checkpoint"
                )
            if any(
                actual["dataset_sha256"][key] != value
                for key, value in frozen["dataset_sha256"].items()
            ):
                raise ValueError("pair changed pilot train/dev identities")
            metrics = scalar_points(read_rows(recipe.output_dir, "eval_metrics"), "dev")
            startup = receipt["launcher_startup_seconds"]
            if startup is not None:
                metrics = [
                    {
                        **point,
                        "runner_elapsed_seconds": point["elapsed_seconds"],
                        "elapsed_seconds": startup + point["elapsed_seconds"],
                    }
                    for point in metrics
                ]
            cost_path = root / "allocation_receipts" / f"{case}.json"
            verified_cost = False
            if cost_path.is_file():
                cost = json.loads(cost_path.read_text())
                verified_cost = (
                    cost.get("state") == "VERIFIED_BY_LAUNCH_OWNER"
                    and cost.get("allocated_h100_count") == 8
                    and cost.get("case") == case
                    and cost.get("source_head") == settings["source_head"]
                    and math.isfinite(cost.get("allocation_seconds", 0))
                    and cost.get("allocation_seconds", 0)
                    >= receipt["controller_seconds"]
                )
            pair[arm] = {
                **receipt,
                "dev_points": metrics,
                "allocation_cost_verified": verified_cost,
                "end_to_end_timing_verified": startup is not None,
            }
            if verified_cost:
                pair[arm]["allocated_h100_hours"] = (
                    cost["allocation_seconds"] * 8 / 3600
                )
            if receipt["budget_completed"]:
                scores[arm].append(
                    final_prompt_scores(
                        read_rows(recipe.output_dir, "eval_results"), recipe.updates
                    )
                )
        results.append(pair)
        write_json(
            root
            / (
                f"pair.seed{seed}.json"
                if launch_runs
                else f"pair.seed{seed}.assessment{time.time_ns()}.json"
            ),
            pair,
        )
    decision_path = root / (
        "decision.json" if launch_runs else f"decision.assessment{time.time_ns()}.json"
    )
    if any(len(values) != 3 for values in scores.values()):
        write_json(
            decision_path,
            {
                "state": "FAIL_OR_INCOMPLETE",
                "seed_pairs": results,
                "reason": "one or more seeds failed; every attempted pair retained",
            },
        )
        return
    quality = paired_quality_interval(scores["baseline"], scores["candidate"])
    decision = decide(results, target=frozen["reward_target"], quality_interval=quality)
    decision["frozen_protocol_sha256"] = sha256_file(protocol_path)
    write_json(decision_path, decision)
    print(
        json.dumps(
            {
                "state": decision["state"],
                "task": template.task,
                "seed_pairs": len(results),
            }
        ),
        flush=True,
    )


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pilot", "paired", "assess"))
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    root = Path(settings["output_dir"])
    if not root.is_absolute():
        raise ValueError("experiment output must be an absolute private path")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    template = validate_template(settings)
    if args.phase == "pilot":
        pilot(settings, template)
    else:
        paired(settings, template, launch_runs=args.phase == "paired")


if __name__ == "__main__":
    main()
