"""Verifier contracts and a fresh-process native reward-runner resume."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import safetensors.torch as st
import torch
import yaml

from astrai.config import TrainConfig
from astrai.model import AutoRegressiveLM
from astrai.trainer.optional_extras import checkpoint_extras
from examples.rl_reward.data import load_splits
from examples.rl_reward.publication import (
    dependency_versions,
    public_pretrained,
    public_recipe,
)
from examples.rl_reward.rewards import (
    TaskReward,
    countdown_score,
    gsm8k_score,
)
from examples.rl_reward.run import Recipe, build_training, restore_runner_rng
from tests.support.models import make_tiny_config
from tests.support.tokenizers import CHAT_TEMPLATE, build_test_tokenizer


@pytest.mark.parametrize(
    "expression,numbers,target,correct,valid",
    [
        ("(7+3)/2", [7, 3, 2], 5, 1, True),
        ("(7+3)/2", [7, 3, 2], 4, 0, True),
        ("2+2+3", [2, 2, 3], 7, 1, True),
        ("2+3", [2, 2, 3], 5, 0, False),
        ("-7+3", [7, 3], -4, 1, True),
        ("(7+3)/(2-2)", [7, 3, 2, 2], 5, 0, False),
        ("2**3", [2, 3], 8, 0, False),
        ("7//3", [7, 3], 2, 0, False),
        ("7.0+3", [7, 3], 10, 0, False),
        ("True+3", [1, 3], 4, 0, False),
    ],
)
def test_countdown_arithmetic_contract(expression, numbers, target, correct, valid):
    result = countdown_score(f"<answer>{expression}</answer>", numbers, target)
    assert (result.accuracy, result.valid_format) == (correct, valid)


def test_countdown_never_executes_generated_calls(tmp_path):
    sentinel = tmp_path / "generated-code-ran"
    response = f"<answer>__import__('pathlib').Path({str(sentinel)!r}).touch()</answer>"
    assert countdown_score(response, [1, 2], 3).accuracy == 0
    assert not sentinel.exists()
    assert (
        countdown_score(
            "<answer>" + "(" * 1100 + "1" + ")" * 1100 + "</answer>", [1, 2], 3
        ).accuracy
        == 0
    )


@pytest.mark.parametrize(
    "response,answer,score",
    [
        ("working\n#### 1,234", "reasoning\n#### 1234", 1),
        (r"\boxed{0.5}", "#### 1/2", 1),
        ("#### 3\n#### 7", "#### 7", 1),
        ("the answer is 7", "#### 7", 0),
        ("#### 6", "#### 7", 0),
    ],
)
def test_gsm8k_numeric_contract(response, answer, score):
    assert gsm8k_score(response, answer).accuracy == score


def test_reward_keeps_every_failed_response_and_rejects_incomplete_groups():
    reward = TaskReward({"prompt": {"numbers": [7, 3, 2], "target": 5}}, "countdown")
    scores = reward.score(
        ["prompt"], [["<answer>(7+3)/2</answer>", "invalid", "<answer>7+3</answer>"]]
    )
    torch.testing.assert_close(scores, torch.tensor([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="complete"):
        reward.score(["prompt", "prompt"], [["invalid"], []])
    with pytest.raises(ValueError, match="absent"):
        reward.score(["unknown"], [["invalid"]])


def test_resume_selects_each_learner_rng_and_rejects_topology_changes():
    states, expected = [], []
    for seed in (23, 71):
        torch.manual_seed(seed)
        states.append(checkpoint_extras()["rng_state"])
        expected.append(torch.rand(5))
    for rank in (0, 1):
        restore_runner_rng(states, rank, 2)
        torch.testing.assert_close(torch.rand(5), expected[rank], rtol=0, atol=0)
    with pytest.raises(ValueError, match="same topology"):
        restore_runner_rng(states, 0, 3)


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_split_identity_rejects_reordered_duplicates_and_bad_labels(tmp_path):
    tokenizer = build_test_tokenizer(chat_template=CHAT_TEMPLATE)
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    _write_jsonl(train, [{"id": "t", "numbers": [7, 3, 2], "target": 5}])
    _write_jsonl(dev, [{"id": "d", "numbers": [2, 7, 3], "target": 5}])
    with pytest.raises(ValueError, match="duplicate task"):
        load_splits({"train": train, "dev": dev}, "countdown", tokenizer, 400)
    _write_jsonl(dev, [{"id": "d", "question": "question", "answer": "no label"}])
    with pytest.raises(ValueError, match="invalid GSM8K answer"):
        load_splits({"dev": dev}, "gsm8k", tokenizer, 400)


def _prepare_run(tmp_path, updates=3):
    model_path = tmp_path / "model"
    model_path.mkdir()
    tokenizer = build_test_tokenizer(chat_template=CHAT_TEMPLATE)
    config = make_tiny_config(
        vocab_size=tokenizer._tokenizer.get_vocab_size(), max_position_embeddings=512
    )
    torch.manual_seed(73)
    AutoRegressiveLM(config).save_pretrained(model_path)
    tokenizer.save_pretrained(str(model_path))
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    _write_jsonl(
        train,
        [
            {"id": "t1", "numbers": [7, 3, 2], "target": 5},
            {"id": "t2", "numbers": [8, 3, 2], "target": 5},
        ],
    )
    _write_jsonl(dev, [{"id": "d1", "numbers": [9, 3, 2], "target": 6}])
    recipe = Recipe(
        model_path=str(model_path),
        model_repo="test/native",
        model_revision="fixture-v1",
        train_file=str(train),
        dev_file=str(dev),
        dataset_repo="test/countdown",
        dataset_revision="fixture-v1",
        output_dir=str(tmp_path / "results"),
        optimizer="adamw",
        optimizer_kwargs={"lr": 0.001, "weight_decay": 0.01},
        updates=updates,
        group_size=2,
        prompt_cap=400,
        response_cap=4,
        eval_interval=1,
        eval_batch_size=8,
        checkpoint_interval=1,
        dtype="float32",
        device_type="cpu",
    )
    return recipe, tokenizer


@pytest.mark.parametrize(
    "option,value,field,required",
    [
        ("learner_microbatch_prompts", 1, "rl_microbatch_prompts", "A2"),
        ("overlap_collection", True, "rollout_enable_overlap", "R1"),
        ("request_seeded_sampling", True, "rollout_seed", "R1"),
    ],
)
def test_recipe_feature_prerequisites(tmp_path, option, value, field, required):
    recipe, tokenizer = _prepare_run(tmp_path)
    setattr(recipe, option, value)
    splits, prompts = load_splits(
        {"train": recipe.train_file, "dev": recipe.dev_file},
        recipe.task,
        tokenizer,
        recipe.prompt_cap,
    )
    if field in TrainConfig.__dataclass_fields__:
        trainer = build_training(recipe, splits, prompts, {}, time.perf_counter())
        assert getattr(trainer.train_config, field) == (
            recipe.seed if field == "rollout_seed" else value
        )
    else:
        with pytest.raises(RuntimeError, match=required):
            build_training(recipe, splits, prompts, {}, time.perf_counter())


def test_torchrun_world_size_and_tail_validation(tmp_path, monkeypatch):
    recipe, tokenizer = _prepare_run(tmp_path)
    splits, prompts = load_splits(
        {"train": recipe.train_file, "dev": recipe.dev_file},
        recipe.task,
        tokenizer,
        recipe.prompt_cap,
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    trainer = build_training(recipe, splits, prompts, {}, time.perf_counter())
    assert trainer.train_config.dp_size == 2
    assert trainer.train_config.nprocs == 2
    assert trainer.train_config.dp_mode == "ddp"
    monkeypatch.setenv("WORLD_SIZE", "3")
    with pytest.raises(ValueError, match="complete global batches"):
        build_training(recipe, splits, prompts, {}, time.perf_counter())


@pytest.mark.slow
@pytest.mark.parametrize("features", [False, True])
def test_reward_example_fresh_process_resume_matches_continuation(tmp_path, features):
    if features and not {
        "rl_microbatch_prompts",
        "rollout_enable_overlap",
        "rollout_seed",
    }.issubset(TrainConfig.__dataclass_fields__):
        pytest.skip("combined microbatch/overlap resume requires A2 and R1")
    recipe, _ = _prepare_run(tmp_path)
    if features:
        recipe.batch_per_device = 2
        recipe.learner_microbatch_prompts = 1
        recipe.overlap_collection = True
        recipe.request_seeded_sampling = True
    recipe_file = tmp_path / "recipe.yaml"
    recipe_file.write_text(yaml.safe_dump(recipe.__dict__))
    repo = Path(__file__).resolve().parents[3]
    runner = repo / "examples" / "rl_reward" / "run.py"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        env.pop(name, None)

    def run(resume=None):
        args = [sys.executable, str(runner), "--config", str(recipe_file)]
        if resume:
            args += ["--resume", str(resume)]
        result = subprocess.run(
            args, cwd=repo, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stdout + result.stderr

    run()
    root = Path(recipe.output_dir)
    final_checkpoint = "epoch_2_step_3" if features else "epoch_1_step_3"
    original = st.load_file(
        str(root / "checkpoints" / final_checkpoint / "model.safetensors")
    )
    run(root / "checkpoints" / "epoch_0_step_1")
    resumed = st.load_file(
        str(root / "checkpoints" / final_checkpoint / "model.safetensors")
    )
    assert original.keys() == resumed.keys()
    for key in original:
        torch.testing.assert_close(original[key], resumed[key], rtol=1e-5, atol=1e-6)
    rows = [
        json.loads(line)
        for path in root.glob("round_metrics.*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert {row["policy_version"] for row in rows} == {1, 2, 3}
    assert all(row["optimizer_step"] == row["policy_version"] for row in rows)
    assert all(row["groups"] == recipe.batch_per_device for row in rows)
    assert len(list(root.glob("resume_manifest.*.json"))) == 1
    assert list((root / "token_traces").glob("*.pt"))
    manifest = json.loads((root / "run_manifest.rank0.json").read_text())
    assert manifest["evaluation"]["effective_batch_size"] == 2 * recipe.batch_per_device
    assert "betas" in manifest["recipe"]["optimizer_kwargs"]
    for key in ("model_path", "train_file", "dev_file", "output_dir"):
        assert manifest["recipe"][key] == "<private>"
    assert "git_diff" not in manifest and len(manifest["git_diff_sha256"]) == 64
    assert "resume_checkpoint" not in manifest
    for path in root.glob("runtime.*.jsonl"):
        for line in path.read_text().splitlines():
            runtime = json.loads(line)
            assert not {
                "hostname",
                "slurm_job_id",
                "slurm_nodes",
                "gpu_name",
                "gpu_memory_bytes",
            }.intersection(runtime)
            assert runtime["h100_count"] == 0
    if features:
        assert manifest["collector"]["enable_overlap"] is True
        assert manifest["sampling"]["request_seed"] == recipe.seed
        assert manifest["learner"]["rl_microbatch_prompts"] == 1
        meta = json.loads(
            (root / "checkpoints" / final_checkpoint / "meta.json").read_text()
        )
        assert meta["optimizer_steps"] == meta["policy_version"] == 3
        assert meta["consumed_samples"] == 6


def test_public_records_drop_environment_identity_and_unknown_provenance(tmp_path):
    recipe, _ = _prepare_run(tmp_path)
    recipe.model_path = "/private-infrastructure-marker/staged/model"
    recipe.output_dir = "/private-infrastructure-marker/results"
    record = public_recipe(recipe)
    assert "private-infrastructure-marker" not in json.dumps(record)
    assert record["model_revision"] == recipe.model_revision
    assert record["seed"] == recipe.seed
    provenance = public_pretrained(
        {
            "source_path": "/private-infrastructure-marker",
            "model_source": "private-infrastructure-marker",
            "unrecognized_future_field": "private-infrastructure-marker",
            "mapping_sha256": "a" * 64,
            "loaded_tensor_count": 17,
        }
    )
    assert provenance == {"mapping_sha256": "a" * 64, "loaded_tensor_count": 17}
    assert all(
        "/" not in value and " @ " not in value
        for value in dependency_versions().values()
    )
