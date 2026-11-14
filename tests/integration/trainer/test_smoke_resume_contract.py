"""The resume gate compares actor, optimizer and every learner RNG state."""

import json
import shutil

import numpy as np
import pytest
import safetensors.torch as st
import torch

from examples.rl_reward.smoke_resume import (
    assert_state_equal,
    compare_checkpoints,
    training_traces,
)


def _checkpoint(root):
    root.mkdir()
    st.save_file(
        {"weight": torch.arange(6).reshape(2, 3).float()}, root / "model.safetensors"
    )
    (root / "meta.json").write_text(
        json.dumps(
            {"optimizer_steps": 10, "policy_version": 10, "consumed_samples": 640}
        )
    )
    for name in ("optimizer", "scheduler", "reference_model", "rng_state"):
        torch.save({"tensor": torch.arange(4), "scalar": 10}, root / f"{name}.pt")
    torch.save(
        {
            "recipe": {"seed": 7013},
            "data_hashes": {"train": "fixture"},
            "verifier_sha256": "fixture",
            "elapsed_seconds": 100.0,
            "rng_by_rank": [
                {"torch": torch.arange(4), "numpy": np.array([i, i + 1])}
                for i in range(8)
            ],
        },
        root / "reward_runner.pt",
    )


@pytest.mark.parametrize(
    "damage", [None, "actor", "optimizer", "rng", "cursor", "missing_extra"]
)
def test_resume_requires_complete_actor_optimizer_rng_and_cursor_parity(
    tmp_path, damage
):
    original = tmp_path / "original"
    resumed = tmp_path / "resumed"
    _checkpoint(original)
    shutil.copytree(original, resumed)
    if damage == "actor":
        st.save_file({"weight": torch.zeros(2, 3)}, resumed / "model.safetensors")
    elif damage == "optimizer":
        torch.save({"tensor": torch.ones(4), "scalar": 10}, resumed / "optimizer.pt")
    elif damage == "rng":
        state = torch.load(resumed / "reward_runner.pt", weights_only=False)
        state["rng_by_rank"][7]["torch"][0] = 999
        torch.save(state, resumed / "reward_runner.pt")
    elif damage == "cursor":
        (resumed / "meta.json").write_text(
            json.dumps(
                {"optimizer_steps": 10, "policy_version": 10, "consumed_samples": 639}
            )
        )
    elif damage == "missing_extra":
        (resumed / "reference_model.pt").unlink()
    if damage is None:
        assert (
            compare_checkpoints(original, resumed, updates=10, consumed_samples=640)
            == 1
        )
    else:
        with pytest.raises((AssertionError, ValueError, FileNotFoundError)):
            compare_checkpoints(original, resumed, updates=10, consumed_samples=640)


def test_trace_gate_rejects_dropped_groups_and_changed_invalid_rewards(tmp_path):
    value = {
        "groups": ["7013:2:prompt"],
        "policy_version": 2,
        "prompts": torch.tensor([[1, 2]]),
        "prompt_mask": torch.tensor([[True, True]]),
        "tokens": torch.tensor([[[3, 4]]]),
        "response_mask": torch.tensor([[[True, True]]]),
        "policy_raw_logp": torch.tensor([[[-1.0, -2.0]]]),
        "rewards": torch.tensor([[0.0]]),
    }
    torch.save(value, tmp_path / "train.v2.rank0.fixture.pt")
    traces = training_traces(tmp_path, after_version=2)
    assert len(traces) == 1 and next(iter(traces.values()))["rewards"].item() == 0
    with pytest.raises(ValueError, match="missing"):
        training_traces(tmp_path, after_version=3)
    changed = {key: dict(row) for key, row in traces.items()}
    changed["7013:2:prompt"]["rewards"] = torch.tensor([1.0])
    with pytest.raises(AssertionError):
        assert_state_equal(traces, changed)
