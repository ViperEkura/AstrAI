"""Unit tests for the online rollout module."""

import pytest
import torch

from astrai.trainer.rollout import (
    BaseRewardModel,
    RawRollout,
    RolloutResult,
)


def test_raw_rollout_fields():
    r = RawRollout(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        logprobs_old=torch.zeros(2, 3, 5),
    )
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.policy_version == 0
    assert r.prompt_texts == []
    assert r.response_texts == []


def test_rollout_result_inherits_raw_rollout_fields():
    r = RolloutResult(
        prompts=torch.zeros(2, 4, dtype=torch.long),
        prompt_mask=torch.ones(2, 4, dtype=torch.bool),
        responses=torch.zeros(2, 3, 5, dtype=torch.long),
        response_mask=torch.ones(2, 3, 5, dtype=torch.bool),
        logprobs_old=torch.zeros(2, 3, 5),
        rewards=torch.zeros(2, 3),
    )
    assert r.rewards.shape == (2, 3)
    assert r.prompts.shape == (2, 4)
    assert r.responses.shape == (2, 3, 5)
    assert r.prompt_mask.shape == (2, 4)
    # RolloutResult must carry every RawRollout field.
    raw_fields = {f for f in RawRollout.__dataclass_fields__}
    assert raw_fields.issubset(set(RolloutResult.__dataclass_fields__))


def test_base_reward_model_is_abstract():
    with pytest.raises(TypeError):
        BaseRewardModel()
