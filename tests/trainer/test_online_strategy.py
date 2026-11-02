import pytest
import torch

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.trainer.strategy import (
    DPOStrategy,
    GRPOStrategy,
)
from tests.support.models import make_frozen, make_model, make_rollout_config
from tests.support.rollout import FakeExecutor, make_rollout_result


def _make_grpo(device, executor=None):
    model, _ = make_model(device)
    ref_model = make_frozen(model, device)
    return GRPOStrategy(
        model=model,
        device=device,
        old_model=None,
        ref_model=ref_model,
        clip_eps=0.2,
        kl_coef=0.01,
        group_size=4,
        model_fn=lambda c=make_rollout_config(): AutoRegressiveLM(c).to(device=device),
        executor=executor or FakeExecutor(),
    )


def _make_dpo(device, executor=None):
    model, _ = make_model(device)
    ref_model = make_frozen(model, device)
    return DPOStrategy(
        model=model,
        device=device,
        ref_model=ref_model,
        beta=0.1,
        reduction="sum",
        model_fn=lambda c=make_rollout_config(): AutoRegressiveLM(c).to(device=device),
        executor=executor or FakeExecutor(),
    )


def test_dpo_prepare_from_rollout_conditions_responses_on_prompt(cpu_device):
    strat = _make_dpo(cpu_device)
    r = make_rollout_result(B=3, G=4, P=6, R=5, device=cpu_device)
    r.responses[2].copy_(r.responses[0])
    r.rewards[2].copy_(r.rewards[0])
    r.prompts[[0, 2], -1] = torch.tensor([101, 102], device=cpu_device)
    r.prompt_mask[0, :2] = False
    r.prompts[0, :2] = 0
    r.response_mask[1, :, -2:] = False
    r.responses[1, :, -2:] = 0
    batch = strat.prepare_from_rollout(r)

    assert batch["chosen"].shape == (3, 11)
    assert batch["rejected"].shape == (3, 11)
    assert batch["chosen_mask"].shape == (3, 11)
    assert batch["rejected_mask"].shape == (3, 11)
    idx = torch.arange(3, device=cpu_device)
    expected_best = r.responses[idx, r.rewards.argmax(dim=-1)]
    expected_worst = r.responses[idx, r.rewards.argmin(dim=-1)]
    expected_best_mask = r.response_mask[idx, r.rewards.argmax(dim=-1)]
    expected_worst_mask = r.response_mask[idx, r.rewards.argmin(dim=-1)]

    assert torch.equal(batch["chosen"][:, :6], r.prompts)
    assert torch.equal(batch["rejected"][:, :6], r.prompts)
    assert torch.equal(batch["chosen"][:, 6:], expected_best)
    assert torch.equal(batch["rejected"][:, 6:], expected_worst)
    assert not batch["chosen_mask"][:, :6].any()
    assert not batch["rejected_mask"][:, :6].any()
    assert torch.equal(batch["chosen_mask"][:, 6:], expected_best_mask)
    assert torch.equal(batch["rejected_mask"][:, 6:], expected_worst_mask)
    assert torch.equal(batch["chosen_attention_mask"][:, :6], r.prompt_mask)
    assert torch.equal(batch["rejected_attention_mask"][:, :6], r.prompt_mask)
    assert torch.equal(batch["chosen_attention_mask"][:, 6:], expected_best_mask)
    assert torch.equal(batch["rejected_attention_mask"][:, 6:], expected_worst_mask)


@pytest.mark.parametrize("make_strategy", [_make_grpo, _make_dpo], ids=["grpo", "dpo"])
def test_policy_gradients_leave_behavior_and_reference_frozen(
    cpu_device, make_strategy
):
    strategy = make_strategy(cpu_device)
    rollout = make_rollout_result(device=cpu_device)
    rollout.logprobs_old.requires_grad_()
    loss = strategy(strategy.prepare_from_rollout(rollout))["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in strategy.model.parameters()
    )
    assert all(parameter.grad is None for parameter in strategy.ref_model.parameters())
    assert rollout.logprobs_old.grad is None
