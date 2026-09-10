"""Integration tests for opt-in checkpoint route validation."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from astrai.model.components.mlp import DeepSeekMoE
from astrai.moe import RecomputeRouteMismatchError
from astrai.trainer.train_callback import GradientCheckpointingCallback
from astrai.trainer.trainer import Trainer


def _make_moe(device):
    model = DeepSeekMoE(
        dim=8,
        dim_ffn=16,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_layers=2,
    )
    model.apply(
        lambda module: (
            module.reset_parameters() if hasattr(module, "reset_parameters") else None
        )
    )
    return model.to(device)


def _loss(output):
    return output["hidden_states"].square().mean() + 0.01 * output["aux_loss"]


def test_checkpoint_validator_preserves_output_and_gradient_parity(device):
    torch.manual_seed(7)
    reference = _make_moe(device).train()
    checkpointed = copy.deepcopy(reference).train()
    reference_input = torch.randn(2, 3, 8, device=device, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_(True)

    reference_output = reference(reference_input)
    _loss(reference_output).backward()

    callback = GradientCheckpointingCallback(
        modules=[DeepSeekMoE], route_validation="record"
    )
    callback._enable(checkpointed)
    checkpointed_output = checkpointed(checkpointed_input)
    _loss(checkpointed_output).backward()

    torch.testing.assert_close(
        reference_output["hidden_states"], checkpointed_output["hidden_states"]
    )
    torch.testing.assert_close(reference_input.grad, checkpointed_input.grad)
    for (reference_name, reference_parameter), (
        checkpointed_name,
        checkpointed_parameter,
    ) in zip(reference.named_parameters(), checkpointed.named_parameters()):
        assert reference_name == checkpointed_name
        torch.testing.assert_close(
            reference_parameter.grad, checkpointed_parameter.grad
        )

    context = SimpleNamespace(metrics={}, rank=0)
    callback.on_batch_end(context)
    summary = callback.last_route_summary
    assert summary.forward_pair_count == 1
    assert summary.compared_pair_count == 1
    assert summary.exact_pair_count == 1
    assert summary.has_failure is False
    assert context.metrics["forward_recompute_route_match"] == 1.0
    callback._disable(checkpointed)


class _ChangingRouteModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4))
        self.calls = 0

    def forward(self, value):
        hidden = torch.sin(value @ self.weight)
        topk_indices = torch.tensor(
            [[0, 1], [2, 3]], dtype=torch.int64, device=value.device
        )
        if self.calls:
            topk_indices[0, 0] = 3
        self.calls += 1
        return {
            "hidden_states": hidden,
            "aux_loss": None,
            "router_stats": {"topk_indices": topk_indices},
        }


def test_error_mode_raises_on_all_checks_after_backward(device):
    module = _ChangingRouteModule().to(device)
    callback = GradientCheckpointingCallback(
        modules=[_ChangingRouteModule], route_validation="error"
    )
    callback._enable(module)
    value = torch.randn(2, 4, device=device, requires_grad=True)
    module(value)["hidden_states"].sum().backward()

    context = SimpleNamespace(metrics={}, rank=0)
    with pytest.raises(RecomputeRouteMismatchError, match="mismatch_pairs=1"):
        callback.on_batch_end(context)
    assert callback.last_route_summary.mismatch_pair_count == 1
    assert context.metrics["route_recompute_mismatch_tokens"] == 1.0
    callback._disable(module)


def test_record_mode_exposes_mismatch_without_changing_backward(device):
    module = _ChangingRouteModule().to(device)
    callback = GradientCheckpointingCallback(
        modules=[_ChangingRouteModule], route_validation="record"
    )
    callback._enable(module)
    value = torch.randn(2, 4, device=device, requires_grad=True)
    module(value)["hidden_states"].sum().backward()

    context = SimpleNamespace(metrics={}, rank=0)
    callback.on_batch_end(context)
    assert callback.last_route_summary.has_failure is True
    assert context.metrics["forward_recompute_route_match"] == 0.0
    assert value.grad is not None
    callback._disable(module)


def test_error_mode_rejects_a_forward_route_without_recomputation():
    callback = GradientCheckpointingCallback(route_validation="error")
    callback.route_diagnostics.record_forward_pair()
    context = SimpleNamespace(metrics={}, rank=0)

    with pytest.raises(RecomputeRouteMismatchError, match="unrecomputed_pairs=1"):
        callback.on_batch_end(context)
    assert context.metrics["route_recompute_unrecomputed_pairs"] == 1.0


def test_callback_rejects_unknown_route_validation_mode():
    with pytest.raises(ValueError, match="route_validation must be one of"):
        GradientCheckpointingCallback(route_validation="unknown")


def test_train_config_wires_route_validation_and_rejects_unsafe_combinations(
    train_config_factory,
    test_model,
    random_dataset,
    temp_dir,
    device,
):
    common = {
        "model_fn": lambda: test_model["model"],
        "dataset": random_dataset,
        "test_dir": temp_dir,
        "device": device,
    }
    config = train_config_factory(
        **common,
        gradient_checkpointing_modules=[DeepSeekMoE],
        gradient_checkpointing_route_validation="record",
    )
    callback = next(
        item
        for item in Trainer(config).callbacks
        if isinstance(item, GradientCheckpointingCallback)
    )
    assert callback.route_validation == "record"

    with pytest.raises(ValueError, match="must be one of"):
        train_config_factory(
            **common,
            gradient_checkpointing_route_validation="unknown",
        )
    with pytest.raises(ValueError, match="requires gradient_checkpointing_modules"):
        train_config_factory(
            **common,
            gradient_checkpointing_route_validation="record",
        )
    with pytest.raises(ValueError, match="not supported with torch.compile"):
        train_config_factory(
            **common,
            gradient_checkpointing_modules=[DeepSeekMoE],
            gradient_checkpointing_route_validation="record",
            compile_mode="default",
        )
