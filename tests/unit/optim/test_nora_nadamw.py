import math
from copy import deepcopy

import pytest
import torch
from torch.utils.data import TensorDataset

from astrai.config import TrainConfig
from astrai.model import AutoRegressiveLM
from astrai.model.components.linear import Linear
from astrai.model.components.lora import LoRALinear, inject_lora
from astrai.optim import (
    NAdamW,
    Nora,
    NoraNAdamW,
    OptimizerFactory,
    nora_lr_scale,
    partition_optimizer_parameters,
)
from astrai.trainer.schedule import SchedulerFactory
from tests.support.models import make_tiny_config


def _set_constant_grads(model, value):
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.full_like(param, value)


def test_nora_one_step_matches_row_geometry():
    param = torch.nn.Parameter(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
    grad = torch.tensor([[4.0, -3.0], [1.0, 1.0]])
    param.grad = grad.clone()

    optimizer = Nora([param], lr=0.1, beta=0.0, momentum=0.0)
    optimizer.step()

    theta_hat = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    tangent = grad - (grad * theta_hat).sum(dim=-1, keepdim=True) * theta_hat
    direction = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    expected = torch.tensor([[3.0, 4.0], [0.0, 2.0]]) - 0.1 * direction
    torch.testing.assert_close(param, expected)


def test_nora_handles_zero_and_pure_radial_rows():
    param = torch.nn.Parameter(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
    param.grad = torch.tensor([[3.0, 4.0], [6.0, 8.0]])

    optimizer = Nora([param], lr=0.1, beta=0.0, momentum=0.0)
    optimizer.step()

    torch.testing.assert_close(param[0], torch.tensor([-0.06, -0.08]))
    torch.testing.assert_close(param[1], torch.tensor([3.0, 4.0]), atol=1e-6, rtol=0)


def test_nora_lr_scale_only_increases_tall_matrices():
    assert nora_lr_scale(0.1, torch.Size([4, 2])) == pytest.approx(0.1 * math.sqrt(2.0))
    assert nora_lr_scale(0.1, torch.Size([2, 4])) == pytest.approx(0.1)


def test_nadamw_one_step_matches_reference_formula():
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    grad = torch.tensor([0.5, -0.25])
    param.grad = grad.clone()
    lr = 0.1
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    optimizer = NAdamW([param], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=0.2)
    optimizer.step()

    m = (1 - beta1) * grad
    v = (1 - beta2) * grad.square()
    m_hat = (beta1 * m + (1 - beta1) * grad) / (1 - beta1)
    v_hat = v / (1 - beta2)
    expected = torch.tensor([1.0, -2.0]) * (1 - lr * 0.2)
    expected.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
    torch.testing.assert_close(param, expected)


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_parameter_partition_is_complete_disjoint_and_role_based(
    tie_word_embeddings,
):
    model = AutoRegressiveLM(make_tiny_config(tie_word_embeddings=tie_word_embeddings))
    inject_lora(model, r=2, alpha=4, target_modules={"q_proj"})

    groups = partition_optimizer_parameters(model)
    all_grouped = [*groups.nora, *groups.nadamw_decay, *groups.nadamw_no_decay]
    trainable = [param for param in model.parameters() if param.requires_grad]

    assert len({id(param) for param in all_grouped}) == len(all_grouped)
    assert {id(param) for param in all_grouped} == {id(param) for param in trainable}
    assert id(model.model.embed_tokens.weight) in {
        id(p) for p in groups.nadamw_no_decay
    }
    assert id(model.lm_head.weight) in {id(p) for p in groups.nadamw_no_decay}

    nora_ids = {id(param) for param in groups.nora}
    no_decay_ids = {id(param) for param in groups.nadamw_no_decay}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            assert id(module.lora_A) in no_decay_ids
            assert id(module.lora_B) in no_decay_ids
        elif isinstance(module, Linear) and name != "lm_head":
            if module.weight.requires_grad:
                assert id(module.weight) in nora_ids


@pytest.mark.parametrize(
    "model_overrides",
    [
        {"attn_type": "gqa", "ffn_type": "mlp"},
        {
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "n_activated_experts": 1,
            "topk_method": "greedy",
        },
        {
            "attn_type": "mla",
            "ffn_type": "mlp",
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
        },
        {
            "attn_type": "mla",
            "ffn_type": "moe",
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "n_activated_experts": 1,
            "topk_method": "greedy",
        },
    ],
)
def test_parameter_partition_covers_all_model_structures(model_overrides):
    model = AutoRegressiveLM(make_tiny_config(**model_overrides))
    groups = partition_optimizer_parameters(model)

    grouped = [*groups.nora, *groups.nadamw_decay, *groups.nadamw_no_decay]
    trainable = [param for param in model.parameters() if param.requires_grad]
    assert {id(param) for param in grouped} == {id(param) for param in trainable}
    assert len(grouped) == len({id(param) for param in grouped})


def test_factory_registers_nora_default_and_legacy_muon():
    assert OptimizerFactory.list_registered() == [
        "mano_adamw",
        "muon_adamw",
        "nora_nadamw",
    ]
    model = AutoRegressiveLM(make_tiny_config())
    optimizer = OptimizerFactory.create("nora_nadamw", model, lr=3e-4)
    assert isinstance(optimizer, NoraNAdamW)


def test_scheduler_preserves_nora_to_nadamw_lr_ratio():
    model = AutoRegressiveLM(make_tiny_config())
    optimizer = NoraNAdamW(model, lr=3e-4, nora_lr=5e-3)
    scheduler = SchedulerFactory.create(
        "cosine", optimizer, warmup_steps=2, lr_decay_steps=2, min_rate=0.1
    )

    initial_ratio = optimizer.param_groups[0]["lr"] / optimizer.param_groups[-1]["lr"]
    _set_constant_grads(model, 0.1)
    optimizer.step()
    scheduler.step()
    stepped_ratio = optimizer.param_groups[0]["lr"] / optimizer.param_groups[-1]["lr"]

    assert initial_ratio == pytest.approx(5e-3 / 3e-4)
    assert stepped_ratio == pytest.approx(initial_ratio)


def test_optimizer_and_scheduler_resume_matches_uninterrupted_step():
    torch.manual_seed(7)
    model_a = AutoRegressiveLM(make_tiny_config())
    optimizer_a = NoraNAdamW(model_a, lr=3e-4, nora_lr=5e-3)
    scheduler_a = SchedulerFactory.create(
        "cosine", optimizer_a, warmup_steps=2, lr_decay_steps=4, min_rate=0.1
    )

    _set_constant_grads(model_a, 0.125)
    optimizer_a.step()
    scheduler_a.step()
    model_state = {key: value.clone() for key, value in model_a.state_dict().items()}
    optimizer_state = deepcopy(optimizer_a.state_dict())
    scheduler_state = deepcopy(scheduler_a.state_dict())

    model_b = AutoRegressiveLM(make_tiny_config())
    model_b.load_state_dict(model_state)
    optimizer_b = NoraNAdamW(model_b, lr=3e-4, nora_lr=5e-3)
    scheduler_b = SchedulerFactory.create(
        "cosine", optimizer_b, warmup_steps=2, lr_decay_steps=4, min_rate=0.1
    )
    optimizer_b.load_state_dict(optimizer_state)
    scheduler_b.load_state_dict(scheduler_state)

    _set_constant_grads(model_a, -0.25)
    _set_constant_grads(model_b, -0.25)
    optimizer_a.step()
    optimizer_b.step()
    scheduler_a.step()
    scheduler_b.step()

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(param_a, param_b)
    assert scheduler_a.get_last_lr() == pytest.approx(scheduler_b.get_last_lr())


def test_train_config_serializes_optimizer_metadata():
    config = TrainConfig(
        model_fn=lambda: torch.nn.Linear(2, 2),
        strategy="seq",
        dataset=TensorDataset(torch.zeros(1, 2)),
        optimizer_fn=lambda model: torch.optim.AdamW(model.parameters()),
        scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        ),
        optimizer_name="nora_nadamw",
        optimizer_hyperparameters={"lr": 3e-4, "nora_lr": 5e-3},
    )

    metadata = config.to_dict()

    assert metadata["optimizer_name"] == "nora_nadamw"
    assert metadata["optimizer_hyperparameters"] == {
        "lr": 3e-4,
        "nora_lr": 5e-3,
    }


def test_nora_nadamw_rejects_legacy_muon_state():
    model = AutoRegressiveLM(make_tiny_config())
    optimizer = NoraNAdamW(model)

    with pytest.raises(ValueError, match="muon_adamw"):
        optimizer.load_state_dict({"muon": {}, "adamw": {}})


def test_combined_optimizer_runs_closure_once():
    model = AutoRegressiveLM(make_tiny_config())
    optimizer = NoraNAdamW(model)
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        return torch.tensor(1.0, requires_grad=True)

    loss = optimizer.step(closure)

    assert calls == 1
    assert loss.item() == 1.0
