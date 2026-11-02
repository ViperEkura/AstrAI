import pytest
import torch
from pydantic import ValidationError

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.model.components.mlp import MLP, DeepSeekMoE
from astrai.trainer.callbacks import GradientCheckpointingCallback
from tests.support.models import TINY_CONFIG

MOE_CONFIG = {
    **TINY_CONFIG,
    "ffn_type": "moe",
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "n_activated_experts": 2,
}


def test_gated_deltanet_forward_backward_and_causal_prefix():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gdn",
        gdn_num_key_heads=2,
        gdn_num_value_heads=4,
        gdn_key_head_dim=3,
        gdn_value_head_dim=2,
        gdn_conv_kernel_size=3,
    )
    model = AutoRegressiveLM(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids)
    outputs["logits"].square().mean().backward()

    layer = model.model.layers[0].attention
    assert layer.q_proj.weight.grad is not None
    assert torch.isfinite(layer.q_proj.weight.grad).all()
    assert layer.beta_proj.weight.grad is not None
    assert torch.isfinite(layer.beta_proj.weight.grad).all()

    model.eval()
    original = torch.tensor([[1, 2, 3, 4]])
    changed_suffix = torch.tensor([[1, 2, 8, 9]])
    with torch.no_grad():
        original_logits = model(original)["logits"]
        changed_logits = model(changed_suffix)["logits"]
    torch.testing.assert_close(original_logits[:, :2], changed_logits[:, :2])


def test_gated_deltanet_rejects_packed_document_mask():
    config = AutoRegressiveLMConfig(**TINY_CONFIG, attn_type="gdn")
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    with pytest.raises(ValueError, match="Packed document masks need GDN state resets"):
        model(input_ids, input_mask=mask)


def test_gated_deltanet_rejects_inference_cache():
    config = AutoRegressiveLMConfig(**TINY_CONFIG, attn_type="gdn")
    model = AutoRegressiveLM(config)
    with pytest.raises(NotImplementedError, match="not wired into the paged cache"):
        model(torch.tensor([1, 2]), kv_cache=object(), fwd="decode")


def test_gated_deltanet_rejects_invalid_configuration():
    with pytest.raises(
        ValidationError, match="gated deltanet dimensions must be positive"
    ):
        AutoRegressiveLMConfig(**TINY_CONFIG, attn_type="gdn", gdn_key_head_dim=0)

    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gdn",
        gdn_num_key_heads=2,
        gdn_num_value_heads=3,
    )
    with pytest.raises(ValueError, match="must be divisible"):
        AutoRegressiveLM(config)


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"mlp_only_layers": [0]}, (MLP, DeepSeekMoE)),
        ({"decoder_sparse_step": 2}, (MLP, DeepSeekMoE)),
        ({}, (DeepSeekMoE, DeepSeekMoE)),
    ],
    ids=["dense-first", "sparse-step", "all-moe"],
)
def test_moe_per_layer_ffn_resolution(overrides, expected):
    model = AutoRegressiveLM(AutoRegressiveLMConfig(**{**MOE_CONFIG, **overrides}))
    for layer, kind in zip(model.model.layers, expected, strict=True):
        assert isinstance(layer.mlp, kind)
        assert isinstance(layer.mlp, DeepSeekMoE) == (kind is DeepSeekMoE)


def test_moe_aux_loss_only_emitted_during_training():
    config = AutoRegressiveLMConfig(**MOE_CONFIG, topk_method="greedy")
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    outputs = model(input_ids)
    assert outputs["aux_loss"].ndim == 0
    assert outputs["aux_loss"].requires_grad
    assert torch.isfinite(outputs["aux_loss"])

    with torch.no_grad():
        outputs = model(input_ids)
    assert "aux_loss" not in outputs

    model.eval()
    outputs = model(input_ids)
    assert "aux_loss" not in outputs


def test_moe_router_selects_experts_from_fp32_probabilities():
    """BF16 storage must not collapse close router probabilities before top-k."""

    class FixedRouter(torch.nn.Module):
        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", logits)

        def forward(self, x):
            return self.logits.expand(x.size(0), -1)

    class IdentityExpert(torch.nn.Module):
        def forward(self, x):
            return {"hidden_states": x, "aux_loss": None, "router_stats": None}

    moe = DeepSeekMoE(
        dim=2,
        dim_ffn=4,
        n_routed_experts=8,
        n_shared_experts=0,
        n_activated_experts=2,
    )
    logits = (torch.arange(8, dtype=torch.float32) * 0.001).to(torch.bfloat16)
    moe.router = FixedRouter(logits)
    moe.routed_experts = torch.nn.ModuleList([IdentityExpert() for _ in range(8)])

    routed = moe._routed_forward(torch.ones(1, 2, dtype=torch.bfloat16), True)
    stats = routed["router_stats"]
    expected = torch.topk(logits.float().softmax(-1), 2, sorted=False).indices

    assert stats is not None
    assert stats["probs"].dtype == torch.float32
    assert routed["hidden_states"].dtype == torch.bfloat16
    assert routed["aux_loss"].dtype == torch.float32
    actual = torch.sort(stats["topk_indices"], dim=-1).values
    expected = torch.sort(expected.unsqueeze(0), dim=-1).values
    assert torch.equal(actual, expected)


def test_moe_routing_replays_across_forward_recompute_and_reload(tmp_path):
    """Actual expert dispatch stays stable across online-RL replay boundaries."""

    kwargs = {
        "dim": 8,
        "dim_ffn": 16,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
    }
    torch.manual_seed(3407)
    moe = DeepSeekMoE(**kwargs).to(dtype=torch.bfloat16)
    moe.apply(
        lambda module: (
            module.reset_parameters() if hasattr(module, "reset_parameters") else None
        )
    )
    hidden_states = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    routes = []
    original_select = moe._select_experts

    def record_selection(router_logits):
        selected = original_select(router_logits)
        routes.append(torch.sort(selected[2].detach(), dim=-1).values.clone())
        return selected

    moe._select_experts = record_selection

    moe.eval()
    with torch.no_grad():
        rollout_output = moe(hidden_states)["hidden_states"].clone()
    rollout_route = routes[-1]

    moe.train()
    train_input = hidden_states.clone().requires_grad_(True)
    train_output = moe(train_input)["hidden_states"]
    train_output.float().sum().backward()
    training_route = routes[-1]
    assert torch.equal(training_route, rollout_route)

    moe.zero_grad(set_to_none=True)
    checkpointing = GradientCheckpointingCallback(modules=[DeepSeekMoE])
    checkpointing._enable(moe)
    recompute_start = len(routes)
    recompute_input = hidden_states.clone().requires_grad_(True)
    recompute_output = moe(recompute_input)["hidden_states"]
    recompute_output.float().sum().backward()
    recompute_routes = routes[recompute_start:]
    checkpointing._disable(moe)

    assert len(recompute_routes) >= 2
    assert all(torch.equal(route, rollout_route) for route in recompute_routes)

    checkpoint = tmp_path / "moe-router.pt"
    torch.save(moe.state_dict(), checkpoint)
    resumed = DeepSeekMoE(**kwargs).to(dtype=torch.bfloat16)
    resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
    resumed.eval()
    resumed_routes = []
    resumed_select = resumed._select_experts

    def record_resumed_selection(router_logits):
        selected = resumed_select(router_logits)
        resumed_routes.append(torch.sort(selected[2].detach(), dim=-1).values.clone())
        return selected

    resumed._select_experts = record_resumed_selection
    with torch.no_grad():
        resumed_output = resumed(hidden_states)["hidden_states"]

    assert torch.equal(resumed_routes[-1], rollout_route)
    torch.testing.assert_close(resumed_output, rollout_output, rtol=0, atol=0)


@pytest.mark.parametrize("decoder_sparse_step", [0, -1])
def test_moe_rejects_invalid_decoder_sparse_step(decoder_sparse_step):
    with pytest.raises(ValidationError, match="decoder_sparse_step must be at least 1"):
        AutoRegressiveLMConfig(
            **TINY_CONFIG,
            ffn_type="moe",
            n_routed_experts=4,
            n_activated_experts=2,
            decoder_sparse_step=decoder_sparse_step,
        )


@pytest.mark.parametrize(
    "moe,training",
    [(True, True), (True, False), (False, True)],
    ids=["moe-training", "moe-eval", "dense-training"],
)
def test_router_diagnostics_follow_model_mode(moe, training):
    config = AutoRegressiveLMConfig(**(MOE_CONFIG if moe else TINY_CONFIG))
    model = AutoRegressiveLM(config).train(training)
    with torch.set_grad_enabled(training):
        output = model(torch.randint(0, config.vocab_size, (2, 8)))
    assert ("router_stats" in output) == (moe and training)
    if moe and training:
        assert len(output["router_stats"]) == config.num_hidden_layers
        for stats in output["router_stats"]:
            assert stats["probs"].shape == (16, 4)
            assert stats["topk_indices"].shape == (16, 2)
    elif not moe:
        assert "aux_loss" not in output


@pytest.mark.parametrize(
    "factory,base,error",
    [
        (AutoRegressiveLMConfig, {**TINY_CONFIG, "ffn_type": "moe"}, ValidationError),
        (DeepSeekMoE, {"dim": 8, "dim_ffn": 16}, ValueError),
    ],
    ids=["config", "component"],
)
@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"n_routed_experts": 0}, "n_routed_experts must be positive"),
        ({"n_shared_experts": -1}, "n_shared_experts must be non-negative"),
        ({"n_activated_experts": 0}, "n_activated_experts must be positive"),
        (
            {"n_routed_experts": 2, "n_activated_experts": 3},
            "cannot exceed n_routed_experts",
        ),
        ({"topk_method": "group_limited_greedy"}, "unsupported topk_method"),
    ],
)
def test_moe_rejects_inconsistent_expert_topology(
    factory, base, error, overrides, message
):
    kwargs = dict(
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
    with pytest.raises(error, match=message):
        factory(**{**base, **kwargs, **overrides})
