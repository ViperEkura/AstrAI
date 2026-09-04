import pytest
import torch

from astrai.model.components.mlp import MLP, DeepSeekMoE
from astrai.model.transformer import AutoRegressiveLM
from tests.helpers import TINY_CONFIG

CONFIGS = [
    pytest.param(
        {**TINY_CONFIG, "attn_type": "gqa", "ffn_type": "mlp"},
        id="gqa_mlp",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "mla",
            "ffn_type": "mlp",
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
        },
        id="mla_mlp",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "topk_method": "greedy",
        },
        id="gqa_moe",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "topk_method": "greedy",
            "mlp_only_layers": [0],
        },
        id="gqa_moe_dense_first",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "topk_method": "greedy",
            "decoder_sparse_step": 2,
        },
        id="gqa_moe_sparse_step",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "topk_method": "greedy",
            "norm_topk_prob": True,
        },
        id="gqa_moe_norm_topk",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "topk_method": "greedy",
            "moe_intermediate_size": 24,
            "shared_expert_intermediate_size": 20,
        },
        id="gqa_moe_custom_intermediate",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "mlp",
            "rope_theta": 100000.0,
        },
        id="gqa_rope_theta",
    ),
    pytest.param(
        {**TINY_CONFIG, "attn_type": "gqa", "ffn_type": "mlp", "use_qk_norm": True},
        id="gqa_qk_norm",
    ),
    pytest.param(
        {
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "mlp",
            "tie_word_embeddings": True,
        },
        id="gqa_tie_word_embeddings",
    ),
]


@pytest.mark.parametrize("config_kwargs", CONFIGS)
def test_model_forward(config_kwargs, device):
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(**config_kwargs)
    model = AutoRegressiveLM(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        output = model(input_ids)

    assert "logits" in output
    assert "hidden_states" in output
    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert output["hidden_states"].shape == (
        batch_size,
        seq_len,
        config.hidden_size,
    )
    assert not torch.isnan(output["logits"]).any()
    assert not torch.isnan(output["hidden_states"]).any()


@pytest.mark.parametrize("config_kwargs", CONFIGS)
def test_model_forward_with_padding(config_kwargs, device):
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(**config_kwargs)
    model = AutoRegressiveLM(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    input_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    input_mask[:, 4:] = False

    with torch.no_grad():
        output = model(input_ids, input_mask=input_mask)

    assert output["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert not torch.isnan(output["logits"]).any()


def test_moe_per_layer_ffn_resolution():
    """Verify that mlp_only_layers and decoder_sparse_step resolve FFN types correctly."""
    from astrai.config.model_config import AutoRegressiveLMConfig

    # mlp_only_layers: first layer dense, rest MoE
    config = AutoRegressiveLMConfig(
        **{
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "mlp_only_layers": [0],
        }
    )
    model = AutoRegressiveLM(config)
    assert isinstance(model.layers[0].mlp, MLP)
    assert not isinstance(model.layers[0].mlp, DeepSeekMoE)
    assert isinstance(model.layers[1].mlp, DeepSeekMoE)

    # decoder_sparse_step=2: every other layer is MoE
    config2 = AutoRegressiveLMConfig(
        **{
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "decoder_sparse_step": 2,
        }
    )
    model2 = AutoRegressiveLM(config2)
    # layer 0 (id=0): (0+1)%2=1 != 0 -> MLP
    assert isinstance(model2.layers[0].mlp, MLP)
    assert not isinstance(model2.layers[0].mlp, DeepSeekMoE)
    # layer 1 (id=1): (1+1)%2=0 -> MoE
    assert isinstance(model2.layers[1].mlp, DeepSeekMoE)

    # decoder_sparse_step=1 (default): all layers MoE
    config3 = AutoRegressiveLMConfig(
        **{
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
        }
    )
    model3 = AutoRegressiveLM(config3)
    for layer in model3.layers:
        assert isinstance(layer.mlp, DeepSeekMoE)


def test_moe_custom_intermediate_shape():
    """Verify MoE uses custom intermediate sizes when specified."""
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(
        **{
            **TINY_CONFIG,
            "attn_type": "gqa",
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "moe_intermediate_size": 24,
            "shared_expert_intermediate_size": 20,
        }
    )
    model = AutoRegressiveLM(config)
    moe_layer = model.layers[0].mlp
    assert isinstance(moe_layer, DeepSeekMoE)
    # routed experts use moe_intermediate_size
    for expert in moe_layer.routed_experts:
        assert expert.up.weight.shape[0] == 24
        assert expert.gate.weight.shape[0] == 24
        assert expert.down.weight.shape[1] == 24
    # shared experts use shared_expert_intermediate_size
    for expert in moe_layer.shared_experts:
        assert expert.up.weight.shape[0] == 20
        assert expert.gate.weight.shape[0] == 20
        assert expert.down.weight.shape[1] == 20


def test_moe_defaults_preserve_normalized_routing():
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
    model = AutoRegressiveLM(config)

    assert config.norm_topk_prob is True
    assert model.layers[0].mlp.norm_topk_prob is True


def test_moe_router_stats_in_output_during_training():
    """Verify forward output carries per-layer router_stats in training mode."""
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
    model = AutoRegressiveLM(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.enable_grad():
        outputs = model(input_ids)

    stats = outputs["router_stats"]
    assert isinstance(stats, list)
    assert len(stats) == config.num_hidden_layers
    for s in stats:
        assert s["probs"].shape == (2 * 8, 4)  # (N, n_routed_experts)
        assert s["topk_indices"].shape == (2 * 8, 2)  # (N, n_activated_experts)


def test_moe_router_stats_absent_in_eval():
    """Verify no router_stats are emitted outside training."""
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )
    model = AutoRegressiveLM(config)
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randint(0, config.vocab_size, (2, 8)))

    assert "router_stats" not in outputs


def test_no_router_stats_for_mlp_model():
    """Verify pure MLP models emit no router_stats and no aux_loss."""
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(**TINY_CONFIG, ffn_type="mlp")
    model = AutoRegressiveLM(config)
    model.train()

    with torch.enable_grad():
        outputs = model(torch.randint(0, config.vocab_size, (2, 8)))

    assert "router_stats" not in outputs
    assert "aux_loss" not in outputs


def test_moe_aux_loss_only_emitted_during_training():
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
    )
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


def test_moe_component_forward_returns_ffn_output():
    from astrai.model.components.mlp import DeepSeekMoE

    moe = DeepSeekMoE(
        dim=8,
        dim_ffn=16,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )

    output = moe(torch.randn(2, 8, 8))

    assert output["hidden_states"].shape == (2, 8, 8)
    assert output["aux_loss"].ndim == 0
    assert output["aux_loss"].requires_grad
    assert torch.isfinite(output["aux_loss"])


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
    from astrai.trainer.train_callback import GradientCheckpointingCallback

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


@pytest.mark.parametrize(
    ("overrides", "message"),
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
def test_moe_rejects_inconsistent_expert_topology(overrides, message):
    from pydantic import ValidationError

    from astrai.config.model_config import AutoRegressiveLMConfig

    kwargs = {
        **TINY_CONFIG,
        "ffn_type": "moe",
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "topk_method": "greedy",
        **overrides,
    }
    with pytest.raises(ValidationError, match=message):
        AutoRegressiveLMConfig(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
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
def test_moe_component_rejects_inconsistent_expert_topology(overrides, message):
    kwargs = {
        "dim": 8,
        "dim_ffn": 16,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "n_activated_experts": 2,
        "topk_method": "greedy",
        **overrides,
    }
    with pytest.raises(ValueError, match=message):
        DeepSeekMoE(**kwargs)


@pytest.mark.parametrize("decoder_sparse_step", [0, -1])
def test_moe_rejects_invalid_decoder_sparse_step(decoder_sparse_step):
    from pydantic import ValidationError

    from astrai.config.model_config import AutoRegressiveLMConfig

    with pytest.raises(ValidationError, match="decoder_sparse_step must be at least 1"):
        AutoRegressiveLMConfig(
            **TINY_CONFIG,
            ffn_type="moe",
            n_routed_experts=4,
            n_activated_experts=2,
            decoder_sparse_step=decoder_sparse_step,
        )
