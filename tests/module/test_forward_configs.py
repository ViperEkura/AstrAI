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


def test_moe_router_probs_populated_after_forward():
    """Verify DeepSeekMoE._router_probs is set after forward in training mode."""
    from astrai.config.model_config import AutoRegressiveLMConfig
    from astrai.model.components.mlp import DeepSeekMoE

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
        model(input_ids)

    # All MoE layers should have router_probs set
    moe_layers = [m for m in model.modules() if isinstance(m, DeepSeekMoE)]
    assert len(moe_layers) > 0
    for layer in moe_layers:
        assert layer._router_probs is not None
        assert layer._router_probs.ndim == 2
        assert layer._router_probs.shape[-1] == 4  # n_routed_experts


def test_get_moe_router_probs_moe_model():
    """Verify get_moe_router_probs() returns a list of tensors for MoE models."""
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )
    model = AutoRegressiveLM(config)
    model.train()

    with torch.enable_grad():
        model(torch.randint(0, config.vocab_size, (2, 8)))

    probs = model.get_moe_router_probs()
    assert isinstance(probs, list)
    assert len(probs) == 2  # num_hidden_layers
    for p in probs:
        assert p.ndim == 2
        assert p.shape[-1] == 4


def test_get_moe_router_probs_non_moe_model():
    """Verify get_moe_router_probs() returns empty list for non-MoE models."""
    from astrai.config.model_config import AutoRegressiveLMConfig

    config = AutoRegressiveLMConfig(**TINY_CONFIG, ffn_type="mlp")
    model = AutoRegressiveLM(config)

    probs_untrained = model.get_moe_router_probs()
    assert probs_untrained == []

    model.train()
    with torch.enable_grad():
        model(torch.randint(0, config.vocab_size, (2, 8)))

    probs = model.get_moe_router_probs()
    assert probs == []


def test_collect_router_probs_static_method():
    """Verify DeepSeekMoE.collect_router_probs static method."""
    from astrai.model.components.mlp import DeepSeekMoE

    moe = DeepSeekMoE(
        dim=8,
        dim_ffn=16,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )
    moe.train()
    with torch.enable_grad():
        moe(torch.randn(2, 8, 8))

    # collect_router_probs should find the MoE layer
    probs = DeepSeekMoE.collect_router_probs(moe)
    assert len(probs) == 1
    assert probs[0].shape[-1] == 4

    # On a plain MLP module, should return empty
    mlp_module = MLP(8, 16)
    assert DeepSeekMoE.collect_router_probs(mlp_module) == []


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
    assert output["aux_loss"] is not None


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
