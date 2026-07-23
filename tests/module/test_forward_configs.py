import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM

TINY_CONFIG = dict(
    vocab_size=128,
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=1,
    intermediate_size=16,
    max_position_embeddings=64,
    num_hidden_layers=2,
    rms_norm_eps=1e-5,
)


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
            "router_aux_loss_coef": 0.01,
            "router_z_loss_coef": 0.001,
        },
        id="gqa_moe",
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
def test_model_forward(config_kwargs):
    config = AutoRegressiveLMConfig(**config_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    if config.ffn_type == "moe":
        assert output["router_loss"].ndim == 0
        assert output["router_aux_loss"].ndim == 0
        assert output["router_z_loss"].ndim == 0
        assert output["router_entropy"].ndim == 0
        assert output["router_expert_load"].shape == (config.n_routed_experts,)
        assert torch.allclose(
            output["router_expert_load"].sum(),
            torch.ones((), device=device),
        )


@pytest.mark.parametrize("config_kwargs", CONFIGS)
def test_model_forward_with_padding(config_kwargs):
    config = AutoRegressiveLMConfig(**config_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def test_moe_router_loss_backpropagates_to_router():
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        topk_method="greedy",
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
    )
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(input_ids)

    output["router_loss"].backward()

    for layer in model.layers:
        grad = layer.mlp.router.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_mqa_uses_native_gqa_sdpa(monkeypatch):
    import astrai.model.components.attention as attention_module

    def fail_repeat(*args, **kwargs):
        raise AssertionError("MQA should not physically repeat KV heads")

    monkeypatch.setattr(attention_module, "repeat_kv", fail_repeat)
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        attn_type="gqa",
        ffn_type="mlp",
    )
    model = AutoRegressiveLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(input_ids)
    assert output["logits"].shape == (2, 8, config.vocab_size)
