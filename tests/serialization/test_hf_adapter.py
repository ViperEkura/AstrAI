"""Tests for HuggingFace checkpoint/config adaptation."""

import json

import pytest
import safetensors.torch as st
import torch
import torch.nn.functional as F

from astrai.config.model_config import ConfigFactory
from astrai.model import AutoModel, AutoRegressiveLM
from astrai.model.components.attention import GQA
from astrai.model.components.rope import get_rotary_emb
from astrai.serialization import (
    adapt_config,
    convert_hf_config,
    convert_hf_weights,
    looks_like_hf_state_dict,
    save_model,
)
from astrai.serialization.hf_adapter import _half_to_interleaved
from tests.helpers import assert_state_dicts_equal, make_tiny_config

LLAMA_RAW = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.44.0",
    "vocab_size": 1000,
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 16,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "rope_theta": 10000.0,
    "attention_bias": False,
    "mlp_bias": False,
    "head_dim": 4,
}

TEST_MAPPING = {
    "version": 1,
    "required": [
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
    ],
    "fields": {
        "vocab_size": "vocab_size",
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_hidden_layers",
        "intermediate_size": "intermediate_size",
        "rms_norm_eps": "rms_norm_eps",
        "tie_word_embeddings": "tie_word_embeddings",
        "max_position_embeddings": "max_position_embeddings",
        "rope_theta": "rope_theta",
        "attention.num_heads": "num_attention_heads",
        "attention.num_kv_heads": "num_key_value_heads",
        "attention.gqa.head_dim": "head_dim",
        "n_routed_experts": "num_local_experts",
        "n_activated_experts": "num_experts_per_tok",
        "n_shared_experts": "n_shared_experts",
        "moe_intermediate_size": "intermediate_size",
        "decoder_sparse_step": "decoder_sparse_step",
        "mlp_only_layers": "mlp_only_layers",
    },
    "defaults": {"attention.num_kv_heads": "num_attention_heads"},
}


def _write_mapping(model_dir, mapping=TEST_MAPPING):
    (model_dir / "hf_mapping.json").write_text(json.dumps(mapping))


MOE_RAW = {
    **LLAMA_RAW,
    "model_type": "mixtral",
    "intermediate_size": 16,
    "num_local_experts": 2,
    "num_experts_per_tok": 1,
    "n_shared_experts": 1,
}


def to_hf_keys(state_dict, head_dim=None):
    """Rename AstrAI state dict keys to HuggingFace LLaMA-style names.

    When *head_dim* is given, q/k projections and q/k norm weights are
    also converted from AstrAI interleaved RoPE coordinates to the HF
    half-split (rotate_half) convention, so the produced state dict is a
    faithful HF-layout checkpoint.
    """
    out = {}
    for key, tensor in state_dict.items():
        # the wrapper view prefixes trunk keys with "model."; the mapping
        # below works on the trunk-relative names
        key = key.removeprefix("model.")
        if head_dim is not None:
            name = key.split(".")
            is_qk_proj = (
                len(name) >= 4
                and name[2] == "attention"
                and name[3] in ("q_proj", "k_proj")
            )
            is_qk_norm = (
                len(name) >= 4
                and name[2] == "attention"
                and name[3] in ("q_norm", "k_norm")
                and name[4] == "weight"
            )
            if is_qk_proj or is_qk_norm:
                inv = torch.argsort(_half_to_interleaved(head_dim))
                rows = tensor.shape[0]
                if rows > head_dim:
                    blocks = torch.arange(rows // head_dim) * head_dim
                    idx = (blocks[:, None] + inv[None, :]).flatten()
                else:
                    idx = inv
                tensor = tensor.index_select(0, idx)
        if key == "embed_tokens.weight":
            out["model.embed_tokens.weight"] = tensor
        elif key == "norm.weight":
            out["model.norm.weight"] = tensor
        elif key.startswith("layers."):
            parts = key.split(".")
            layer = parts[1]
            if parts[2] == "attention":
                out[f"model.layers.{layer}.self_attn.{parts[3]}.{parts[4]}"] = tensor
            elif parts[2] == "input_norm":
                out[f"model.layers.{layer}.input_layernorm.weight"] = tensor
            elif parts[2] == "post_attention_norm":
                out[f"model.layers.{layer}.post_attention_layernorm.weight"] = tensor
            elif parts[2] == "mlp":
                if parts[3] in ("gate", "up", "down"):
                    out[f"model.layers.{layer}.mlp.{parts[3]}_proj.weight"] = tensor
                elif parts[3] == "router":
                    out[f"model.layers.{layer}.mlp.gate.weight"] = tensor
                elif parts[3] == "routed_experts":
                    sub, name = parts[4], parts[5]
                    out[
                        f"model.layers.{layer}.mlp.experts.{sub}.{name}_proj.weight"
                    ] = tensor
                elif parts[3] == "shared_experts":
                    sub, name = parts[4], parts[5]
                    out[
                        f"model.layers.{layer}.mlp.shared_experts.{sub}.{name}_proj.weight"
                    ] = tensor
        else:
            out[key] = tensor
    return out


MOE_KWARGS = {
    "ffn_type": "moe",
    "n_routed_experts": 2,
    "n_shared_experts": 1,
    "n_activated_experts": 1,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
}


def _hf_keyed_state_dict(model, cfg):
    return to_hf_keys(model.state_dict(), cfg.hidden_size // cfg.num_attention_heads)


def _assert_hf_roundtrip(cfg, convert_cfg=None):
    """HF-keyed weights of a fresh model must convert back exactly."""
    model = AutoRegressiveLM(cfg)
    sd = model.state_dict()
    converted = convert_hf_weights(_hf_keyed_state_dict(model, cfg), convert_cfg or cfg)
    assert_state_dicts_equal(converted, sd)


def _assert_hf_directory_load(tmp_path, cfg, raw_config):
    """Save HF-format weights and check from_pretrained reproduces logits."""
    model = AutoRegressiveLM(cfg).eval()
    save_model(
        config=raw_config,
        state_dict=_hf_keyed_state_dict(model, cfg),
        save_directory=str(tmp_path),
    )
    mapping = (
        {**TEST_MAPPING, "constants": {"ffn_type": "moe"}}
        if raw_config.get("num_local_experts")
        else TEST_MAPPING
    )
    _write_mapping(tmp_path, mapping)
    loaded = AutoModel.from_pretrained(tmp_path).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids)["logits"], model(input_ids)["logits"]
        )


def test_convert_hf_config_with_explicit_mapping():
    cfg = convert_hf_config(LLAMA_RAW, TEST_MAPPING)
    assert cfg["model_type"] == "autoregressive_lm"
    assert cfg["hidden_size"] == 8
    assert cfg["attention"]["num_kv_heads"] == 1
    loaded = ConfigFactory.load(cfg)
    assert loaded.num_attention_heads == 2


def test_convert_hf_config_defaults_kv_heads():
    raw = {k: v for k, v in LLAMA_RAW.items() if k != "num_key_value_heads"}
    cfg = ConfigFactory.load(convert_hf_config(raw, TEST_MAPPING))
    assert cfg.num_key_value_heads == 2


def test_convert_hf_config_requires_declared_fields():
    raw = {k: v for k, v in LLAMA_RAW.items() if k != "hidden_size"}
    with pytest.raises(ValueError, match="missing hidden_size"):
        convert_hf_config(raw, TEST_MAPPING)


def test_convert_hf_config_rejects_bias():
    with pytest.raises(NotImplementedError):
        convert_hf_config({**LLAMA_RAW, "attention_bias": True}, TEST_MAPPING)


def test_convert_hf_config_nested_source_and_operations():
    raw = {
        "model_type": "test_wrapper",
        "text_config": {
            **LLAMA_RAW,
            "layer_types": ["full_attention", "linear_attention"],
            "rope_parameters": {"partial_rotary_factor": 0.5},
        },
    }
    mapping = {
        **TEST_MAPPING,
        "source": "text_config",
        "operations": [
            {
                "op": "map_values",
                "source": "layer_types",
                "target": "attention.layers",
                "values": {"full_attention": "gqa", "linear_attention": "gdn"},
            },
            {
                "op": "multiply",
                "sources": ["head_dim", "rope_parameters.partial_rotary_factor"],
                "target": "attention.gqa.rotary_dim",
            },
        ],
    }
    cfg = convert_hf_config(raw, mapping)
    assert cfg["attention"]["layers"] == ["gqa", "gdn"]
    assert cfg["attention"]["gqa"]["rotary_dim"] == 2


def test_convert_hf_config_rejects_unknown_operation():
    with pytest.raises(ValueError, match="unknown HF mapping operation"):
        convert_hf_config(
            LLAMA_RAW,
            {**TEST_MAPPING, "operations": [{"op": "python", "target": "hidden_size"}]},
        )


def test_looks_like_hf_state_dict():
    assert looks_like_hf_state_dict({"model.layers.0.self_attn.q_proj.weight": 1})
    assert looks_like_hf_state_dict({"model.layers.0.input_layernorm.weight": 1})
    # "model." alone no longer implies HF: AstrAI trunk keys share the prefix
    assert not looks_like_hf_state_dict({"model.embed_tokens.weight": 1})
    assert not looks_like_hf_state_dict({"model.layers.0.attention.q_proj.weight": 1})
    assert not looks_like_hf_state_dict({"layers.0.attention.q_proj.weight": 1})


def test_adapt_config_passthrough():
    raw = dict(LLAMA_RAW, model_type="autoregressive_lm")
    assert adapt_config(raw) is raw


def test_adapt_config_uses_model_directory(tmp_path):
    _write_mapping(tmp_path)
    assert adapt_config(LLAMA_RAW, str(tmp_path))["hidden_size"] == 8


def test_adapt_config_without_mapping_is_unchanged(tmp_path):
    assert adapt_config(LLAMA_RAW, str(tmp_path)) is LLAMA_RAW


def test_load_mapping_rejects_unsupported_version(tmp_path):
    _write_mapping(tmp_path, {"version": 2})
    with pytest.raises(ValueError, match="version 1"):
        adapt_config(LLAMA_RAW, str(tmp_path))


def test_convert_hf_weights_dense_roundtrip():
    _assert_hf_roundtrip(make_tiny_config())


def test_convert_hf_weights_moe_roundtrip():
    cfg = make_tiny_config(**MOE_KWARGS)
    hf_raw = convert_hf_config(
        MOE_RAW, {**TEST_MAPPING, "constants": {"ffn_type": "moe"}}
    )
    hf_cfg = ConfigFactory.load(hf_raw)
    _assert_hf_roundtrip(cfg, convert_cfg=hf_cfg)


def test_convert_hf_weights_keeps_astrai_keys():
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg)
    converted = convert_hf_weights(dict(model.state_dict()), cfg)
    assert_state_dicts_equal(converted, model.state_dict())


def test_convert_hf_weights_skips_unmapped_keys():
    cfg = make_tiny_config()
    sd = {"model.rotary_emb.inv_freq": torch.zeros(4), "model.embed_tokens.weight": 1}
    converted = convert_hf_weights(sd, cfg)
    assert "model.embed_tokens.weight" in converted
    assert "model.rotary_emb.inv_freq" not in converted


def test_convert_hf_weights_rejects_mla():
    cfg = make_tiny_config(attn_type="mla", kv_lora_rank=2)
    sd = {"model.layers.0.self_attn.kv_a_proj_with_mqa.weight": 1}
    with pytest.raises(NotImplementedError):
        convert_hf_weights(sd, cfg)


def test_convert_hf_config_preserves_sparse_fields():
    raw = {
        **MOE_RAW,
        "decoder_sparse_step": 2,
        "mlp_only_layers": [0],
    }
    cfg = ConfigFactory.load(
        convert_hf_config(raw, {**TEST_MAPPING, "constants": {"ffn_type": "moe"}})
    )
    assert cfg.decoder_sparse_step == 2
    assert cfg.mlp_only_layers == [0]


def test_convert_hf_weights_moe_with_dense_layers_roundtrip():
    _assert_hf_roundtrip(
        make_tiny_config(**MOE_KWARGS, mlp_only_layers=[0], decoder_sparse_step=1)
    )


def test_convert_hf_weights_qwen2_moe_singular_shared_expert_roundtrip():
    cfg = make_tiny_config(**MOE_KWARGS)
    model = AutoRegressiveLM(cfg)
    hf_sd = {
        k.replace("shared_experts.", "shared_expert.", 1): v
        for k, v in _hf_keyed_state_dict(model, cfg).items()
    }
    converted = convert_hf_weights(hf_sd, cfg)
    assert_state_dicts_equal(converted, model.state_dict())


def test_convert_hf_weights_gemma_qk_norm_roundtrip():
    _assert_hf_roundtrip(make_tiny_config(use_qk_norm=True))


def test_from_pretrained_hf_directory(tmp_path):
    _assert_hf_directory_load(tmp_path, make_tiny_config(), LLAMA_RAW)


def test_from_pretrained_astrai_directory(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg).eval()
    save_model(
        config=cfg.to_dict(),
        state_dict=model.state_dict(),
        save_directory=str(tmp_path),
    )
    loaded = AutoModel.from_pretrained(tmp_path, disable_random_init=False)
    assert_state_dicts_equal(loaded.state_dict(), model.state_dict())


def test_from_pretrained_hf_config_with_astrai_weights(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg)
    save_model(
        config=LLAMA_RAW,
        state_dict=model.state_dict(),
        save_directory=str(tmp_path),
    )
    _write_mapping(tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path, weights_format="astrai")
    assert_state_dicts_equal(loaded.state_dict(), model.state_dict())


def test_from_pretrained_weights_format_hf_on_astrai_dir(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg)
    save_model(
        config=cfg.to_dict(),
        state_dict=model.state_dict(),
        save_directory=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="hf_mapping.json"):
        AutoModel.from_pretrained(
            tmp_path, disable_random_init=False, weights_format="hf"
        )


def test_from_pretrained_weights_format_astrai_rejects_hf(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg)
    save_model(
        config=LLAMA_RAW,
        state_dict=_hf_keyed_state_dict(model, cfg),
        save_directory=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="hf_mapping.json"):
        AutoModel.from_pretrained(tmp_path, weights_format="astrai")


def test_from_pretrained_invalid_weights_format(tmp_path):
    cfg = make_tiny_config()
    save_model(
        config=cfg.to_dict(),
        state_dict={},
        save_directory=str(tmp_path),
    )
    with pytest.raises(ValueError):
        AutoModel.from_pretrained(tmp_path, weights_format="llama")


def test_from_pretrained_hf_directory_sharded(tmp_path):
    cfg = make_tiny_config()
    model = AutoRegressiveLM(cfg).eval()
    hf_sd = _hf_keyed_state_dict(model, cfg)
    keys = sorted(hf_sd)
    split = len(keys) // 2
    shard_a = {k: hf_sd[k] for k in keys[:split]}
    shard_b = {k: hf_sd[k] for k in keys[split:]}
    st.save_file(shard_a, str(tmp_path / "model-00001-of-00002.safetensors"))
    st.save_file(shard_b, str(tmp_path / "model-00002-of-00002.safetensors"))
    index = {
        "metadata": {},
        "weight_map": {
            k: (
                "model-00001-of-00002.safetensors"
                if k in shard_a
                else "model-00002-of-00002.safetensors"
            )
            for k in keys
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(json.dumps(LLAMA_RAW))

    _write_mapping(tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids)["logits"], model(input_ids)["logits"]
        )


def _half_split_rope(q, theta=10000.0):
    """HF llama-style rotate_half RoPE on [batch, seq, heads, head_dim]."""
    b, s, h, d = q.shape
    inv_freq = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    freqs = torch.outer(torch.arange(s, dtype=torch.float64), inv_freq).float()
    cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]
    q1, q2 = q[..., : d // 2], q[..., d // 2 :]
    return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)


def _rms_norm_hf(t, weight, eps):
    t = t.float()
    t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)
    return weight.float() * t


def _hf_reference_attn(
    x, Wq, Wk, Wv, Wo, n_heads, n_kv, head_dim, q_norm_w=None, k_norm_w=None, eps=1e-5
):
    """Ground-truth HF attention: per-head RMSNorm BEFORE RoPE (half-split)."""
    b, s, dim = x.shape
    q = (x @ Wq.T).reshape(b, s, n_heads, head_dim).float()
    k = (x @ Wk.T).reshape(b, s, n_kv, head_dim).float()
    v = (x @ Wv.T).reshape(b, s, n_kv, head_dim).float()
    if q_norm_w is not None:
        q = _rms_norm_hf(q, q_norm_w, eps)
        k = _rms_norm_hf(k, k_norm_w, eps)
    q, k = _half_split_rope(q), _half_split_rope(k)
    rep = n_heads // n_kv
    k = k.repeat_interleave(rep, dim=2).transpose(1, 2)
    v = v.repeat_interleave(rep, dim=2).transpose(1, 2)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, is_causal=True)
    out = out.transpose(1, 2).reshape(b, s, n_heads * head_dim)
    return out @ Wo.T


def _run_converted_gqa(x, hf_sd, cfg):

    attn = GQA(
        dim=cfg.hidden_size,
        n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads,
        use_qk_norm=cfg.use_qk_norm,
        norm_eps=cfg.rms_norm_eps,
        use_gated_attention=False,
        layer_id=0,
    ).eval()
    converted = convert_hf_weights(hf_sd, cfg)
    local = {
        k.removeprefix("model.layers.0.attention."): v
        for k, v in converted.items()
        if k.startswith("model.layers.0.attention.")
    }
    attn.load_state_dict(local, strict=True)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    seq = x.shape[1]
    rot = get_rotary_emb(head_dim, seq)[None, :seq].expand(x.shape[0], seq, -1, -1)
    with torch.no_grad():
        return attn(x, rot, is_causal=True)


def test_hf_import_rope_permutation_matches_half_split_reference():
    torch.manual_seed(0)
    n_heads, n_kv, head_dim = 4, 2, 8
    dim = n_heads * head_dim
    Wq = torch.randn(n_heads * head_dim, dim)
    Wk = torch.randn(n_kv * head_dim, dim)
    Wv = torch.randn(n_kv * head_dim, dim)
    Wo = torch.randn(dim, dim)
    x = torch.randn(2, 16, dim)

    hf_sd = {
        "model.layers.0.self_attn.q_proj.weight": Wq,
        "model.layers.0.self_attn.k_proj.weight": Wk,
        "model.layers.0.self_attn.v_proj.weight": Wv,
        "model.layers.0.self_attn.o_proj.weight": Wo,
    }
    cfg = make_tiny_config(
        hidden_size=dim, num_attention_heads=n_heads, num_key_value_heads=n_kv
    )
    ref = _hf_reference_attn(x, Wq, Wk, Wv, Wo, n_heads, n_kv, head_dim)
    out = _run_converted_gqa(x, hf_sd, cfg)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_hf_import_qk_norm_matches_norm_before_rope_reference():
    torch.manual_seed(1)
    n_heads, n_kv, head_dim = 4, 2, 8
    dim = n_heads * head_dim
    Wq = torch.randn(n_heads * head_dim, dim)
    Wk = torch.randn(n_kv * head_dim, dim)
    Wv = torch.randn(n_kv * head_dim, dim)
    Wo = torch.randn(dim, dim)
    gq = torch.randn(head_dim)
    gk = torch.randn(head_dim)
    x = torch.randn(2, 16, dim)

    hf_sd = {
        "model.layers.0.self_attn.q_proj.weight": Wq,
        "model.layers.0.self_attn.k_proj.weight": Wk,
        "model.layers.0.self_attn.v_proj.weight": Wv,
        "model.layers.0.self_attn.o_proj.weight": Wo,
        "model.layers.0.self_attn.q_norm.weight": gq,
        "model.layers.0.self_attn.k_norm.weight": gk,
    }
    cfg = make_tiny_config(
        hidden_size=dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        use_qk_norm=True,
    )
    ref = _hf_reference_attn(
        x, Wq, Wk, Wv, Wo, n_heads, n_kv, head_dim, q_norm_w=gq, k_norm_w=gk
    )
    out = _run_converted_gqa(x, hf_sd, cfg)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_from_pretrained_hf_directory_with_moe(tmp_path):
    _assert_hf_directory_load(tmp_path, make_tiny_config(**MOE_KWARGS), MOE_RAW)


def test_hybrid_weight_layout_converts_fused_gates_and_gdn_parameters():
    cfg = ConfigFactory.load(
        {
            "model_type": "autoregressive_lm",
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "attention": {
                "layers": ["gdn", "gqa"],
                "num_heads": 2,
                "num_kv_heads": 1,
                "qk_norm": True,
                "output_gate": True,
                "gqa": {"head_dim": 8, "rotary_dim": 4},
                "gdn": {
                    "num_key_heads": 1,
                    "num_value_heads": 2,
                    "key_head_dim": 4,
                    "value_head_dim": 4,
                },
            },
        }
    )
    fused_q_gate = torch.arange(2 * 2 * 8 * 16).reshape(32, 16).float()
    qkv = torch.arange(16 * 16).reshape(16, 16).float()
    source = {
        "model.language_model.layers.1.self_attn.q_proj.weight": fused_q_gate,
        "model.language_model.layers.1.self_attn.q_norm.weight": torch.arange(
            8
        ).float(),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": qkv,
        "model.language_model.layers.0.linear_attn.A_log": torch.ones(2),
        "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(2),
        "model.language_model.layers.0.linear_attn.norm.weight": torch.full((4,), 0.5),
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(16),
        "model.visual.dummy.weight": torch.ones(1),
        "mtp.layers.0.dummy.weight": torch.ones(1),
    }
    mapping = {
        "weights": {
            "norm_weight_offset": 1,
            "skip_prefixes": ["model.visual.", "mtp."],
        }
    }
    converted = convert_hf_weights(source, cfg, mapping)
    paired = fused_q_gate.reshape(2, 2, 8, 16)
    perm = torch.tensor([0, 2, 1, 3, 4, 5, 6, 7])
    expected_q = paired[:, 0].index_select(1, perm).reshape(16, 16)
    torch.testing.assert_close(
        converted["model.layers.1.attention.q_proj.weight"], expected_q
    )
    torch.testing.assert_close(
        converted["model.layers.1.attention.gate.weight"],
        paired[:, 1].reshape(16, 16),
    )
    torch.testing.assert_close(
        converted["model.layers.1.attention.q_norm.weight"],
        torch.arange(8).float().index_select(0, perm) + 1,
    )
    for name, chunk in zip(("q", "k", "v"), qkv.split((4, 4, 8))):
        torch.testing.assert_close(
            converted[f"model.layers.0.attention.{name}_proj.weight"], chunk
        )
    assert converted["model.layers.0.attention.A_log"].shape == (2,)
    assert converted["model.layers.0.attention.dt_bias"].shape == (2,)
    torch.testing.assert_close(
        converted["model.layers.0.attention.g_norm.weight"], torch.full((4,), 0.5)
    )
    torch.testing.assert_close(
        converted["model.layers.0.input_norm.weight"], torch.ones(16)
    )
    assert len(converted) == 10


def test_from_pretrained_gated_zero_centered_hf_directory(tmp_path):
    cfg = make_tiny_config(use_gated_attention=True, use_qk_norm=True)
    reference = AutoRegressiveLM(cfg).eval()
    hf_state = to_hf_keys(reference.state_dict(), head_dim=4)
    for layer_id in range(cfg.num_hidden_layers):
        root = f"model.layers.{layer_id}.self_attn."
        q = hf_state.pop(root + "q_proj.weight").reshape(2, 4, 8)
        gate = hf_state.pop(root + "gate.weight").reshape(2, 4, 8)
        hf_state[root + "q_proj.weight"] = torch.stack((q, gate), dim=1).reshape(16, 8)
    centered_norms = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
    )
    for key in tuple(hf_state):
        if key == "model.norm.weight" or key.endswith(centered_norms):
            hf_state[key] = hf_state[key] - 1
    hf_state = {
        key.replace("model.", "model.language_model.", 1): value.contiguous()
        for key, value in hf_state.items()
    }
    hf_state["model.visual.dummy.weight"] = torch.ones(1)
    hf_state["mtp.layers.0.dummy.weight"] = torch.ones(1)
    save_model(LLAMA_RAW, hf_state, str(tmp_path))
    mapping = {
        **TEST_MAPPING,
        "constants": {"attention.qk_norm": True, "attention.output_gate": True},
        "weights": {
            "norm_weight_offset": 1,
            "skip_prefixes": ["model.visual.", "mtp."],
        },
    }
    _write_mapping(tmp_path, mapping)
    loaded = AutoModel.from_pretrained(tmp_path, strict=True).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids)["logits"], reference(input_ids)["logits"]
        )


@pytest.mark.parametrize(
    "weights",
    [
        {"norm_weight_offset": True},
        {"norm_weight_offset": "one"},
        {"skip_prefixes": "mtp."},
        {"skip_prefixes": [""]},
        {"unknown": 1},
    ],
)
def test_load_mapping_rejects_invalid_weight_options(tmp_path, weights):
    _write_mapping(tmp_path, {**TEST_MAPPING, "weights": weights})
    with pytest.raises(ValueError):
        adapt_config(LLAMA_RAW, str(tmp_path))
