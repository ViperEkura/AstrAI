import tempfile

import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model import AutoRegressiveLM
from astrai.model.components.linear import Linear
from astrai.model.components.lora import (
    LoRAConfig,
    LoRALinear,
    _collect_lora_info,
    _get_lora_count,
    inject_lora,
    load_lora,
    merge_lora,
    save_lora,
)

MODEL_KWARGS = dict(
    vocab_size=1000,
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    num_hidden_layers=2,
    max_position_embeddings=32,
    rms_norm_eps=1e-5,
)


def _make_model(**kwargs):
    kw = {**MODEL_KWARGS, **kwargs}
    config = AutoRegressiveLMConfig(**kw)
    model = AutoRegressiveLM(config)
    model.eval()
    return model


def test_loralinear_init():
    base = Linear(64, 128)
    lora = LoRALinear(base, r=8, alpha=16)

    assert lora.weight is base.weight
    assert not lora.weight.requires_grad
    assert lora.lora_A.shape == (8, 64)
    assert lora.lora_B.shape == (128, 8)
    assert lora.scaling == 2.0
    assert not lora._merged
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_loralinear_forward_init_zero_delta():
    base = Linear(4, 4)
    with torch.no_grad():
        base.weight.zero_()

    x = torch.randn(2, 4)
    lora = LoRALinear(base, r=2, alpha=2)
    base_out = base(x)
    lora_out = lora(x)

    assert torch.allclose(base_out, lora_out)


def test_loralinear_forward_with_delta():
    base = Linear(4, 4)
    with torch.no_grad():
        base.weight.zero_()

    x = torch.randn(2, 4)
    lora = LoRALinear(base, r=2, alpha=2)
    base_out = base(x)

    with torch.no_grad():
        lora.lora_B.fill_(1.0)

    lora_out = lora(x)
    assert not torch.allclose(base_out, lora_out)


def test_loralinear_merge():
    base = Linear(4, 4)
    with torch.no_grad():
        base.weight.zero_()

    x = torch.randn(2, 4)
    lora = LoRALinear(base, r=2, alpha=2)
    with torch.no_grad():
        lora.lora_B.fill_(1.0)

    out_before = lora(x).clone()
    lora.merge()
    out_after = lora(x)

    torch.testing.assert_close(out_before, out_after)
    assert lora._merged
    assert not hasattr(lora, "lora_A")


def test_loralinear_merge_is_idempotent():
    base = Linear(4, 4)
    with torch.no_grad():
        base.weight.zero_()

    lora = LoRALinear(base, r=2, alpha=2)
    with torch.no_grad():
        lora.lora_B.fill_(1.0)

    lora.merge()
    lora.merge()


def test_inject_lora_default_target():
    model = _make_model()
    n_before = sum(1 for m in model.modules() if isinstance(m, Linear))

    inject_lora(model, r=4, alpha=8)

    lora_count = _get_lora_count(model)
    assert lora_count > 0
    assert lora_count < n_before


def test_inject_lora_ffn():
    model = _make_model()
    from astrai.model.components.lora import TARGET_MODULES_FFN

    inject_lora(model, r=4, alpha=8, target_modules=TARGET_MODULES_FFN)
    assert _get_lora_count(model) > 0


def test_inject_lora_returns_config():
    model = _make_model()
    cfg = inject_lora(model, r=8, alpha=32)
    assert isinstance(cfg, LoRAConfig)
    assert cfg.r == 8
    assert cfg.alpha == 32


def test_inject_lora_no_matching_targets_warns(caplog):
    model = _make_model()
    inject_lora(model, r=4, alpha=8, target_modules={"nonexistent"})
    assert "No LoRA layers injected" in caplog.text


def test_inject_lora_preserves_base_output():
    model = _make_model()
    x = torch.randint(0, 1000, (2, 16))

    with torch.no_grad():
        out_before = model(x)["logits"].clone()

    inject_lora(model, r=4, alpha=8)

    with torch.no_grad():
        out_after = model(x)["logits"]

    torch.testing.assert_close(out_before, out_after)


def test_inject_lora_does_not_reinject():
    model = _make_model()
    inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})
    first_count = _get_lora_count(model)

    inject_lora(model, r=2, alpha=4, target_modules={"q_proj"})
    assert _get_lora_count(model) == first_count


def test_inject_lora_adds_new_modules():
    model = _make_model()
    inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})
    first = _get_lora_count(model)

    inject_lora(model, r=4, alpha=8, target_modules={"v_proj"})
    assert _get_lora_count(model) > first


def test_inject_lora_on_mla_model():
    model = _make_model(
        attn_type="mla", kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=16
    )
    inject_lora(model, r=4, alpha=8, target_modules={"q_proj", "o_proj"})
    assert _get_lora_count(model) > 0


def test_inject_lora_on_moe_model():
    model = _make_model(
        ffn_type="moe",
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        intermediate_size=32,
    )
    inject_lora(model, r=4, alpha=8, target_modules={"up", "gate", "down"})
    assert _get_lora_count(model) > 0


def test_state_dict_key_format():
    model = _make_model()
    inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})

    sd = model.state_dict()
    assert "layers.0.attention.q_proj.weight" in sd
    assert "layers.0.attention.q_proj.lora_A" in sd
    assert "layers.0.attention.q_proj.lora_B" in sd


def test_only_lora_params_trainable():
    model = _make_model()
    inject_lora(model, r=4, alpha=8, target_modules={"q_proj", "v_proj"})

    for name, param in model.named_parameters():
        if isinstance(name.split(".")[-1], str) and "lora" in name:
            assert param.requires_grad, f"lora param should be trainable: {name}"
        elif any(name.endswith(f".{t}.weight") for t in ("q_proj", "v_proj")):
            assert not param.requires_grad, f"injected weight should be frozen: {name}"


def test_state_dict_after_inject_consistent_with_original():
    model = _make_model()
    sd_before = {k: v for k, v in model.state_dict().items()}

    inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})
    sd_after = model.state_dict()

    # original keys unchanged
    for k in sd_before:
        assert k in sd_after
        assert sd_before[k].shape == sd_after[k].shape

    # new lora keys present
    lora_keys = [k for k in sd_after if "lora" in k]
    assert len(lora_keys) > 0


def test_save_load_roundtrip():
    model = _make_model()
    cfg = inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.fill_(0.5)

    x = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        out_src = model(x)["logits"].clone()

    tmpdir = tempfile.mkdtemp()
    save_lora(model, tmpdir, cfg)

    model2 = _make_model()
    model2.load_state_dict(model.state_dict(), strict=False)
    load_lora(model2, tmpdir)

    with torch.no_grad():
        out_dst = model2(x)["logits"]

    torch.testing.assert_close(out_src, out_dst)


def test_save_after_merge_raises():
    model = _make_model()
    cfg = inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.fill_(0.5)

    tmpdir = tempfile.mkdtemp()
    save_lora(model, tmpdir, cfg)
    merge_lora(model)

    tmpdir2 = tempfile.mkdtemp()
    with pytest.raises(RuntimeError, match="No LoRA parameters"):
        save_lora(model, tmpdir2, cfg)


def test_load_lora_on_already_injected():
    model = _make_model()
    inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.fill_(0.5)

    tmpdir = tempfile.mkdtemp()
    save_lora(model, tmpdir, LoRAConfig(r=4, alpha=8, target_modules=("q_proj",)))

    model2 = _make_model()
    model2.load_state_dict(model.state_dict(), strict=False)
    inject_lora(model2, r=4, alpha=8, target_modules={"q_proj"})

    # load onto already-injected model
    load_lora(model2, tmpdir)
    assert _get_lora_count(model2) > 0


def test_load_lora_mismatched_r_raises():
    model = _make_model()
    cfg = inject_lora(model, r=8, alpha=16, target_modules={"q_proj"})

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.fill_(0.5)

    tmpdir = tempfile.mkdtemp()
    save_lora(model, tmpdir, cfg)

    model2 = _make_model()
    model2.load_state_dict(model.state_dict(), strict=False)
    inject_lora(model2, r=4, alpha=8, target_modules={"q_proj"})

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_lora(model2, tmpdir)  # strict=False, only lora keys


def test_merge_preserves_output():
    model = _make_model()
    inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.fill_(0.5)

    x = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        out_before = model(x)["logits"].clone()

    merge_lora(model)

    with torch.no_grad():
        out_after = model(x)["logits"]
    torch.testing.assert_close(out_before, out_after)


def test_merge_no_lora_warns(caplog):
    model = _make_model()
    merge_lora(model)
    assert "No LoRA layers to merge" in caplog.text


def test_collect_lora_info():
    model = _make_model()
    info = _collect_lora_info(model)
    assert "q_proj" in info
    assert "o_proj" in info
    assert "q_proj" in info  # each layer has one
