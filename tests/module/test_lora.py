import pytest
import torch

from astrai.model import AutoRegressiveLM
from astrai.model.components.linear import Linear
from astrai.model.components.lora import (
    LoRALinear,
    _get_lora_count,
    inject_lora,
    load_lora,
    merge_lora,
    save_lora,
)
from tests.support.models import make_tiny_config

LORA_MODEL_KWARGS = dict(
    vocab_size=1000,
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    max_position_embeddings=32,
)


def _make_model(**kwargs):
    config = make_tiny_config(**{**LORA_MODEL_KWARGS, **kwargs})
    model = AutoRegressiveLM(config)
    model.eval()
    return model


@pytest.mark.parametrize("delta", [0.0, 1.0])
def test_loralinear_forward_and_merge(delta):
    base = Linear(4, 4)
    lora = LoRALinear(base, r=2, alpha=4)
    x = torch.randn(2, 4)
    with torch.no_grad():
        lora.lora_B.fill_(delta)
        expected = base(x) + (4 / 2) * torch.nn.functional.linear(
            torch.nn.functional.linear(x, lora.lora_A), lora.lora_B
        )
        torch.testing.assert_close(lora(x), expected)
        lora.merge()
        torch.testing.assert_close(lora(x), expected)
        # A second merge must not apply the low-rank delta again.
        lora.merge()
        torch.testing.assert_close(lora(x), expected)


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


def test_only_lora_params_trainable():
    model = _make_model()
    inject_lora(model, r=4, alpha=8, target_modules={"q_proj", "v_proj"})

    for name, param in model.named_parameters():
        if isinstance(name.split(".")[-1], str) and "lora" in name:
            assert param.requires_grad, f"lora param should be trainable: {name}"
        elif any(name.endswith(f".{t}.weight") for t in ("q_proj", "v_proj")):
            assert not param.requires_grad, f"injected weight should be frozen: {name}"


@pytest.mark.parametrize("already_injected", [False, True])
def test_save_load_roundtrip(temp_dir, already_injected):
    model = _make_model()
    cfg = inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.fill_(0.5)

    x = torch.randint(0, 1000, (2, 16))
    with torch.no_grad():
        out_src = model(x)["logits"].clone()

    save_lora(model, temp_dir, cfg)

    model2 = _make_model()
    model2.load_state_dict(model.state_dict(), strict=False)
    if already_injected:
        inject_lora(model2, r=4, alpha=8, target_modules={"q_proj"})
    load_lora(model2, temp_dir)

    with torch.no_grad():
        out_dst = model2(x)["logits"]

    torch.testing.assert_close(out_src, out_dst)


def test_save_after_merge_raises(temp_dir):
    model = _make_model()
    cfg = inject_lora(model, r=4, alpha=8, target_modules={"q_proj"})

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.fill_(0.5)

    save_lora(model, temp_dir, cfg)
    merge_lora(model)

    with pytest.raises(RuntimeError, match="No LoRA parameters"):
        save_lora(model, temp_dir, cfg)


def test_load_lora_mismatched_r_raises(temp_dir):
    model = _make_model()
    cfg = inject_lora(model, r=8, alpha=16, target_modules={"q_proj"})

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.fill_(0.5)

    save_lora(model, temp_dir, cfg)

    model2 = _make_model()
    model2.load_state_dict(model.state_dict(), strict=False)
    inject_lora(model2, r=4, alpha=8, target_modules={"q_proj"})

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_lora(model2, temp_dir)


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
