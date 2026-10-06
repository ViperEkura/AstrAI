import pytest
import torch

from astrai.config.model_config import AutoRegressiveLMConfig, EncoderConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.model.encoder import EmbeddingEncoder
from astrai.model.masking import prepare_decoder_masks, process_attention_mask
from astrai.model.value import ValueModel
from tests.helpers import TINY_CONFIG


def _decoder_config(attn_type: str) -> AutoRegressiveLMConfig:
    extra = {}
    if attn_type == "mla":
        extra = {
            "kv_lora_rank": 4,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
        }
    return AutoRegressiveLMConfig(**TINY_CONFIG, attn_type=attn_type, **extra)


def _decoder_output(model, input_ids, input_mask=None):
    output = model(input_ids, input_mask=input_mask)
    return output["values"] if isinstance(model, ValueModel) else output["logits"]


def _padding_mask(input_ids, valid_tokens, additive):
    if additive:
        mask = torch.zeros_like(input_ids, dtype=torch.float32)
        mask[:, valid_tokens:] = float("-inf")
        return mask
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    mask[:, valid_tokens:] = False
    return mask


@pytest.mark.parametrize("model_cls", [AutoRegressiveLM, ValueModel])
@pytest.mark.parametrize("attn_type", ["gqa", "mla"])
@pytest.mark.parametrize("additive", [False, True], ids=["bool", "additive"])
def test_2d_padding_mask_keeps_decoder_causal(model_cls, attn_type, additive, device):
    torch.manual_seed(31415)
    model = model_cls(_decoder_config(attn_type)).to(device).eval()
    if isinstance(model, ValueModel):
        with torch.no_grad():
            model.value_head.weight.normal_()

    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
    changed = ids.clone()
    changed[:, 4:] = torch.tensor([[11, 12, 13, 14]], device=device)
    all_valid = _padding_mask(ids, ids.size(1), additive)
    right_padded = _padding_mask(ids, 4, additive)
    left_padded = all_valid.clone()
    left_padded[:, :2] = float("-inf") if additive else False
    changed_padding = ids.clone()
    changed_padding[:, :2] = torch.tensor([[21, 22]], device=device)

    with torch.no_grad():
        all_valid_output = _decoder_output(model, ids, all_valid)
        changed_output = _decoder_output(model, changed, all_valid)
        padded_output = _decoder_output(model, ids, right_padded)
        unpadded_output = _decoder_output(model, ids[:, :4])
        left_padded_output = _decoder_output(model, ids, left_padded)
        changed_padding_output = _decoder_output(model, changed_padding, left_padded)

    torch.testing.assert_close(all_valid_output[:, :4], changed_output[:, :4])
    torch.testing.assert_close(padded_output[:, :4], unpadded_output)
    torch.testing.assert_close(left_padded_output[:, 2:], changed_padding_output[:, 2:])


@pytest.mark.parametrize("shape", [(1, 3, 3), (1, 1, 3, 3)])
def test_explicit_attention_masks_keep_their_existing_layout(shape):
    mask = torch.ones(shape, dtype=torch.bool)

    actual = process_attention_mask(mask, causal=True)
    expected = mask[:, None] if mask.ndim == 3 else mask
    torch.testing.assert_close(actual, expected)


def test_hybrid_decoder_uses_layer_specific_padding_masks(device):
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        layer_types=["gqa", "gdn"],
    )
    model = AutoRegressiveLM(config).to(device).eval()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    mask = torch.tensor([[True, True, True, False]], device=device)
    with torch.no_grad():
        output = model(ids, input_mask=mask)
    assert torch.isfinite(output["logits"]).all()


def test_hybrid_decoder_rejects_additive_mask_for_gdn(device):
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        layer_types=["gqa", "gdn"],
    )
    model = AutoRegressiveLM(config).to(device)
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    mask = torch.tensor([[0.0, 0.0, 0.0, float("-inf")]], device=device)
    with pytest.raises(ValueError, match="boolean 2D right-padding mask"):
        model(ids, input_mask=mask)


def test_hybrid_mask_plan_keeps_recurrent_padding_2d(device):
    padding = torch.tensor([[True, True, False]], device=device)
    masks = prepare_decoder_masks(padding, ("gqa", "gdn", "mla"))
    assert masks["gdn"].tensor is padding
    assert masks["gdn"].is_causal
    assert masks["gqa"] is masks["mla"]
    assert masks["gqa"].tensor.shape == (1, 1, 3, 3)
    assert not masks["gqa"].is_causal


def test_hybrid_gqa_mla_use_their_own_rope_dimensions(device):
    config = AutoRegressiveLMConfig(
        **TINY_CONFIG,
        layer_types=["gqa", "mla"],
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
    )
    model = AutoRegressiveLM(config).to(device).eval()
    assert model.model.rotary_embeddings["gqa"].dim == 4
    assert model.model.rotary_embeddings["mla"].dim == 2
    with torch.no_grad():
        output = model(torch.tensor([[1, 2, 3, 4]], device=device))
    assert torch.isfinite(output["logits"]).all()


def test_encoder_2d_padding_mask_remains_bidirectional(device):
    torch.manual_seed(2718)
    config = EncoderConfig(**TINY_CONFIG, pooling_type="cls")
    model = EmbeddingEncoder(config).to(device).eval()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    changed = torch.tensor([[1, 2, 8, 9]], device=device)
    mask = torch.ones_like(ids, dtype=torch.bool)

    with torch.no_grad():
        original = model(ids, input_mask=mask)
        changed_output = model(changed, input_mask=mask)

    assert not torch.allclose(original, changed_output)
