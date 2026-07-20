import json
import os
import tempfile

import pytest
import safetensors.torch as st
import torch

from astrai.config.model_config import EncoderConfig
from astrai.model.automodel import AutoModel
from astrai.model.encoder import EmbeddingEncoder

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

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _make_model(**kwargs):
    config = EncoderConfig(**{**TINY_CONFIG, **kwargs})
    return EmbeddingEncoder(config).to(device=_device)


@pytest.mark.parametrize("pooling_type", ["mean", "cls", "last"])
def test_encoder_forward_pooling(pooling_type):
    model = _make_model(pooling_type=pooling_type)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, TINY_CONFIG["vocab_size"], (batch_size, seq_len), device=_device
    )

    with torch.no_grad():
        output = model(input_ids)

    assert output.shape == (batch_size, TINY_CONFIG["hidden_size"])
    assert not torch.isnan(output).any()


def test_encoder_forward_with_padding():
    model = _make_model()
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, TINY_CONFIG["vocab_size"], (batch_size, seq_len), device=_device
    )
    input_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=_device)
    input_mask[:, 4:] = False

    with torch.no_grad():
        output = model(input_ids, input_mask=input_mask)

    assert output.shape == (batch_size, TINY_CONFIG["hidden_size"])
    assert not torch.isnan(output).any()


def test_encoder_normalize():
    model = _make_model(pooling_type="mean", normalize_embeddings=True)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, TINY_CONFIG["vocab_size"], (batch_size, seq_len), device=_device
    )

    with torch.no_grad():
        output = model(input_ids)

    norms = output.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_encoder_register():
    assert AutoModel.is_registered("embedding")
    cls = AutoModel.get_component_class("embedding")
    assert cls is EmbeddingEncoder


def test_encoder_from_transformer_checkpoint():
    model = _make_model()
    state_dict = model.state_dict()
    state_dict["lm_head.weight"] = torch.randn(
        TINY_CONFIG["vocab_size"], TINY_CONFIG["hidden_size"], device=_device
    )

    new_model = _make_model()
    new_model.load_state_dict(state_dict, strict=True)

    for key in model.state_dict():
        assert torch.equal(new_model.state_dict()[key], model.state_dict()[key])


def test_encoder_save_load():
    test_dir = tempfile.mkdtemp(prefix="encoder_test_")
    config_path = os.path.join(test_dir, "config.json")
    weights_path = os.path.join(test_dir, "model.safetensors")

    try:
        config_data = {**TINY_CONFIG, "pooling_type": "mean"}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = EncoderConfig.from_file(config_path)
        original = EmbeddingEncoder(config)
        st.save_file(original.state_dict(), weights_path)

        loaded = EmbeddingEncoder(config)
        loaded.load_state_dict(st.load_file(weights_path))

        for key in original.state_dict():
            assert torch.equal(original.state_dict()[key], loaded.state_dict()[key])
    finally:
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, f))
            os.rmdir(test_dir)
