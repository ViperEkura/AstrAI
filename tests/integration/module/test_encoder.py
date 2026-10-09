import json
import os
import tempfile

import pytest
import safetensors.torch as st
import torch

from astrai.config.model_config import EncoderConfig
from astrai.model.automodel import ModelFactory
from astrai.model.encoder import EmbeddingEncoder
from tests.support.checkpoint import assert_state_dicts_equal
from tests.support.models import TINY_CONFIG


def _make_model(device, **kwargs):
    config = EncoderConfig(**{**TINY_CONFIG, **kwargs})
    return EmbeddingEncoder(config).to(device=device)


@pytest.mark.parametrize("pooling_type", ["mean", "cls", "last"])
def test_encoder_forward_pooling(pooling_type, device):
    model = _make_model(device, pooling_type=pooling_type)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, TINY_CONFIG["vocab_size"], (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        output = model(input_ids)

    assert output.shape == (batch_size, TINY_CONFIG["hidden_size"])
    assert not torch.isnan(output).any()


def test_encoder_forward_with_padding(device):
    model = _make_model(device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, TINY_CONFIG["vocab_size"], (batch_size, seq_len), device=device
    )
    input_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    input_mask[:, 4:] = False

    with torch.no_grad():
        output = model(input_ids, input_mask=input_mask)

    assert output.shape == (batch_size, TINY_CONFIG["hidden_size"])
    assert not torch.isnan(output).any()


def test_encoder_normalize(device):
    model = _make_model(device, pooling_type="mean", normalize_embeddings=True)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, TINY_CONFIG["vocab_size"], (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        output = model(input_ids)

    norms = output.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_encoder_register():
    assert ModelFactory.is_registered("embedding")
    cls = ModelFactory.get_component_class("embedding")
    assert cls is EmbeddingEncoder


def test_encoder_from_transformer_checkpoint(device):
    model = _make_model(device)
    state_dict = model.state_dict()
    state_dict["lm_head.weight"] = torch.randn(
        TINY_CONFIG["vocab_size"], TINY_CONFIG["hidden_size"], device=device
    )

    new_model = _make_model(device)
    new_model.load_state_dict(state_dict, strict=True)

    assert_state_dicts_equal(new_model.state_dict(), model.state_dict())


def test_encoder_save_load(device):
    with tempfile.TemporaryDirectory(prefix="encoder_test_") as test_dir:
        config_path = os.path.join(test_dir, "config.json")
        weights_path = os.path.join(test_dir, "model.safetensors")

        config_data = {**TINY_CONFIG, "pooling_type": "mean"}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = EncoderConfig.from_file(config_path)
        original = EmbeddingEncoder(config)
        st.save_file(original.state_dict(), weights_path)

        loaded = EmbeddingEncoder(config)
        loaded.load_state_dict(st.load_file(weights_path))

        assert_state_dicts_equal(original.state_dict(), loaded.state_dict())
