import json
import os
import tempfile

import pytest
import safetensors.torch as st
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.transformer import AutoRegressiveLM


@pytest.fixture
def transformer_test_env():
    test_dir = tempfile.mkdtemp(prefix="transformer_test_")
    config_path = os.path.join(test_dir, "config.json")

    config = {
        "vocab_size": 1000,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 16,
        "max_position_embeddings": 64,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5,
    }

    with open(config_path, "w") as f:
        json.dump(config, f)

    yield {"test_dir": test_dir, "config_path": config_path, "config": config}

    if os.path.exists(test_dir):
        try:
            for file in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, file))
            os.rmdir(test_dir)
        except Exception:
            pass


def test_tie_weight_init(transformer_test_env):
    config_path = transformer_test_env["config_path"]
    config_data = transformer_test_env["config"].copy()

    # case 1: tie weight
    config_data["tie_word_embeddings"] = True

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    config = AutoRegressiveLMConfig.from_file(config_path)
    model = AutoRegressiveLM(config)

    assert torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()

    original_weight = model.embed_tokens.weight.clone()
    model.embed_tokens.weight.data[0, 0] = 100.0

    assert torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert not torch.equal(model.lm_head.weight, original_weight)

    # case 2: not tie weight
    config_data["tie_word_embeddings"] = False

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    config = AutoRegressiveLMConfig.from_file(config_path)
    model = AutoRegressiveLM(config)

    assert not torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()

    original_weight = model.embed_tokens.weight.clone()
    model.embed_tokens.weight.data[0, 0] = 100.0

    assert not torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert not torch.equal(model.lm_head.weight, original_weight)


def test_model_save_load_with_tie_weight(transformer_test_env):
    test_dir = transformer_test_env["test_dir"]
    model_path = os.path.join(test_dir, "model.safetensors")

    config_data = transformer_test_env["config"].copy()

    # case 1: tie weight
    config_data["tie_word_embeddings"] = True
    config_path = os.path.join(test_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    config = AutoRegressiveLMConfig.from_file(config_path)
    original_model = AutoRegressiveLM(config)

    st.save_file(original_model.state_dict(), model_path)

    loaded_config = AutoRegressiveLMConfig.from_file(config_path)
    model = AutoRegressiveLM(loaded_config)
    model.load_state_dict(st.load_file(model_path))

    assert torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    assert "lm_head.weight" not in model.state_dict()

    # case 2: not tie weight (form tie-weight state dict load)
    config_data["tie_word_embeddings"] = False
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    loaded_config = AutoRegressiveLMConfig.from_file(config_path)
    model = AutoRegressiveLM(loaded_config)
    model.load_state_dict(st.load_file(model_path))

    assert torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
    assert "lm_head.weight" in model.state_dict()
