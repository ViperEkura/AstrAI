import json
import os

import safetensors.torch as st
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import TINY_CONFIG


def test_tie_weight_init(base_test_env):
    config_path = base_test_env["config_path"]

    config_data = TINY_CONFIG.copy()
    config_data["tie_word_embeddings"] = True

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    config = AutoRegressiveLMConfig.from_file(config_path)
    model = AutoRegressiveLM(config)

    assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    original_weight = model.model.embed_tokens.weight.clone()
    model.model.embed_tokens.weight.data[0, 0] = 100.0

    assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    assert not torch.equal(model.lm_head.weight, original_weight)

    config_data["tie_word_embeddings"] = False

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    config = AutoRegressiveLMConfig.from_file(config_path)
    model = AutoRegressiveLM(config)

    assert not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()

    original_weight = model.model.embed_tokens.weight.clone()
    model.model.embed_tokens.weight.data[0, 0] = 100.0

    assert not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    assert not torch.equal(model.lm_head.weight, original_weight)


def test_model_save_load_with_tie_weight(base_test_env):
    test_dir = base_test_env["test_dir"]
    model_path = os.path.join(test_dir, "model.safetensors")

    config_data = TINY_CONFIG.copy()
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

    assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert "lm_head.weight" not in model.state_dict()

    config_data["tie_word_embeddings"] = False
    with open(config_path, "w") as f:
        json.dump(config_data, f)

    loaded_config = AutoRegressiveLMConfig.from_file(config_path)
    model = AutoRegressiveLM(loaded_config)
    model.load_state_dict(st.load_file(model_path))

    assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    assert "lm_head.weight" in model.state_dict()
