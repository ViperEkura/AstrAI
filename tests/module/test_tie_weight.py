import json
import os

import pytest
import safetensors.torch as st
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from tests.support.models import TINY_CONFIG


@pytest.mark.parametrize("tied", [False, True])
def test_tie_weight_mutation_follows_shared_storage(tied):
    model = AutoRegressiveLM(
        AutoRegressiveLMConfig(**{**TINY_CONFIG, "tie_word_embeddings": tied})
    )
    embedding = model.model.embed_tokens.weight
    head = model.lm_head.weight
    assert (head.data_ptr() == embedding.data_ptr()) == tied
    original = head.detach().clone()
    with torch.no_grad():
        embedding[0, 0] += 1
    if tied:
        torch.testing.assert_close(head, embedding)
        assert not torch.equal(head, original)
    else:
        torch.testing.assert_close(head, original)


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
    saved_weights = st.load_file(model_path)
    initial_head = model.lm_head.weight.detach().clone()

    with pytest.raises(RuntimeError, match="lm_head.weight"):
        model.load_state_dict(saved_weights)

    incompatible = model.load_state_dict(saved_weights, strict=False)
    assert incompatible.missing_keys == ["lm_head.weight"]
    assert incompatible.unexpected_keys == []
    assert torch.equal(model.lm_head.weight, initial_head)
    assert torch.equal(
        model.model.embed_tokens.weight, saved_weights["model.embed_tokens.weight"]
    )
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    assert "lm_head.weight" in model.state_dict()
