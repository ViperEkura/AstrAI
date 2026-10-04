import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.model.transformer import TransformerModel
from astrai.model.value import ValueModel
from tests.helpers import TINY_CONFIG


def _config(**overrides):
    return AutoRegressiveLMConfig(**{**TINY_CONFIG, **overrides})


def test_trunk_keys_live_under_model_prefix():
    torch.manual_seed(0)
    model = AutoRegressiveLM(_config())
    lm_keys = set(model.state_dict())
    assert all(key.startswith("model.") for key in lm_keys if key != "lm_head.weight")
    # the trunk's own view has no prefix; only the wrapper adds it
    assert {k.removeprefix("model.") for k in lm_keys} - {"lm_head.weight"} == set(
        model.model.state_dict()
    )


def test_tied_lm_state_dict_keeps_single_embedding():
    model = AutoRegressiveLM(_config(tie_word_embeddings=True))
    keys = set(model.state_dict())
    assert "model.embed_tokens.weight" in keys
    assert "lm_head.weight" not in keys


def test_value_model_carries_no_lm_head():
    model = ValueModel(_config())
    assert not any(key.startswith("lm_head.") for key in model.state_dict())
    assert not any("lm_head" in name for name, _ in model.named_parameters())
    assert {k.removeprefix("model.") for k in model.state_dict()} - {
        "value_head.weight",
        "value_head.bias",
    } == set(model.model.state_dict())


def test_value_model_warm_starts_from_policy():
    torch.manual_seed(0)
    policy = AutoRegressiveLM(_config())
    critic = ValueModel(policy.config)
    result = critic.load_state_dict(policy.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(key.startswith("value_head.") for key in result.missing_keys)
    assert torch.equal(
        critic.model.layers[0].attention.q_proj.weight,
        policy.model.layers[0].attention.q_proj.weight,
    )


def test_trunk_forward_matches_wrapper_forward():
    torch.manual_seed(0)
    model = AutoRegressiveLM(_config()).eval()
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (2, 5))
    with torch.inference_mode():
        hidden = model.model(ids)["hidden_states"]
        output = model(ids)
    assert torch.equal(hidden, output["hidden_states"])
