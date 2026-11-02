import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.model.value import ValueModel
from tests.support.models import TINY_CONFIG


def test_value_model_warm_starts_from_policy():
    torch.manual_seed(0)
    policy = AutoRegressiveLM(AutoRegressiveLMConfig(**TINY_CONFIG))
    critic = ValueModel(policy.config)
    result = critic.load_state_dict(policy.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(key.startswith("value_head.") for key in result.missing_keys)
    assert torch.equal(
        critic.model.layers[0].attention.q_proj.weight,
        policy.model.layers[0].attention.q_proj.weight,
    )
