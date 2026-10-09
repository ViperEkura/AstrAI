import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.model.autoregressive_lm import AutoRegressiveLM

TINY_CONFIG = dict(
    vocab_size=1000,
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=1,
    intermediate_size=16,
    max_position_embeddings=64,
    num_hidden_layers=2,
    rms_norm_eps=1e-5,
)


def make_tiny_config(**overrides):
    """Create a tiny ``AutoRegressiveLMConfig`` for tests.

    All keyword arguments override ``TINY_CONFIG`` defaults.
    """
    return AutoRegressiveLMConfig(**{**TINY_CONFIG, **overrides})


def make_rollout_config(vocab_size=200, max_position_embeddings=64, **kwargs):
    """Create a tiny config sized for rollout / strategy tests."""
    return make_tiny_config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=max_position_embeddings,
        **kwargs,
    )


def make_model(device, **cfg_overrides):
    """Create a tiny ``AutoRegressiveLM`` on *device* and return ``(model, config)``."""
    cfg = make_rollout_config(**cfg_overrides)
    model = AutoRegressiveLM(cfg).to(device=device)
    model.eval()
    return model, cfg


def make_seeded_gqa_model(device, *, dtype=None, max_position_embeddings=64):
    """Seeded GQA model for parallel-path equivalence tests.

    ``dtype`` casts after the device move; the context-parallel ring path
    needs a half-precision dtype (flash/cuDNN SDPA), the tensor-parallel
    path stays fp32.
    """
    torch.manual_seed(3407)
    cfg = make_tiny_config(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
    )
    model = AutoRegressiveLM(cfg).to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.train()
    return model


def make_frozen(model, device):
    """Create a frozen, eval-mode copy of *model* with identical weights."""
    cfg = make_rollout_config()
    copy = AutoRegressiveLM(cfg).to(device=device)
    copy.load_state_dict(model.state_dict())
    copy.requires_grad_(False)
    copy.eval()
    return copy


D = 64
