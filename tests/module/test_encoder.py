import torch

from astrai.config.model_config import EncoderConfig
from astrai.model.encoder import EmbeddingEncoder
from tests.support.models import TINY_CONFIG


def _make_model(device, **kwargs):
    config = EncoderConfig(**{**TINY_CONFIG, **kwargs})
    return EmbeddingEncoder(config).to(device=device)


def test_encoder_normalize(cpu_device):
    model = _make_model(cpu_device, pooling_type="mean", normalize_embeddings=True)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, TINY_CONFIG["vocab_size"], (batch_size, seq_len), device=cpu_device
    )

    with torch.no_grad():
        output = model(input_ids)

    norms = output.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_encoder_from_transformer_checkpoint(cpu_device):
    model = _make_model(cpu_device)
    state_dict = model.state_dict()
    state_dict["lm_head.weight"] = torch.randn(
        TINY_CONFIG["vocab_size"], TINY_CONFIG["hidden_size"], device=cpu_device
    )

    new_model = _make_model(cpu_device)
    new_model.load_state_dict(state_dict, strict=True)

    torch.testing.assert_close(
        new_model.state_dict(), model.state_dict(), rtol=0, atol=0
    )
