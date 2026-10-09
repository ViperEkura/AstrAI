import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import torch
from safetensors.torch import save_file

from astrai.model.automodel import AutoModel
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.model.components.conv import Conv1d
from astrai.model.components.initialization import skip_parameter_init
from astrai.model.components.linear import Linear
from tests.support.models import make_tiny_config


def _write_config(path, config):
    (path / "config.json").write_text(json.dumps(config.to_dict()))


def test_full_checkpoint_preserves_weights_forward_and_tie(tmp_path):
    config = make_tiny_config(tie_word_embeddings=True)
    source = AutoRegressiveLM(config).eval()
    source.save_pretrained(tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path).eval()
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    for key, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(tokens)["logits"], source(tokens)["logits"], rtol=0, atol=0
        )


def test_missing_and_partial_weights_initialize_missing_parameters(tmp_path):
    config = make_tiny_config()
    _write_config(tmp_path, config)
    torch.manual_seed(91)
    expected = AutoRegressiveLM(config)
    torch.manual_seed(91)
    without_weights = AutoModel.from_pretrained(tmp_path)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(
            without_weights.state_dict()[key], value, rtol=0, atol=0
        )

    key = "model.embed_tokens.weight"
    save_file(
        {key: torch.ones_like(expected.state_dict()[key])},
        tmp_path / "model.safetensors",
    )
    torch.manual_seed(91)
    partial = AutoModel.from_pretrained(tmp_path, strict=False)
    assert torch.all(partial.state_dict()[key] == 1)
    missing = "model.layers.0.attention.q_proj.weight"
    torch.testing.assert_close(
        partial.state_dict()[missing], expected.state_dict()[missing], rtol=0, atol=0
    )


def test_scoped_skip_does_not_change_torch_init_or_other_thread():
    original = torch.nn.init.kaiming_uniform_
    barrier = Barrier(2)

    def fast():
        with skip_parameter_init(True):
            skipped = Linear(16, 16)
            conv = Conv1d(16, 16, 3)
            barrier.wait()
            return skipped, conv

    def normal():
        barrier.wait()
        return Linear(16, 16), Conv1d(16, 16, 3)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(fast)
        b = pool.submit(normal)
        skipped, skipped_conv = a.result()
        initialized, initialized_conv = b.result()
    assert torch.nn.init.kaiming_uniform_ is original
    assert skipped.weight.shape == initialized.weight.shape
    assert skipped_conv.weight.shape == initialized_conv.weight.shape
    assert torch.count_nonzero(initialized.weight) > 0
    assert torch.count_nonzero(initialized_conv.weight) > 0


def test_local_conv1d_matches_torch_convolution():
    torch.manual_seed(38)
    reference = torch.nn.Conv1d(6, 6, 4, groups=6, bias=False)
    torch.manual_seed(38)
    local = Conv1d(6, 6, 4, groups=6, bias=False)
    torch.testing.assert_close(local.weight, reference.weight, rtol=0, atol=0)
    signal = torch.randn(2, 6, 9)
    torch.testing.assert_close(local(signal), reference(signal), rtol=0, atol=0)


def test_custom_model_uses_scoped_fast_load_for_astrai_layers(tmp_path, monkeypatch):
    from astrai.model.automodel import ModelFactory
    from astrai.model.components.initialization import should_initialize

    config = make_tiny_config()
    observations = []

    class CustomLM(AutoRegressiveLM):
        def __init__(self, cfg):
            observations.append(should_initialize())
            super().__init__(cfg)
            self.custom_head = torch.nn.Linear(cfg.hidden_size, 2)

    source = CustomLM(config)
    source.save_pretrained(tmp_path)
    monkeypatch.setattr(
        ModelFactory, "get_component_class", classmethod(lambda cls, name: CustomLM)
    )
    loaded = AutoModel.from_pretrained(tmp_path)
    assert observations == [True, False]
    for key, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
