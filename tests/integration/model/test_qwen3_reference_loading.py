"""Optional independent HF implementation checks for dense Qwen3 loading."""

import json

import pytest
import torch

from astrai.config import TrainConfig
from astrai.model import AutoModel, AutoRegressiveLM
from astrai.trainer.train_context import TrainContextBuilder


@pytest.mark.parametrize("tied", [False, True])
def test_qwen3_explicit_head_and_qk_norm_match_hf_masked_logits(
    tmp_path, tied, monkeypatch
):
    monkeypatch.setenv("LOCAL_DEVICE", "cpu")
    hf = pytest.importorskip("transformers", minversion="4.51.3")
    torch.manual_seed(615)
    config = hf.Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=10000,
        tie_word_embeddings=tied,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    reference = hf.Qwen3ForCausalLM(config).eval()
    # Learned, nonuniform Q/K RMSNorm scales expose permutation errors that
    # the all-ones initialization would conceal.
    with torch.no_grad():
        for layer in reference.model.layers:
            layer.self_attn.q_norm.weight.uniform_(0.5, 1.5)
            layer.self_attn.k_norm.weight.uniform_(0.5, 1.5)
    reference.save_pretrained(tmp_path)
    mapping = {
        "version": 1,
        "required": ["vocab_size", "hidden_size", "num_hidden_layers", "head_dim"],
        "fields": {
            key: key
            for key in (
                "vocab_size",
                "hidden_size",
                "num_hidden_layers",
                "intermediate_size",
                "rms_norm_eps",
                "tie_word_embeddings",
                "max_position_embeddings",
                "rope_theta",
            )
        },
        "constants": {
            "attention.default_type": "gqa",
            "attention.qk_norm": True,
            "ffn_type": "mlp",
        },
        "weights": {"norm_weight_offset": 0, "skip_prefixes": []},
    }
    mapping["fields"].update(
        {
            "attention.num_heads": "num_attention_heads",
            "attention.num_kv_heads": "num_key_value_heads",
            "attention.gqa.head_dim": "head_dim",
        }
    )
    (tmp_path / "hf_mapping.json").write_text(json.dumps(mapping))
    native = AutoModel.from_pretrained(tmp_path, strict=True).eval()
    assert native.config.head_dim == 16
    cfg = TrainConfig(
        model_fn=lambda: AutoRegressiveLM(native.config),
        dataset=torch.utils.data.TensorDataset(torch.ones(2, 4, dtype=torch.long)),
        optimizer_fn=lambda model: torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler_fn=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        ),
        strategy="seq",
        device_type="cpu",
        dp_mode="none",
        num_workers=0,
        pin_memory=False,
        batch_per_device=1,
        grad_accum_steps=1,
        ckpt_dir=str(tmp_path / "unused-checkpoints"),
    )
    context = TrainContextBuilder(cfg).with_param_path(str(tmp_path)).build()
    context.model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 9, 7, 12]])
    mask = torch.tensor([[True] * 5, [False, False, True, True, True]])
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        expected = reference(
            ids, attention_mask=mask, position_ids=positions, use_cache=False
        ).logits[mask]
        actual = native(ids, input_mask=mask, position_ids=positions)["logits"][mask]
        train = context.model(ids, input_mask=mask, position_ids=positions)["logits"][
            mask
        ]
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(train, expected, rtol=2e-4, atol=2e-5)
