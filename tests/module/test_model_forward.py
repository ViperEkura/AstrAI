"""Dense/packed forward boundaries and MoE loss composition."""

import pytest
import torch
from torch import nn

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.inference.core.cache import BlockPool, KVCacheManager
from astrai.inference.worker.workspace import InferenceWorkspace
from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.trainer.strategy import (
    SEQStrategy,
    SFTStrategy,
    _collect_moe_diagnostics,
)
from tests.support.models import TINY_CONFIG


def _make_tiny_moe_config(**overrides) -> AutoRegressiveLMConfig:
    return AutoRegressiveLMConfig(
        **{
            **TINY_CONFIG,
            "ffn_type": "moe",
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "n_activated_experts": 2,
            "topk_method": "greedy",
            **overrides,
        }
    )


def _make_batch(config, batch_size=2, seq_len=8, with_extra=False):
    vocab = config.vocab_size
    batch = {
        "input_ids": torch.randint(0, vocab, (batch_size, seq_len)),
        "target_ids": torch.randint(0, vocab, (batch_size, seq_len)),
    }
    if with_extra:
        batch["position_ids"] = (
            torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        )
        batch["loss_mask"] = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return batch


def _make_moe_fixture(device):
    config = _make_tiny_moe_config()
    model = AutoRegressiveLM(config).to(device)
    model.train()
    return config, model


def test_model_forward_contract_uses_dense_training_and_packed_inference():
    config = AutoRegressiveLMConfig(**TINY_CONFIG)
    model = AutoRegressiveLM(config).eval()
    dense = model(torch.tensor([[1, 2, 3]]))
    assert dense["logits"].shape == (1, 3, config.vocab_size)

    pool = BlockPool(
        n_layers=config.num_hidden_layers,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_batch_size=1,
        max_seq_len=config.max_position_embeddings,
        device="cpu",
        dtype=torch.float32,
    )
    cache = KVCacheManager(pool)
    workspace = InferenceWorkspace(
        1,
        config.max_position_embeddings,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
        torch.device("cpu"),
        torch.float32,
    )
    assert cache.alloc_slots("t", [1, 2, 3])
    packed = model(
        torch.tensor([1, 2, 3]),
        position_ids=torch.arange(3),
        kv_cache=cache.bind(["t"], workspace, start_pos=0),
        fwd="prefill",
    )
    assert packed["logits"].shape == (3, config.vocab_size)

    with pytest.raises(ValueError, match="training input_ids"):
        model(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError, match="inference input_ids"):
        model(
            torch.tensor([[1, 2, 3]]),
            kv_cache=cache.bind(["t"], workspace, start_pos=0),
            fwd="prefill",
        )


def test_cached_prefill_key_mask_matches_dense_attention():
    torch.manual_seed(13)
    config = AutoRegressiveLMConfig(**TINY_CONFIG)
    model = AutoRegressiveLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    key_mask = torch.tensor([[True, False, True, True]])

    pool = BlockPool(
        n_layers=config.num_hidden_layers,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_batch_size=1,
        max_seq_len=config.max_position_embeddings,
        device="cpu",
        dtype=torch.float32,
    )
    cache = KVCacheManager(pool)
    workspace = InferenceWorkspace(
        1,
        config.max_position_embeddings,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
        torch.device("cpu"),
        torch.float32,
    )
    assert cache.alloc_slots("t", [1, 2, 3, 4])

    with torch.inference_mode():
        dense = model(ids, input_mask=key_mask)["logits"][:, 2:]
        model(
            ids[0, :2],
            input_mask=key_mask[:, :2],
            position_ids=torch.arange(2),
            kv_cache=cache.bind(["t"], workspace, start_pos=0, seq_ends=[2]),
            fwd="prefill",
        )
        chunk = model(
            ids[0, 2:],
            input_mask=key_mask,
            position_ids=torch.arange(2, 4),
            kv_cache=cache.bind(["t"], workspace, start_pos=2),
            fwd="prefill",
        )["logits"]

    torch.testing.assert_close(chunk, dense[0])


def test_forward_logits_positions_projects_only_requested_rows():
    config = AutoRegressiveLMConfig(**TINY_CONFIG)
    model = AutoRegressiveLM(config).eval()
    prompts = [[1, 2, 3], [4, 5]]
    last_rows = torch.tensor([len(prompts[0]) - 1, len(prompts) - 1 + len(prompts[1])])

    pool = BlockPool(
        n_layers=config.num_hidden_layers,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        max_batch_size=2,
        max_seq_len=config.max_position_embeddings,
        device="cpu",
        dtype=torch.float32,
    )
    cache = KVCacheManager(pool)
    workspace = InferenceWorkspace(
        2,
        config.max_position_embeddings,
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
        torch.device("cpu"),
        torch.float32,
    )
    for tid, ids in zip(("t1", "t2"), prompts):
        assert cache.alloc_slots(tid, ids)
    input_ids = torch.tensor(sum(prompts, []), dtype=torch.long)
    position_ids = torch.cat([torch.arange(len(p)) for p in prompts])
    with torch.inference_mode():
        kwargs = dict(
            position_ids=position_ids,
            kv_cache=cache.bind(["t1", "t2"], workspace, start_pos=0),
            fwd="prefill",
        )
        full = model(input_ids, **kwargs)
        sliced = model(input_ids, logits_positions=last_rows, **kwargs)

    assert full["logits"].shape == (5, config.vocab_size)
    assert sliced["logits"].shape == (2, config.vocab_size)
    # Projecting M=2 rows vs M=5 rows may pick different GEMM kernels and
    # differ in the last float32 bits — assert_close, not bit equality.
    torch.testing.assert_close(sliced["logits"], full["logits"][last_rows])
    assert torch.equal(sliced["hidden_states"], full["hidden_states"][last_rows])


def test_collect_moe_diagnostics_uniform_routing():
    """Uniform routing with top_k=2 → tie-breaking by index.

    torch.topk breaks ties by index, so with equal probabilities
    experts 0 and 1 always win over experts 2 and 3:
      - dead_expert_fraction = 2/4 = 0.5
      - load_ratios = [2, 2, 0, 0] → |ratio-1| = [1, 1, 1, 1] → mean = 1.0
      - load_imbalance_max = 2.0
    """
    probs = torch.ones(128, 4) / 4.0
    topk = torch.tensor([[0, 1]] * 128)
    diag = _collect_moe_diagnostics([{"probs": probs, "topk_indices": topk}])

    assert diag["router_entropy"] == pytest.approx(
        float(torch.log(torch.tensor(4.0))), abs=1e-5
    )
    assert diag["dead_expert_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert diag["load_imbalance_mean"] == pytest.approx(1.0, abs=1e-6)
    assert diag["load_imbalance_max"] == pytest.approx(2.0, abs=1e-6)


def test_moe_metrics_flow_through_wrapped_model(cpu_device):
    """DDP-like wrappers (no .config / get_moe_router_probs) still collect MoE metrics."""

    class ForwardOnlyWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module = model

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    config = _make_tiny_moe_config()
    model = AutoRegressiveLM(config).to(cpu_device)
    wrapped = ForwardOnlyWrapper(model)
    wrapped.train()

    strategy = SEQStrategy(wrapped, cpu_device, moe_aux_loss_coef=0.01)
    output = strategy.compute_loss_output(
        {
            "input_ids": torch.randint(0, config.vocab_size, (2, 8)),
            "target_ids": torch.randint(0, config.vocab_size, (2, 8)),
        }
    )

    assert "moe_aux_loss" in output["metrics"]
    assert "router_entropy" in strategy._moe_metrics


def test_seq_no_aux_loss_for_mlp_model(cpu_device):
    config = AutoRegressiveLMConfig(**TINY_CONFIG)
    mlp_config = config
    mlp_model = AutoRegressiveLM(mlp_config).to(cpu_device)
    mlp_model.train()

    strategy = SEQStrategy(
        mlp_model,
        cpu_device,
        moe_aux_loss_coef=0.01,
    )
    output = strategy.compute_loss_output(_make_batch(config))
    metrics = output["metrics"]

    assert "moe_aux_loss" not in metrics
    assert "moe_aux_loss_weighted" not in metrics
    assert metrics["loss"] == pytest.approx(metrics["task_loss"], abs=1e-6)
    assert strategy._moe_metrics == {}


@pytest.mark.parametrize("strategy_type", [SEQStrategy, SFTStrategy])
@pytest.mark.parametrize("coefficient", [0.0, 0.01])
def test_moe_aux_loss_composition_and_gradients(cpu_device, strategy_type, coefficient):
    config, model = _make_moe_fixture(cpu_device)
    batch = _make_batch(config, with_extra=strategy_type is SFTStrategy)
    strategy = strategy_type(model, cpu_device, moe_aux_loss_coef=coefficient)
    output = strategy.compute_loss_output(batch)
    reference = model(batch["input_ids"])
    task_loss = torch.nn.functional.cross_entropy(
        reference["logits"].flatten(0, 1), batch["target_ids"].flatten()
    )
    torch.testing.assert_close(
        output["loss"], task_loss + coefficient * reference["aux_loss"]
    )
    metrics = output["metrics"]
    assert metrics["moe_aux_loss_weighted"] == pytest.approx(
        coefficient * metrics["moe_aux_loss"], abs=1e-6
    )
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])
    assert "router_entropy" in strategy._moe_metrics
    output["loss"].backward()
    assert model.model.embed_tokens.weight.grad.abs().sum() > 0
