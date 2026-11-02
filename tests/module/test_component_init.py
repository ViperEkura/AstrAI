"""A component must be usable the moment it is constructed.

``Linear`` and ``Embedding`` allocate their weight with ``torch.empty``.  A
model built through ``AutoRegressiveLM`` re-initializes every submodule in
its ``apply(self._init_weights)`` pass, so the bytes ``torch.empty`` returned
are never observed there — but a directly constructed component has no such
pass and exposes whatever the allocator recycled.  That memory is zeroed on
first touch (dead layer) or, once a NaN block has been freed, NaN: a NaN
router weight makes the MoE ``aux_loss`` non-finite.
"""

import pytest
import torch

from astrai.model.components.embedding import Embedding
from astrai.model.components.linear import Linear
from astrai.model.components.mlp import DeepSeekMoE


def test_linear_initializes_its_weight():
    linear = Linear(64, 128, init_std=0.02)

    assert torch.isfinite(linear.weight).all()
    # Never-written (all zero) and recycled-NaN memory both fail this.
    assert linear.weight.std().item() == pytest.approx(0.02, rel=0.35)


def test_linear_initializes_its_bias():
    linear = Linear(64, 4, bias=True)

    assert torch.isfinite(linear.bias).all()
    assert linear.bias.abs().max().item() > 0


def test_embedding_initializes_its_weight():
    embedding = Embedding(128, 64)

    assert torch.isfinite(embedding.weight).all()
    assert embedding.weight.std().item() == pytest.approx(0.02, rel=0.35)


def test_standalone_moe_is_finite_without_model_level_init():
    moe = DeepSeekMoE(
        dim=8,
        dim_ffn=16,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
    )

    for name, param in moe.named_parameters():
        assert torch.isfinite(param).all(), name

    output = moe(torch.randn(2, 8, 8))

    assert torch.isfinite(output["hidden_states"]).all()
    assert output["aux_loss"].ndim == 0
    assert output["aux_loss"].requires_grad
    assert torch.isfinite(output["aux_loss"])
