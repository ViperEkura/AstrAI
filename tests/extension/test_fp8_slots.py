"""Slot addressing: module-path identity for the fp8 per-weight state.

Pure host-side naming, so these run without CUDA or the extension.
"""

import gc

import pytest
import torch
import torch.nn as nn

from astrai.extension.policy.quantization.autocast import (
    assign_slots,
    clear_slots,
    fp8_slot_of,
    refresh_slots,
)
from astrai.model.components.embedding import Embedding
from astrai.model.components.linear import Linear

DIM = 8


def _model(layers=2, tie=False):
    model = nn.Module()
    model.embed_tokens = Embedding(16, DIM)
    model.layers = nn.ModuleList()
    for _ in range(layers):
        layer = nn.Module()
        layer.attention = nn.ModuleDict(
            {
                name: Linear(DIM, DIM)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            }
        )
        layer.mlp = nn.ModuleDict(
            {
                "up": Linear(DIM, 2 * DIM),
                "gate": Linear(DIM, 2 * DIM),
                "down": Linear(2 * DIM, DIM),
            }
        )
        model.layers.append(layer)
    model.lm_head = Linear(DIM, 16, bias=False)
    if tie:
        model.lm_head.weight = model.embed_tokens.weight
    return model


@pytest.fixture(autouse=True)
def _clean_table():
    clear_slots()
    yield
    clear_slots()


def _names(table):
    return {slot.name for slot in table}


def test_slot_lookup_is_identity_based():
    model = _model()
    table = assign_slots(model)
    weight = model.layers[0].attention.q_proj.weight

    slot_id = table.slot_of(weight)
    assert slot_id >= 0
    assert fp8_slot_of(weight) == slot_id
    assert table.name_of(weight) == "layers.0.attention.q_proj"

    # A distinct tensor is never a hit, however equal it looks.
    assert table.slot_of(weight.detach().clone()) == -1
    assert fp8_slot_of(torch.zeros_like(weight)) == -1


def test_ids_are_stable_across_reassign_and_replacements():
    model = _model(layers=2)
    table = assign_slots(model)
    before = {slot.name: slot.slot_id for slot in table}

    table.assign(model)
    assert {slot.name: slot.slot_id for slot in table} == before

    # Swapping the module object under the same path (the TP/FSDP rebuild
    # shape) keeps the slot: the path is the identity, not the instance.
    old_id = before["layers.0.attention.o_proj"]
    old_weight = model.layers[0].attention.o_proj.weight
    model.layers[0].attention.o_proj = Linear(DIM, DIM)
    table.assign(model)

    after = {slot.name: slot.slot_id for slot in table}
    assert after["layers.0.attention.o_proj"] == old_id
    replacement = model.layers[0].attention.o_proj.weight
    assert table.slot_of(replacement) == old_id
    assert table.slot_of(old_weight) == -1

    # Adding a path never aliases a live id.
    model.layers[0].attention.extra = Linear(DIM, DIM)
    table.assign(model)
    ids = [slot.slot_id for slot in table]
    assert len(set(ids)) == len(ids)


def test_rebind_follows_a_replaced_weight():
    model = _model()
    table = assign_slots(model)
    linear = model.layers[0].attention.q_proj
    old_slot = table.slot_of(linear.weight)

    replacement = nn.Parameter(torch.randn_like(linear.weight))
    linear.weight = replacement
    # The old address is still mapped (its weight is kept alive), the new one
    # is not yet.
    assert table.slot_of(replacement) == -1

    table.rebind(model)
    assert table.slot_of(replacement) == old_slot
    assert table.name_of(replacement) == "layers.0.attention.q_proj"


def test_stale_counts_swapped_weights_and_refresh_repairs_them():
    model = _model(layers=1)
    table = assign_slots(model)
    assert table.stale() == 0
    assert refresh_slots() is False  # nothing moved: the hot path stays free

    linear = model.layers[0].mlp.down
    replacement = nn.Parameter(torch.randn_like(linear.weight))
    linear.weight = replacement
    assert table.stale() == 1

    assert refresh_slots() is True
    assert table.stale() == 0
    assert table.name_of(replacement) == "layers.0.mlp.down"


def test_refresh_is_a_noop_once_the_model_is_gone():
    model = _model(layers=1)
    table = assign_slots(model)
    weight = model.layers[0].mlp.up.weight
    slot_id = table.slot_of(weight)

    del model
    gc.collect()

    assert refresh_slots() is False
    assert table.stale() == 0  # nothing to compare against, nothing repaired
    assert table.slot_of(weight) == slot_id


def test_tied_weight_is_one_slot_and_survives_either_module_name():
    model = _model(tie=True)
    table = assign_slots(model)

    weight = model.lm_head.weight
    assert weight is model.embed_tokens.weight
    assert table.slot_of(weight) >= 0
    assert table.name_of(weight) == "lm_head"
    # One shared weight, one slot — not one per module.
    assert len([s for s in table if s.name == "lm_head"]) == 1


def test_two_linears_sharing_one_weight_get_one_slot():
    model = _model(layers=1)
    model.layers[0].attention.k_proj.weight = model.layers[0].attention.q_proj.weight
    table = assign_slots(model)

    weight = model.layers[0].attention.q_proj.weight
    assert table.name_of(weight) == "layers.0.attention.q_proj"
    assert table.slot_of(model.layers[0].attention.k_proj.weight) == table.slot_of(
        weight
    )


def test_wrapper_prefixes_are_stripped():
    model = _model(layers=1)
    inner = model.layers[0]

    wrapper = nn.Module()
    wrapper._orig_mod = model
    table = assign_slots(wrapper)

    assert "layers.0.attention.q_proj" in _names(table)
    assert all(not s.name.startswith(("_orig_mod.", "module.")) for s in table)
    assert table.slot_of(inner.attention.q_proj.weight) >= 0


def test_ddp_prefix_is_stripped():
    model = _model(layers=1)
    wrapper = nn.Module()
    wrapper.module = model
    table = assign_slots(wrapper)
    assert "layers.0.mlp.up" in _names(table)


def test_mapping_stays_valid_while_the_model_is_dropped():
    model = _model(layers=1)
    table = assign_slots(model)
    weight = model.layers[0].mlp.up.weight
    slot_id = table.slot_of(weight)

    del model
    gc.collect()

    # The table holds a strong reference, so the address cannot be recycled
    # into a false hit — the same guarantee the C++ ActivationCast anchor
    # gives, and the reason ``by_weight`` is not keyed on bare ``data_ptr``.
    assert table.slot_of(weight) == slot_id

    table.clear()
    assert table.slot_of(weight) == -1
    assert len(table) == 0


def test_lora_wrapped_linear_keeps_the_base_weight_slot():
    lora = pytest.importorskip("astrai.model.components.lora")
    base = Linear(DIM, DIM)
    wrapped = lora.LoRALinear(base, r=2, alpha=4)
    model = nn.Module()
    model.proj = wrapped

    table = assign_slots(model)
    assert table.slot_of(wrapped.weight) >= 0
    assert table.name_of(wrapped.weight) == "proj"
    # ``register_parameter("weight", base.weight)`` keeps the same object, so a
    # slot assigned before injection would still resolve.
    assert table.slot_of(base.weight) == table.slot_of(wrapped.weight)
