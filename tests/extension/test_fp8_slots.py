"""Slot addressing: module-path identity for the fp8 per-weight state.

Pure host-side naming, so these run without CUDA or the extension.
"""

import gc

import pytest
import torch
import torch.nn as nn

from astrai.extension.fp8_slots import (
    Fp8Slot,
    assign_slots,
    clear_slots,
    fp8_slot_of,
    slot_table,
)
from astrai.model.components.embedding import Embedding
from astrai.model.components.linear import Linear

DIM = 8


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = Linear(DIM, DIM)
        self.k_proj = Linear(DIM, DIM)
        self.v_proj = Linear(DIM, DIM)
        self.o_proj = Linear(DIM, DIM)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = Linear(DIM, 2 * DIM)
        self.gate = Linear(DIM, 2 * DIM)
        self.down = Linear(2 * DIM, DIM)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attn()
        self.mlp = _Mlp()


class _Model(nn.Module):
    def __init__(self, layers: int = 2, tie: bool = False):
        super().__init__()
        self.embed_tokens = Embedding(16, DIM)
        self.layers = nn.ModuleList(_Block() for _ in range(layers))
        self.lm_head = Linear(DIM, 16, bias=False)
        if tie:
            self.lm_head.weight = self.embed_tokens.weight


@pytest.fixture(autouse=True)
def _clean_table():
    clear_slots()
    yield
    clear_slots()


def _names(table):
    return {slot.name for slot in table}


def test_assign_names_every_linear():
    model = _Model(layers=2)
    table = assign_slots(model)

    # 2 blocks x 7 projections + lm_head; the embedding is not a linear.
    assert len(table) == 15
    assert "layers.0.attention.q_proj" in _names(table)
    assert "layers.1.mlp.down" in _names(table)
    assert "lm_head" in _names(table)
    assert "embed_tokens" not in _names(table)


def test_role_parse_and_pattern():
    table = assign_slots(_Model())
    by_name = {slot.name: slot for slot in table}

    q = by_name["layers.0.attention.q_proj"]
    assert (q.module_type, q.tensor_type) == ("attention", "q_proj")
    assert q.pattern == "layers.*.attention.q_proj"

    up = by_name["layers.1.mlp.up"]
    assert (up.module_type, up.tensor_type) == ("mlp", "up")

    head = by_name["lm_head"]
    assert (head.module_type, head.tensor_type) == ("", "lm_head")
    assert head.pattern == "lm_head"


def test_moe_expert_path_reports_container_not_index():
    model = _Model(layers=1)
    model.layers[0].mlp = nn.ModuleDict({"routed_experts": nn.ModuleList([_Mlp()])})
    table = assign_slots(model)

    expert = table.by_name("layers.0.mlp.routed_experts.0.up")
    assert expert is not None
    assert expert.module_type == "routed_experts"
    assert expert.pattern == "layers.*.mlp.routed_experts.*.up"


def test_slot_lookup_is_identity_based():
    model = _Model()
    table = assign_slots(model)
    weight = model.layers[0].attention.q_proj.weight

    slot_id = table.slot_of(weight)
    assert slot_id >= 0
    assert fp8_slot_of(weight) == slot_id
    assert table.name_of(weight) == "layers.0.attention.q_proj"

    # A distinct tensor is never a hit, however equal it looks.
    assert table.slot_of(weight.detach().clone()) == -1
    assert fp8_slot_of(torch.zeros_like(weight)) == -1


def test_module_attribute_is_attached():
    model = _Model()
    table = assign_slots(model)

    for name, module in model.named_modules():
        if isinstance(module, Linear):
            assert module._fp8_slot == table.slot_of(module.weight)


def test_ids_are_stable_across_reassign_and_replacements():
    model = _Model(layers=2)
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
    model = _Model()
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
    from astrai.extension.fp8_slots import refresh_slots

    model = _Model(layers=1)
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
    from astrai.extension.fp8_slots import refresh_slots

    model = _Model(layers=1)
    table = assign_slots(model)
    weight = model.layers[0].mlp.up.weight
    slot_id = table.slot_of(weight)

    del model
    gc.collect()

    assert refresh_slots() is False
    assert table.stale() == 0  # nothing to compare against, nothing repaired
    assert table.slot_of(weight) == slot_id


def test_tied_weight_is_one_slot_and_survives_either_module_name():
    model = _Model(tie=True)
    table = assign_slots(model)

    weight = model.lm_head.weight
    assert weight is model.embed_tokens.weight
    assert table.slot_of(weight) >= 0
    assert table.name_of(weight) == "lm_head"
    # One shared weight, one slot — not one per module.
    assert len([s for s in table if s.name == "lm_head"]) == 1


def test_two_linears_sharing_one_weight_get_one_slot():
    model = _Model(layers=1)
    model.layers[0].attention.k_proj.weight = model.layers[0].attention.q_proj.weight
    table = assign_slots(model)

    weight = model.layers[0].attention.q_proj.weight
    assert table.name_of(weight) == "layers.0.attention.q_proj"
    assert table.slot_of(model.layers[0].attention.k_proj.weight) == table.slot_of(
        weight
    )


def test_wrapper_prefixes_are_stripped():
    model = _Model(layers=1)
    inner = model.layers[0]

    wrapper = nn.Module()
    wrapper._orig_mod = model
    table = assign_slots(wrapper)

    assert "layers.0.attention.q_proj" in _names(table)
    assert all(not s.name.startswith(("_orig_mod.", "module.")) for s in table)
    assert table.slot_of(inner.attention.q_proj.weight) >= 0


def test_ddp_prefix_is_stripped():
    model = _Model(layers=1)
    wrapper = nn.Module()
    wrapper.module = model
    table = assign_slots(wrapper)
    assert "layers.0.mlp.up" in _names(table)


def test_mapping_stays_valid_while_the_model_is_dropped():
    model = _Model(layers=1)
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


def test_tablereports_unassigned_weights_as_minus_one():
    model = _Model(layers=1)
    table = assign_slots(model)
    stranger = torch.randn(4, 4)
    assert table.slot_of(stranger) == -1
    assert table.name_of(stranger) == ""
    assert slot_table() is table


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


def test_slot_is_hashable_and_frozen():
    slot = Fp8Slot(3, "layers.0.mlp.up", "mlp", "up")
    assert {slot: 1}[slot] == 1
    with pytest.raises(Exception):
        slot.slot_id = 4
