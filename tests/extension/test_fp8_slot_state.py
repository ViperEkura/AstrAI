"""Slot-addressed fp8 state: module identity instead of weight address.

What the slot table buys over the ``(data_ptr, shape, dtype)`` key:
a replaced weight parameter keeps its rings, and a snapshot binds state to the
module path rather than to registration order.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from astrai.extension.fp8_slots import assign_slots, clear_slots
from astrai.extension.loader import get_module
from astrai.extension.quantize import fp8_autocast, fp8_state_dict
from astrai.model.components.linear import Linear
from tests.conftest import skip_no_fp8

FMT_A = torch.float8_e4m3fn
FMT_B = torch.float8_e5m2
HIST = 16


def _gemm():
    return get_module("gemm")


def _call(x, weight, slot=-1):
    return _gemm().fp8_linear(
        x, weight, None, True, False, False, HIST, 0, FMT_A, FMT_B, slot
    )


@pytest.fixture(autouse=True)
def _clean_fp8_state():
    gemm = _gemm()
    clear_slots()
    gemm.fp8_reset()
    gemm.fp8_debug_reset_stats()
    gemm.fp8_set_act_cache(True)
    yield
    gemm.fp8_reset()
    gemm.fp8_set_act_cache(True)
    clear_slots()


def _bf16(*shape, scale=1.0):
    return (
        torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * scale
    ).contiguous()


def _meta(weight, slot):
    return _gemm().fp8_debug_meta(weight, HIST, 0, slot)


@skip_no_fp8
def test_slot_rebind_keeps_the_rings_and_drops_the_stale_weight_cast():
    gemm = _gemm()
    gemm.fp8_set_slots([(0, "layers.0.q_proj", "layers.*.q_proj")])
    x = _bf16(8, 16)
    w = _bf16(16, 16)

    _call(x, w, slot=0)
    _call(x, w, slot=0)
    first = _meta(w, 0)
    assert (first["slot"], first["slot_name"]) == (0, "layers.0.q_proj")
    assert first["role"] == "layers.*.q_proj"
    assert gemm.fp8_debug_stats()["metas"] == 1
    idx_before = first["w"]["idx"]

    # The module is the same, its weight tensor is not (TP/FSDP swap). The
    # replacement stays in the old weight's magnitude band, as a real swap does
    # (the ring keeps the previous step's scale, so a wild change would clip
    # once — the usual delayed-scaling transient, not a rebind bug).
    w_new = _bf16(16, 16)
    out = _call(x, w_new, slot=0)

    assert gemm.fp8_debug_stats()["metas"] == 1, "rebind must not rebuild the meta"
    second = _meta(w_new, 0)
    assert second["w"]["initialized"], "the ring survived the replacement"
    assert second["w"]["idx"] == (idx_before + 1) % HIST, "history kept advancing"
    # The cast cache validates on the version counter, which a fresh parameter
    # shares with its predecessor: it must have been dropped, so the GEMM ran
    # on the NEW weight's bytes.
    ref = F.linear(x, w_new, None)
    assert (out - ref).norm() / ref.norm() < 0.1


@skip_no_fp8
def test_snapshot_binds_by_name_not_registration_order():
    gemm = _gemm()
    gemm.fp8_set_slots([(0, "mod.a", "mod.a"), (1, "mod.b", "mod.b")])
    x_small = _bf16(8, 16, scale=1.0)
    x_big = _bf16(8, 16, scale=100.0)
    w_a, w_b = _bf16(16, 16), _bf16(16, 16)
    _call(x_small, w_a, slot=0)
    _call(x_big, w_b, slot=1)

    saved = fp8_state_dict()
    assert saved["version"] == 2
    assert [e["slot_name"] for e in saved["entries"]] == ["mod.a", "mod.b"]
    by_name = {e["slot_name"]: e for e in saved["entries"]}
    assert not torch.equal(
        by_name["mod.a"]["x"]["state"], by_name["mod.b"]["x"]["state"]
    )

    # Fresh run: the ids name the *other* modules now, and the first call
    # registers in the opposite order. Registration-order binding would hand
    # mod.a's history to whichever module ran first.
    gemm.fp8_reset()
    gemm.fp8_set_slots([(0, "mod.b", "mod.b"), (1, "mod.a", "mod.a")])
    gemm.fp8_load_state_dict(saved)

    w_b = _bf16(16, 16)
    _call(x_big, w_b, slot=0)  # slot 0 is mod.b
    meta_b = _meta(w_b, 0)
    assert meta_b["slot_name"] == "mod.b"
    assert torch.equal(meta_b["x"]["state"].cpu(), by_name["mod.b"]["x"]["state"].cpu())
    assert not torch.equal(
        meta_b["x"]["state"].cpu(), by_name["mod.a"]["x"]["state"].cpu()
    )


@skip_no_fp8
def test_v1_snapshot_still_binds_by_shape_and_dtype():
    gemm = _gemm()
    gemm.fp8_set_slots([(0, "mod.a", "mod.a")])
    x = _bf16(8, 16)
    w = _bf16(16, 16)
    _call(x, w, slot=0)
    saved = fp8_state_dict()
    legacy = {
        "version": 1,
        "entries": [
            {k: v for k, v in entry.items() if k not in ("slot_name", "role")}
            for entry in saved["entries"]
        ],
    }
    assert all("slot_name" not in e for e in legacy["entries"])

    gemm.fp8_reset()
    clear_slots()  # no slot table at all: a pre-slot caller
    gemm.fp8_load_state_dict(legacy)

    w = _bf16(16, 16)
    meta = _meta(w, -1)
    assert meta["slot"] == -1
    assert meta["w"]["initialized"]
    assert torch.equal(
        meta["w"]["state"].cpu(), saved["entries"][0]["w"]["state"].cpu()
    )


@skip_no_fp8
def test_named_entry_never_leaks_into_an_unslotted_meta():
    gemm = _gemm()
    gemm.fp8_set_slots([(0, "mod.a", "mod.a")])
    x = _bf16(8, 16)
    w = _bf16(16, 16)
    _call(x, w, slot=0)
    saved = fp8_state_dict()

    # Same shape and dtype, but no slot: binding by shape here would give this
    # module another module's amax history — the cross-wiring names exist to
    # prevent.
    gemm.fp8_reset()
    clear_slots()
    gemm.fp8_load_state_dict(saved)
    fresh = _bf16(16, 16)
    meta = _meta(fresh, -1)
    assert not torch.equal(
        meta["w"]["state"].cpu(), saved["entries"][0]["w"]["state"].cpu()
    )


@skip_no_fp8
def test_dispatcher_names_metas_from_the_model():
    """End to end: assign_slots -> aten::linear -> named snapshot entries."""

    class TwoLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Linear(16, 16)
            self.b = Linear(16, 16)

    model = TwoLinear().to(device="cuda", dtype=torch.bfloat16)
    table = assign_slots(model)
    assert len(table) == 2

    with fp8_autocast(enabled=True):
        x = _bf16(8, 16)
        model.a(x)
        model.b(x)

    assert _meta(model.a.weight, -1)["slot_name"] == "a"
    assert _meta(model.b.weight, -1)["slot_name"] == "b"
    assert _meta(model.a.weight, -1)["slot"] == table.slot_of(model.a.weight)

    names = [e["slot_name"] for e in fp8_state_dict()["entries"]]
    assert names == ["a", "b"]


@skip_no_fp8
def test_region_entry_repairs_a_swapped_weight():
    """A parameter swap between regions is repaired on the next region entry."""

    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = Linear(16, 16)

    model = One().to(device="cuda", dtype=torch.bfloat16)
    table = assign_slots(model)
    x = _bf16(8, 16)
    with fp8_autocast(enabled=True):
        model.proj(x)
    slot_id = table.slot_of(model.proj.weight)
    assert slot_id >= 0

    model.proj.weight = nn.Parameter(_bf16(16, 16))
    assert table.slot_of(model.proj.weight) == -1  # stale until repaired

    with fp8_autocast(enabled=True):
        model.proj(x)

    assert table.slot_of(model.proj.weight) == slot_id
    meta = _meta(model.proj.weight, -1)
    assert (meta["slot"], meta["slot_name"]) == (slot_id, "proj")
    assert _gemm().fp8_debug_stats()["metas"] == 1


@skip_no_fp8
def test_unslotted_calls_keep_the_address_keyed_semantics():
    """The pre-slot contract, byte for byte: same weight, one meta, no names."""
    gemm = _gemm()
    x = _bf16(8, 16)
    w = _bf16(16, 16)

    first = _call(x, w)
    second = _call(x, w)

    assert torch.equal(first, second)
    meta = _meta(w, -1)
    assert (meta["slot"], meta["slot_name"], meta["role"]) == (-1, "", "")
    assert gemm.fp8_debug_stats()["metas"] == 1
    assert [e["slot_name"] for e in fp8_state_dict()["entries"]] == [""]
