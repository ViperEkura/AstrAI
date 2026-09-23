"""FP8 lifecycle: ring checkpointing, recompute safety, recipe invalidation.

The TE-pattern adaptations (2026-09-20, see
notes/fp8-framework-survey-2026-09-20.md):

- A1 checkpointing — the delayed-scaling rings (scale / amax history / index)
  survive save/resume, the way TE persists fp8_meta with the checkpoint.
  Without this a resumed run re-seeds every scale and its first step is
  numerically disjoint from the checkpointed one.
- A2 recompute safety — a no-grad forward reads the rings without folding or
  advancing, so gradient-checkpointing's recomputed forward quantizes with
  exactly the scale the original pass used (TE clones fp8 meta "to ensure
  both forward steps are numerically same").
- A3 recipe invalidation — a recipe change under the same weight rebuilds the
  rings (TE's recipe-change workspace clear); stale geometry or fold
  constants would silently corrupt the fold.

The ring registry lives in the C++ op (``gemm.fp8_linear``'s translation
unit); tests observe it through the debug hooks (``fp8_debug_meta``) and the
snapshot pair. ``fp8_debug_meta(w, history_len, margin)`` follows the same
registry rules as a forward — including consuming pending snapshot entries —
so tests can bind restores explicitly the way a real resume would.

Isolation discipline: the ring registry is a process-wide singleton keyed by
(data_ptr, shape, dtype) — freed allocations are recycled, so every test must
reset() before AND after, or the next test's "fresh" linear inherits a stale
ring and quantizes with a scale from someone else's amax.
"""

import gc

import pytest
import torch
from torch import nn

import astrai.extension.quantize as f8mod
from astrai.extension.loader import get_module
from astrai.extension.quantize import (
    FP8Recipe,
    fp8_autocast,
    fp8_load_state_dict,
    fp8_state_dict,
)
from tests.conftest import skip_no_fp8


@pytest.fixture(autouse=True)
def _clean_fp8_state():
    f8mod.fp8_reset()
    yield
    f8mod.fp8_reset()


def _linear(seed, n=32, k=64):
    torch.manual_seed(seed)
    return nn.Linear(k, n, device="cuda", dtype=torch.bfloat16)


def _step(lin, x):
    with fp8_autocast(enabled=True):
        return lin(x)


def _meta(w, history_len=16, margin=0):
    """The weight's registry entry as plain dicts/ints (debug hook)."""
    return get_module("gemm").fp8_debug_meta(w, history_len, margin)


@skip_no_fp8
def test_ring_snapshot_resume_continuity():
    """Resume with a restored ring produces the same next scale — and the
    same next output — as an uninterrupted run (A1)."""
    dev = torch.device("cuda")
    xs = [torch.randn(8, 64, device=dev, dtype=torch.bfloat16) for _ in range(4)]

    lin = _linear(0)
    for x in xs[:3]:
        _step(lin, x)
    saved = fp8_state_dict()
    assert saved["entries"], "delayed scaling must have allocated ring metas"

    # Uninterrupted: step 4 (and the published next scale it leaves behind).
    _step(lin, xs[3])
    meta = _meta(lin.weight)
    ref_scale = meta["x"]["scale"].clone()
    ref_hist = meta["x"]["hist"].clone()
    ref_idx = meta["x"]["idx"]

    # Restart: fresh state, identical weights, snapshot restored before the
    # first forward (the real resume order — the registry is empty then).
    f8mod.fp8_reset()
    lin2 = _linear(0)
    lin2.load_state_dict(lin.state_dict())
    fp8_load_state_dict(saved)
    _step(lin2, xs[3])
    meta2 = _meta(lin2.weight)

    assert meta2["x"]["idx"] == ref_idx
    torch.testing.assert_close(meta2["x"]["scale"], ref_scale, rtol=0, atol=0)
    torch.testing.assert_close(meta2["x"]["hist"], ref_hist, rtol=0, atol=0)
    torch.testing.assert_close(meta2["g"]["scale"], meta["g"]["scale"], rtol=0, atol=0)


@skip_no_fp8
def test_ring_restore_binds_fifo_by_shape_and_dtype():
    """FIFO re-binding matches entries by (shape, dtype) in creation order:
    two same-shape linears with distinct amax must each get their own ring
    back — a backwards match would hand linear a the wrong scale."""
    dev = torch.device("cuda")
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    lin_a, lin_b = _linear(1), _linear(2)  # same (shape, dtype), distinct amax
    _step(lin_a, x)
    _step(lin_b, x * 3)
    saved = fp8_state_dict()
    meta_a = _meta(lin_a.weight)
    meta_b = _meta(lin_b.weight)
    scale_a, hist_a = meta_a["x"]["scale"].clone(), meta_a["x"]["hist"].clone()
    scale_b, hist_b = meta_b["x"]["scale"].clone(), meta_b["x"]["hist"].clone()
    assert not torch.equal(scale_a, scale_b), "test needs distinct amax"

    f8mod.fp8_reset()
    lin_a2, lin_b2 = _linear(1), _linear(2)
    fp8_load_state_dict(saved)
    # Bind both explicitly, before any forward: the registry is empty on a
    # real resume, so entries are consumed from the pending queue in order.
    meta_a2 = _meta(lin_a2.weight)
    meta_b2 = _meta(lin_b2.weight)
    torch.testing.assert_close(meta_a2["x"]["scale"], scale_a, rtol=0, atol=0)
    torch.testing.assert_close(meta_a2["x"]["hist"], hist_a, rtol=0, atol=0)
    torch.testing.assert_close(meta_b2["x"]["scale"], scale_b, rtol=0, atol=0)
    torch.testing.assert_close(meta_b2["x"]["hist"], hist_b, rtol=0, atol=0)


@skip_no_fp8
def test_nograd_forward_leaves_ring_untouched():
    """A no-grad linear (checkpointing first pass / inference) must not fold
    or advance the ring — before this test the idx advanced on every call
    and inference silently consumed the delayed window (A2)."""
    dev = torch.device("cuda")
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    lin = _linear(3)
    with fp8_autocast(enabled=True):
        lin(x)  # grad pass: seed + fold + advance
    meta = _meta(lin.weight)
    idx = meta["x"]["idx"]
    scale, hist = meta["x"]["scale"].clone(), meta["x"]["hist"].clone()

    with fp8_autocast(enabled=True), torch.no_grad():
        lin(x)
        lin(x)
    meta = _meta(lin.weight)
    assert meta["x"]["idx"] == idx
    torch.testing.assert_close(meta["x"]["scale"], scale, rtol=0, atol=0)
    torch.testing.assert_close(meta["x"]["hist"], hist, rtol=0, atol=0)


@skip_no_fp8
def test_recompute_output_identity():
    """The checkpointing invariant, end to end: no-grad pass + grad pass
    must produce bitwise-equal outputs to a grad pass alone (the recomputed
    forward quantizes with the original scale)."""
    dev = torch.device("cuda")
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    lin = _linear(4)

    with fp8_autocast(enabled=True):
        ref = lin(x)

    f8mod.fp8_reset()
    lin2 = _linear(4)
    lin2.load_state_dict(lin.state_dict())
    with fp8_autocast(enabled=True):
        with torch.no_grad():
            lin2(x)  # checkpoint first pass
        out = lin2(x)  # recompute (grad-enabled)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@skip_no_fp8
def test_first_forward_nograd_then_grad_matches_grad_only():
    """An eval pass before training must not shift the step-1 semantics: the
    no-grad pass seeds host-side, the first grad pass folds normally."""
    dev = torch.device("cuda")
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    lin = _linear(5)

    with fp8_autocast(enabled=True):
        with torch.no_grad():
            lin(x)
        out = lin(x)
    assert _meta(lin.weight)["x"]["idx"] == 1  # one grad pass, not two calls

    f8mod.fp8_reset()
    lin2 = _linear(5)
    lin2.load_state_dict(lin.state_dict())
    with fp8_autocast(enabled=True):
        ref = lin2(x)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@skip_no_fp8
def test_recipe_change_rebuilds_rings():
    """A recipe change under the same weight rebuilds the rings (A3): stale
    history_len would break the state-buffer geometry, stale margin the
    fold constants."""
    dev = torch.device("cuda")
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    lin = _linear(6)
    from astrai.extension.ops.quantize import K_FOLD_SLOTS

    with fp8_autocast(enabled=True, recipe=FP8Recipe(history_len=4)):
        lin(x)
    meta = _meta(lin.weight, history_len=4)
    assert meta["x"]["state"].numel() == 4 + 6 + K_FOLD_SLOTS
    assert meta["x"]["idx"] == 1

    with fp8_autocast(enabled=True, recipe=FP8Recipe(history_len=8)):
        lin(x)
    meta = _meta(lin.weight, history_len=8)
    assert meta["x"]["state"].numel() == 8 + 6 + K_FOLD_SLOTS
    # Rebuilt fresh inside the forward above: seeded, folded, advanced once.
    assert meta["x"]["idx"] == 1 and meta["x"]["initialized"]

    # Same recipe twice more must NOT rebuild (the ordinary flow). The
    # debug hook returns a snapshot, so re-query after the two forwards.
    with fp8_autocast(enabled=True, recipe=FP8Recipe(history_len=8)):
        lin(x)
        lin(x)
    assert _meta(lin.weight, history_len=8)["x"]["idx"] == 3


@skip_no_fp8
def test_registry_evicts_a_dead_weights_ring():
    """A weight that goes away takes its registry entry with it: an orphan can
    never be restored (the entry is keyed by its buffer), and left in the
    registration order it would shift the FIFO snapshot binding of every entry
    after it. The sweep is lazy — at save time — because an orphan is already
    inert (its anchor is dead, so no lookup can match it) and scanning the
    registry on the lookup path would cost a pass per linear."""
    dev = torch.device("cuda")
    gemm = get_module("gemm")
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)

    # A helper, not a loop over the two weights: `for w in (...)` leaves the
    # loop variable bound to the last tensor, keeping `doomed` alive past its
    # `del` and making this test pass for the wrong reason.
    def run_once(weight):
        gemm.fp8_linear(
            x,
            weight,
            None,
            False,
            False,
            False,
            16,
            0,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
        )

    keep = torch.randn(32, 64, device=dev, dtype=torch.bfloat16)
    doomed = torch.randn(96, 64, device=dev, dtype=torch.bfloat16)
    run_once(keep)
    run_once(doomed)
    assert gemm.fp8_debug_stats()["metas"] == 2
    assert len(fp8_state_dict()["entries"]) == 2

    del doomed
    gc.collect()
    torch.cuda.empty_cache()
    # The next snapshot prunes it, and reports only what a resume can bind.
    assert len(fp8_state_dict()["entries"]) == 1
    assert gemm.fp8_debug_stats()["metas"] == 1


@skip_no_fp8
def test_state_dict_geometry_mismatch_stays_fresh():
    """A snapshot saved under history_len=4 must not be forced into a
    history_len=8 ring: the load declines and the ring re-seeds."""
    dev = torch.device("cuda")
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    lin = _linear(7)
    with fp8_autocast(enabled=True, recipe=FP8Recipe(history_len=4)):
        lin(x)
    saved = fp8_state_dict()

    f8mod.fp8_reset()
    lin2 = _linear(7)
    fp8_load_state_dict(saved)
    with fp8_autocast(enabled=True, recipe=FP8Recipe(history_len=8)):
        lin2(x)  # pending entry declined (geometry), ring seeds fresh
    meta = _meta(lin2.weight, history_len=8)
    assert meta["x"]["idx"] == 1
    assert meta["x"]["initialized"]
