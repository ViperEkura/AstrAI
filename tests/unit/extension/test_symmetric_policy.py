"""Symmetric planning priority, metadata caching and CUDA execution contracts."""

import math
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from astrai.extension.backend.newton_schulz import select
from astrai.extension.kernel import newton_schulz as kernel
from astrai.extension.policy import newton_schulz as plan
from astrai.extension.runtime.dispatch import op_backend

backend = import_module("astrai.extension.backend.newton_schulz")


class MatrixMetadata:
    """CUDA-shaped metadata without allocating a tensor or opening a device."""

    dtype = torch.bfloat16
    is_cuda = True
    requires_grad = False

    def __init__(self, rows, cols, *, batch=None, column=False, device=0):
        self.shape = (rows, cols) if batch is None else (batch, rows, cols)
        self.ndim = len(self.shape)
        self.device = torch.device("cuda", device)
        strides = (1, rows) if column else (cols, 1)
        self._strides = strides if batch is None else (rows * cols, *strides)
        self._column = column

    def size(self, dimension):
        return self.shape[dimension]

    def stride(self, dimension=None):
        return self._strides if dimension is None else self._strides[dimension]

    def is_contiguous(self):
        return not self._column

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return 16


@pytest.mark.parametrize(
    "operation,addend,batch_size",
    [("syrk", False, 1), ("symm", True, 1), ("symm", True, 4)],
)
def test_default_policy_delegates_previous_measured_shapes_to_native_model(
    monkeypatch, operation, addend, batch_size
):
    assert plan.configure() == []
    calls = []

    def native(*query):
        calls.append(query)
        return {"tile": "model-tile", "raster": -2}

    monkeypatch.setattr(plan, "supports", lambda tensor: True)
    monkeypatch.setattr(plan, "_capability", lambda device: (12, 0))
    monkeypatch.setattr(plan, "heuristic_plan", native)
    plan._heuristic_decision.cache_clear()
    try:
        x = MatrixMetadata(256, 1536, batch=4 if batch_size == 4 else None)
        assert plan.probe(operation, x, addend=addend) == plan.Plan(
            "cuda", "model-tile", -2
        )
        assert calls == [(operation, 256, 1536, batch_size, "row", "row", addend, 0)]
    finally:
        plan._heuristic_decision.cache_clear()


@pytest.fixture
def planner(monkeypatch):
    requests = []

    def native(*query):
        requests.append(query)
        return {"tile": "64x64x64_W16x32_S2", "raster": -2}

    monkeypatch.setattr(plan, "heuristic_plan", native)
    monkeypatch.setattr(plan, "is_available", lambda: True)
    monkeypatch.setattr(plan, "_capability", lambda device: (12, 0))
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", True
    )
    monkeypatch.setattr(
        plan,
        "tiles",
        lambda operation: [
            {
                "name": "64x64x64_W16x32_S2",
                "input_layouts": ("row", "column"),
            }
        ],
    )
    plan._heuristic_decision.cache_clear()
    backend._selection_cache.clear()
    with plan.override([], heuristic=True):
        yield requests
    plan._heuristic_decision.cache_clear()
    backend._selection_cache.clear()


def test_measured_rows_and_torch_veto_precede_native_model(planner):
    x = MatrixMetadata(192, 320)
    row = {"operation": "syrk", "cc": 120, "rows": 192, "cols": 320, "backend": "torch"}
    plan.configure([row], heuristic=True)
    assert plan.probe("syrk", x) == plan.Plan()
    assert not planner
    row.update(backend="cuda", tile="64x64x64_W16x32_S2", raster=0)
    plan.configure([row], heuristic=True)
    assert plan.probe("syrk", x) == plan.Plan("cuda", row["tile"], 0)
    assert not planner
    assert plan.probe("syrk", MatrixMetadata(256, 320)).backend == "cuda"
    assert len(planner) == 1


def test_row_replacement_defaults_to_table_only_and_read_is_inert(planner):
    x = MatrixMetadata(192, 320)
    original = plan.probe("syrk", x)
    revision = plan.revision()
    assert plan.configure() == []
    assert plan.revision() == revision
    assert plan.probe("syrk", x) == original
    plan.configure([])
    assert plan.probe("syrk", x) == plan.Plan()
    plan.configure(heuristic=True)
    assert plan.probe("syrk", x) == original
    # Toggling policy does not re-query the unchanged native model.
    assert len(planner) == 1


def test_nested_override_restores_dispatch_mode_after_exception(planner):
    x = MatrixMetadata(192, 320)
    original = plan.probe("syrk", x)
    revision = plan.revision()
    with pytest.raises(RuntimeError, match="leave scope"), plan.override([]):
        assert plan.probe("syrk", x) == plan.Plan()
        with plan.override([], heuristic=True):
            assert plan.probe("syrk", x) == original
        assert plan.probe("syrk", x) == plan.Plan()
        raise RuntimeError("leave scope")
    assert plan.probe("syrk", x) == original
    assert plan.revision() > revision


def test_failed_configuration_keeps_rows_and_fallback_mode(planner):
    x = MatrixMetadata(192, 320)
    original = plan.probe("syrk", x)
    row = {"operation": "syrk", "cc": 120, "rows": 192, "cols": 320, "backend": "torch"}
    revision = plan.revision()
    with pytest.raises(ValueError, match="duplicate"):
        plan.configure([row, row], heuristic=False)
    with pytest.raises(ValueError, match="heuristic"):
        plan.configure([row], heuristic="enabled")
    assert plan.configure() == []
    assert plan.revision() == revision
    assert plan.probe("syrk", x) == original


def test_native_model_cache_tracks_batch_layout_addend_and_device(planner):
    for batch, input_column, output_column, addend, device, operation in [
        (None, False, False, False, 0, "syrk"),
        (1, False, False, False, 0, "syrk"),
        (4, False, False, False, 0, "syrk"),
        (4, True, False, False, 0, "syrk"),
        (4, True, True, False, 0, "syrk"),
        (4, True, True, True, 0, "syrk"),
        (4, True, True, True, 1, "syrk"),
        (4, True, True, True, 1, "symm"),
    ]:
        x = MatrixMetadata(192, 320, batch=batch, column=input_column, device=device)
        output = MatrixMetadata(
            192,
            192 if operation == "syrk" else 320,
            batch=batch,
            column=output_column,
            device=device,
        )
        assert plan.probe(operation, x, output=output, addend=addend).backend == "cuda"
        plan.probe(operation, x, output=output, addend=addend)
    assert len(planner) == 7
    assert planner[-1] == ("symm", 192, 320, 4, "column", "column", True, 1)
    # A requested future layout can differ from the currently stored view.
    x = MatrixMetadata(192, 320, column=True)
    plan.probe("syrk", x, input_layout="row")
    assert len(planner) == 7


def test_no_eligible_native_plan_and_unsupported_runtime_fall_back(
    planner, monkeypatch
):
    x = MatrixMetadata(192, 320)
    monkeypatch.setattr(plan, "heuristic_plan", lambda *query: {})
    assert plan.probe("syrk", x) == plan.Plan()
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    assert plan.probe("syrk", MatrixMetadata(256, 320)) == plan.Plan()
    assert not planner


def test_callable_cache_follows_fallback_mode_and_runtime_flags(planner, monkeypatch):
    x, output = MatrixMetadata(192, 320), MatrixMetadata(192, 192)
    original = select("syrk", x, output)
    assert original.func is kernel.syrk_out
    assert select("syrk", x, output) is original
    plan.configure(heuristic=False)
    assert select("syrk", x, output).func is backend._torch_syrk
    plan.configure(heuristic=True)
    assert select("syrk", x, output).func is kernel.syrk_out
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False
    )
    assert select("syrk", x, output).func is backend._torch_syrk
    assert len(planner) == 1


@pytest.mark.parametrize("operation", ["syrk", "symm"])
def test_measured_dispatch_handle_uses_native_model(planner, operation):
    x = MatrixMetadata(192, 320)
    output = MatrixMetadata(192, 192 if operation == "syrk" else 320)
    with op_backend(**{operation: "measured"}):
        selected = select(operation, x, output)
    expected = kernel.syrk_out if operation == "syrk" else kernel.symm_out
    assert selected.func is expected
    assert selected.keywords["tile"] == "64x64x64_W16x32_S2"
    assert len(planner) == 1


def test_native_adapter_caches_metadata_and_returns_independent_candidates(monkeypatch):
    requests = []

    def native(*query):
        requests.append(query)
        return {
            "tile": "64x64x64_W16x32_S2",
            "raster": 1,
            "score": 2.0,
            "candidates": [{"tile": "64x64x64_W16x32_S2", "score": 2.0}],
        }

    monkeypatch.setattr(kernel, "get_module", lambda name: SimpleNamespace(plan=native))
    kernel._plan.cache_clear()
    try:
        query = ("syrk", 192, 320, 4, "column", "row", True, 1)
        first = kernel.plan(*query)
        first["tile"] = "mutated"
        first["candidates"][0]["score"] = -1
        first["candidates"].append({})
        restored = kernel.plan(*query)
        assert restored["tile"] == "64x64x64_W16x32_S2"
        assert restored["candidates"] == [{"tile": "64x64x64_W16x32_S2", "score": 2.0}]
        assert requests == [query]
        kernel.plan(*query[:-1], device=0)
        assert len(requests) == 2
        kernel.plan(*query, mode="geometry")
        assert requests[-1] == (*query, "geometry")
        assert len(requests) == 3
    finally:
        kernel._plan.cache_clear()
