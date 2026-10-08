"""Symmetric planning priority, metadata caching and CUDA execution contracts."""

import math
from types import SimpleNamespace

import pytest
import torch

from astrai.extension.backend import symmetric as backend
from astrai.extension.backend.symmetric import select, symm_out, syrk_out
from astrai.extension.kernel import symmetric as kernel
from astrai.extension.policy import symmetric as plan
from astrai.extension.runtime.dispatch import op_backend


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


def test_measured_rows_and_torch_veto_precede_geometry(planner):
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
    # Toggling policy does not re-query the unchanged native geometry.
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


def test_failed_configuration_keeps_rows_and_geometry_mode(planner):
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


def test_geometry_cache_tracks_batch_layout_addend_and_device(planner):
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


def test_callable_cache_follows_geometry_mode_and_runtime_flags(planner, monkeypatch):
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
def test_measured_dispatch_handle_remains_compatible_with_geometry(planner, operation):
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
    finally:
        kernel._plan.cache_clear()


CUDA_AVAILABLE = torch.cuda.is_available() and kernel.is_available()
CUDA_ONLY = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable"
)


def _cuda_layout(x, column):
    return (
        x.transpose(-2, -1).contiguous().transpose(-2, -1) if column else x.contiguous()
    )


def _cuda_operands(operation, rows, cols, batch, input_column, output_column):
    x = _cuda_layout(
        torch.randn(batch, rows, cols, device="cuda", dtype=torch.bfloat16),
        input_column,
    )
    x.div_(x.norm(dim=(-2, -1), keepdim=True))
    symmetric = torch.randn(batch, rows, rows, device=x.device, dtype=x.dtype)
    symmetric = (symmetric + symmetric.transpose(-2, -1)) / 2
    symmetric.div_(symmetric.norm(dim=(-2, -1), keepdim=True))
    shape = symmetric.shape if operation == "syrk" else x.shape
    output = _cuda_layout(
        torch.empty(shape, device=x.device, dtype=x.dtype), output_column
    )
    addend = torch.randn(shape, device=x.device, dtype=x.dtype)
    if operation == "syrk":
        addend = (addend + addend.transpose(-2, -1)) / 2
    addend = _cuda_layout(addend, not output_column)
    addend.div_(addend.norm(dim=(-2, -1), keepdim=True))
    return symmetric, x, output, addend


def _cuda_reference(operation, symmetric, x, addend, alpha, beta):
    return torch.stack(
        [
            torch.addmm(
                addend[index],
                x[index] if operation == "syrk" else symmetric[index],
                x[index].T if operation == "syrk" else x[index],
                alpha=alpha,
                beta=beta,
            )
            for index in range(x.size(0))
        ]
    )


def _assert_cuda_output(operation, output, expected):
    relative_l2 = torch.norm(output.float() - expected.float()) / torch.norm(
        expected.float()
    )
    assert relative_l2.item() < 0.01
    if operation == "syrk":
        assert torch.equal(output, output.transpose(-2, -1))


def _assert_native_geometry(operation, x, output, addend):
    rows, cols, batch = x.size(-2), x.size(-1), x.size(0)
    input_layout, output_layout = plan.layout(x), plan.layout(output)
    metadata = kernel.plan(
        operation,
        rows,
        cols,
        batch,
        input_layout,
        output_layout,
        addend,
        x.device.index,
    )
    assert metadata and metadata["candidates"]
    decision = plan.probe(operation, x, output=output, addend=addend)
    assert decision == plan.Plan("cuda", metadata["tile"], metadata["raster"])
    assert metadata["score"] == min(row["score"] for row in metadata["candidates"])
    vocabulary = {tile["name"]: tile for tile in kernel.tiles(operation)}
    device = torch.cuda.get_device_properties(x.device)
    tails = []
    for candidate in metadata["candidates"]:
        tile = vocabulary[candidate["tile"]]
        assert input_layout in tile["input_layouts"]
        assert math.isfinite(candidate["score"])
        assert candidate["resident_ctas"] > 0
        assert candidate["registers"] > 0
        assert candidate["local_bytes"] >= 0
        assert 0 < candidate["shared_memory"] <= device.shared_memory_per_block_optin
        if operation == "syrk":
            assert tile["block_m"] == tile["block_n"]
            tile_rows = math.ceil(rows / tile["block_m"])
            expected_blocks = batch * tile_rows * (tile_rows + 1) // 2
            tails.append(rows % tile["block_m"])
        else:
            # SYMM computes (S X).T = X.T S before restoring output orientation.
            expected_blocks = (
                batch
                * math.ceil(cols / tile["block_m"])
                * math.ceil(rows / tile["block_n"])
            )
        assert candidate["blocks"] == expected_blocks
    if operation == "syrk" and rows == 192:
        # The 128-wide recipes have a partial diagonal and mirrored edge tile.
        assert any(tails)
    return metadata


@CUDA_ONLY
@pytest.mark.parametrize("operation", ["syrk", "symm"])
@pytest.mark.parametrize(
    "rows,cols,batch", [(192, 320, 2), (384, 768, 3), (512, 64, 2)]
)
@pytest.mark.parametrize(
    "input_column,output_column",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_native_geometry_executes_independent_batches(
    operation, rows, cols, batch, input_column, output_column
):
    torch.manual_seed(113)
    symmetric, x, output, addend = _cuda_operands(
        operation, rows, cols, batch, input_column, output_column
    )
    with plan.override([], heuristic=True):
        for alpha, beta in [(1.0, 0.0), (-0.7, 0.3)]:
            _assert_native_geometry(operation, x, output, beta != 0)
            if operation == "syrk":
                syrk_out(x, output, addend=addend, alpha=alpha, beta=beta)
            else:
                symm_out(symmetric, x, output, addend=addend, alpha=alpha, beta=beta)
            _assert_cuda_output(
                operation,
                output,
                _cuda_reference(operation, symmetric, x, addend, alpha, beta),
            )


@CUDA_ONLY
@pytest.mark.parametrize(
    "operation,input_column,output_column",
    [("syrk", True, False), ("symm", False, True)],
)
def test_native_geometry_graph_replay_reads_updated_inputs_and_weights(
    operation, input_column, output_column
):
    torch.manual_seed(127)
    symmetric, x, output, addend = _cuda_operands(
        operation, 192, 320, 3, input_column, output_column
    )
    alpha, beta = -0.7, 0.3

    def run():
        if operation == "syrk":
            syrk_out(x, output, addend=addend, alpha=alpha, beta=beta)
        else:
            symm_out(symmetric, x, output, addend=addend, alpha=alpha, beta=beta)

    with plan.override([], heuristic=True):
        _assert_native_geometry(operation, x, output, True)
        run()
        _assert_cuda_output(
            operation,
            output,
            _cuda_reference(operation, symmetric, x, addend, alpha, beta),
        )
        previous = output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        x.copy_(torch.randn_like(x))
        x.div_(x.norm(dim=(-2, -1), keepdim=True))
        symmetric.mul_(0.5)
        addend.fill_(0.001)
        output.fill_(float("nan"))
        graph.replay()
        _assert_cuda_output(
            operation,
            output,
            _cuda_reference(operation, symmetric, x, addend, alpha, beta),
        )
        assert not torch.equal(output, previous)


@pytest.mark.skipif(
    not CUDA_AVAILABLE or torch.cuda.device_count() < 2,
    reason="two CUDA devices required",
)
def test_native_planner_restores_current_device():
    original = torch.cuda.current_device()
    other = (original + 1) % torch.cuda.device_count()
    kernel._plan.cache_clear()
    try:
        metadata = kernel.plan("syrk", 192, 320, 3, "column", "row", True, other)
        assert torch.cuda.current_device() == original
        if torch.cuda.get_device_capability(other)[0] >= 8:
            assert metadata and metadata["resident_ctas"] > 0
        else:
            assert metadata == {}
    finally:
        torch.cuda.set_device(original)
        kernel._plan.cache_clear()


@CUDA_ONLY
@pytest.mark.parametrize("operation", ["syrk", "symm"])
def test_native_planner_rejects_batched_square_overflow_without_tensors(operation):
    rows, cols, batch = 32768, 64, 3
    limit = (1 << 31) - 1
    assert rows * rows <= limit
    assert batch * rows * cols <= limit
    assert batch * rows * rows > limit
    # Both the SYRK output and the SYMM left operand require this square storage.
    # Query metadata only: allocating either operand would defeat this guard test.
    assert kernel.plan(operation, rows, cols, batch, device=0) == {}


@CUDA_ONLY
@pytest.mark.parametrize("ordinal", ["negative", "out_of_range"])
def test_native_planner_rejects_invalid_device_for_valid_geometry(ordinal):
    device = -1 if ordinal == "negative" else torch.cuda.device_count()
    with pytest.raises(RuntimeError, match="invalid CUDA device ordinal"):
        kernel.plan("syrk", 192, 320, 2, device=device)


@CUDA_ONLY
def test_native_planner_rejects_grid_height_overflow_without_tensors():
    # The input size is supported, but every compiled SYMM CTA would exceed
    # CUDA's grid.y limit after transposing the execution geometry.
    assert kernel.plan("symm", 64, 8388608, device=0) == {}
