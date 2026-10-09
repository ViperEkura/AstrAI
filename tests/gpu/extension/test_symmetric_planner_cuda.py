"""CUDA execution and metadata contracts for the native NS planner."""

import math

import pytest
import torch

from astrai.extension.backend.newton_schulz import symm_out, syrk_out
from astrai.extension.kernel import newton_schulz as kernel
from astrai.extension.policy import newton_schulz as plan

CUDA_AVAILABLE = torch.cuda.is_available() and kernel.is_available()
pytestmark = pytest.mark.skipif(
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


def _assert_native_model(operation, x, output, addend):
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
    assert metadata["source"] == "model"
    previous = kernel.plan(
        operation,
        rows,
        cols,
        batch,
        input_layout,
        output_layout,
        addend,
        x.device.index,
        mode="geometry",
    )
    assert previous["source"] == "geometry"
    assert previous["score"] == previous["geometry_score"]
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
        assert candidate["k_steps"] > 0
        assert candidate["waves"] > 0
        assert candidate["load_bytes"] > 0
        assert candidate["mma_instructions"] > 0
        assert candidate["epilogue_bytes"] > 0
        assert candidate["local_traffic_bytes"] == (
            2 * candidate["local_bytes"] * tile["threads"]
        )
        assert candidate["model_score"] == pytest.approx(
            candidate["waves"]
            * (
                candidate["load_bytes"]
                + candidate["epilogue_bytes"]
                + candidate["local_traffic_bytes"]
            )
        )
        assert math.isfinite(candidate["geometry_score"])
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


@pytest.mark.parametrize("operation", ["syrk", "symm"])
@pytest.mark.parametrize(
    "rows,cols,batch", [(192, 320, 2), (384, 768, 3), (512, 64, 2)]
)
@pytest.mark.parametrize(
    "input_column,output_column",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_native_model_executes_independent_batches(
    operation, rows, cols, batch, input_column, output_column
):
    torch.manual_seed(113)
    symmetric, x, output, addend = _cuda_operands(
        operation, rows, cols, batch, input_column, output_column
    )
    with plan.override([], heuristic=True):
        for alpha, beta in [(1.0, 0.0), (-0.7, 0.3)]:
            _assert_native_model(operation, x, output, beta != 0)
            if operation == "syrk":
                syrk_out(x, output, addend=addend, alpha=alpha, beta=beta)
            else:
                symm_out(symmetric, x, output, addend=addend, alpha=alpha, beta=beta)
            _assert_cuda_output(
                operation,
                output,
                _cuda_reference(operation, symmetric, x, addend, alpha, beta),
            )


@pytest.mark.parametrize(
    "operation,input_column,output_column",
    [("syrk", True, False), ("symm", False, True)],
)
def test_native_model_graph_replay_reads_updated_inputs_and_weights(
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
        _assert_native_model(operation, x, output, True)
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


@pytest.mark.parametrize("ordinal", ["negative", "out_of_range"])
def test_native_planner_rejects_invalid_device_for_valid_geometry(ordinal):
    device = -1 if ordinal == "negative" else torch.cuda.device_count()
    with pytest.raises(RuntimeError, match="invalid CUDA device ordinal"):
        kernel.plan("syrk", 192, 320, 2, device=device)


def test_native_planner_rejects_grid_height_overflow_without_tensors():
    # The input size is supported, but every compiled SYMM CTA would exceed
    # CUDA's grid.y limit after transposing the execution geometry.
    assert kernel.plan("symm", 64, 8388608, device=0) == {}
