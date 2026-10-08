"""Generic symmetric BLAS contracts, dispatch and every compiled tile."""

import pytest
import torch

from astrai.extension.backend import symmetric as backend
from astrai.extension.backend.symmetric import select, symm_out, syrk_out
from astrai.extension.kernel.symmetric import is_available, tiles
from astrai.extension.policy import symmetric as plan
from astrai.extension.runtime.dispatch import (
    ExplicitSelectionError,
    ImplRecord,
    Spec,
    op_backend,
    register_impl,
    set_op,
    unregister_impl,
)

CUDA_AVAILABLE = torch.cuda.is_available() and is_available()


@pytest.mark.parametrize("operation", ["syrk", "symm"])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (-0.3, 0.7), (0.0, -0.5)])
def test_arbitrary_cpu_geometry_and_independent_addend(operation, alpha, beta):
    x = torch.randn((7, 13), dtype=torch.float64)
    symmetric = torch.randn((7, 7), dtype=x.dtype)
    symmetric = (symmetric + symmetric.T) / 2
    output = torch.empty_like(symmetric if operation == "syrk" else x)
    addend = torch.randn_like(output)
    if operation == "syrk":
        addend = (addend + addend.T) / 2
        syrk_out(x, output, alpha=alpha, beta=beta, addend=addend)
        expected = alpha * (x @ x.T) + beta * addend
    else:
        symm_out(symmetric, x, output, alpha=alpha, beta=beta, addend=addend)
        expected = alpha * (symmetric @ x) + beta * addend
    torch.testing.assert_close(output, expected)


def test_invalid_contracts_and_strict_cuda_selection():
    x = torch.randn(7, 13)
    output = torch.empty(7, 7)
    with pytest.raises(ValueError, match="addend"):
        syrk_out(x, output, beta=1)
    with pytest.raises(ValueError, match="finite"):
        syrk_out(x, output, alpha=float("nan"))
    with pytest.raises(ValueError, match="alias"):
        symm_out(torch.eye(7), x, x)
    with pytest.raises(ExplicitSelectionError):
        syrk_out(x, output, backend="cuda")
    with op_backend(syrk="torch"):
        syrk_out(x, output)
    torch.testing.assert_close(output, x @ x.T)


def test_plan_configuration_is_atomic_and_scoped():
    before = plan.configure()
    with plan.override([]):
        assert plan.configure() == []
        with pytest.raises(ValueError, match="duplicate"):
            plan.configure(
                [
                    {
                        "operation": "symm",
                        "cc": 0,
                        "rows": 7,
                        "cols": 13,
                        "backend": "torch",
                    }
                ]
                * 2
            )
        assert plan.configure() == []
    assert plan.configure() == before


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable")
@pytest.mark.parametrize("operation", ["syrk", "symm"])
@pytest.mark.parametrize("input_layout", ["row", "column"])
@pytest.mark.parametrize("output_layout", ["row", "column"])
def test_every_tile_on_nonproduction_geometry_and_graph(
    operation, input_layout, output_layout
):
    torch.manual_seed(29)
    x = torch.randn(
        (192, 320) if input_layout == "row" else (320, 192),
        device="cuda",
        dtype=torch.bfloat16,
    )
    if input_layout == "column":
        x = x.T
    x.div_(x.norm())
    symmetric = torch.randn((192, 192), device="cuda", dtype=x.dtype)
    symmetric = (symmetric + symmetric.T) / 2
    symmetric.div_(symmetric.norm())
    shape = symmetric.shape if operation == "syrk" else x.shape
    output = torch.empty(
        shape if output_layout == "row" else shape[::-1], device=x.device, dtype=x.dtype
    )
    if output_layout == "column":
        output = output.T
    # The addend is independently laid out, exercising both stride dimensions.
    addend = torch.randn(
        shape[::-1] if output_layout == "row" else shape, device=x.device, dtype=x.dtype
    )
    if output_layout == "row":
        addend = addend.T
    if operation == "syrk":
        addend = (addend + addend.T) / 2
    addend.div_(addend.norm())
    alpha, beta = -0.7, 0.3
    expected = torch.addmm(
        addend,
        x if operation == "syrk" else symmetric,
        x.T if operation == "syrk" else x,
        alpha=alpha,
        beta=beta,
    )
    for tile in tiles(operation):
        if (
            input_layout not in tile["input_layouts"]
            or tile["shared_memory"]
            > torch.cuda.get_device_properties(x.device).shared_memory_per_block_optin
        ):
            continue
        options = dict(
            alpha=alpha, beta=beta, addend=addend, backend="cuda", tile=tile["name"]
        )

        def run():
            if operation == "syrk":
                syrk_out(x, output, **options)
            else:
                symm_out(symmetric, x, output, raster=-2, **options)

        run()
        torch.testing.assert_close(
            output, expected, atol=0.0001, rtol=0.02, msg=tile["name"]
        )
        if operation == "syrk":
            assert torch.equal(output, output.T)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        output.zero_()
        graph.replay()
        torch.testing.assert_close(
            output, expected, atol=0.0001, rtol=0.02, msg=tile["name"]
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable")
def test_injected_plan_handles_new_shape_and_respects_runtime_settings():
    x = torch.randn(192, 320, device="cuda", dtype=torch.bfloat16)
    cc = sum(a * b for a, b in zip(torch.cuda.get_device_capability(), (10, 1)))
    row = dict(
        operation="symm",
        cc=cc,
        rows=192,
        cols=320,
        backend="cuda",
        tile="128x64x32_W32x32_S2",
        raster=-2,
    )
    original = plan.probe("symm", x)
    with plan.override([row]):
        assert plan.probe("symm", x).tile == row["tile"]
        saved = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            assert plan.probe("symm", x).backend == "torch"
        finally:
            torch.use_deterministic_algorithms(saved)
    assert plan.probe("symm", x) == original


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable")
def test_unaligned_input_falls_back_and_explicit_cuda_rejects_it():
    storage = torch.randn(64 * 128 + 1, device="cuda", dtype=torch.bfloat16)
    x = storage[1:].view(64, 128)
    output = torch.empty(64, 64, device=x.device, dtype=x.dtype)
    assert not plan.supports(x)
    syrk_out(x, output)
    assert torch.equal(output, x @ x.T)
    with pytest.raises(ExplicitSelectionError):
        syrk_out(x, output, backend="cuda")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable"
            ),
        ),
    ],
)
def test_zero_beta_does_not_read_addend(device):
    x = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
    x.div_(x.norm())
    output = torch.empty(64, 64, device=device, dtype=x.dtype)
    addend = torch.full_like(output, float("nan"))
    syrk_out(
        x,
        output,
        addend=addend,
        beta=0,
        backend="cuda" if device == "cuda" else "torch",
    )
    torch.testing.assert_close(output, x @ x.T, atol=0.0001, rtol=0.02)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable")
def test_layout_plans_and_nondense_fallback():
    x = torch.randn(320, 192, device="cuda", dtype=torch.bfloat16).T
    output = torch.empty_like(x)
    cc = sum(a * b for a, b in zip(torch.cuda.get_device_capability(), (10, 1)))
    row = dict(
        operation="symm",
        cc=cc,
        rows=192,
        cols=320,
        input_layout="column",
        output_layout="column",
        backend="cuda",
        tile="64x64x32_W16x32_S2",
    )
    with plan.override([row]):
        assert plan.probe("symm", x, output=output).backend == "cuda"
        assert plan.probe("symm", x, output=output.contiguous()).backend == "torch"
        assert plan.probe("symm", x.contiguous(), output=output).backend == "torch"
    sliced = torch.randn(192, 640, device=x.device, dtype=x.dtype)[:, ::2]
    assert not plan.supports(sliced)
    gram = torch.empty(192, 192, device=x.device, dtype=x.dtype)
    syrk_out(sliced, gram)
    assert torch.equal(gram, sliced @ sliced.T)
    with pytest.raises(ExplicitSelectionError):
        syrk_out(sliced, gram, backend="cuda")
    with pytest.raises(ValueError, match="layout"):
        plan.configure([dict(row, operation="syrk", tile="wmma64")])


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable")
def test_cached_selection_tracks_plans_settings_and_overrides():
    x = torch.randn(192, 320, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(192, 192, device=x.device, dtype=x.dtype)
    original = select("syrk", x, output)
    assert select("syrk", x, output) is original
    cc = sum(a * b for a, b in zip(torch.cuda.get_device_capability(), (10, 1)))
    row = dict(
        operation="syrk",
        cc=cc,
        rows=192,
        cols=320,
        backend="cuda",
        tile="64x64x32_W16x32_S2",
    )
    with plan.override([row]):
        selected = select("syrk", x, output)
        assert selected is select("syrk", x, output)
        assert selected.func is backend.cuda.syrk_out
        with op_backend(syrk="torch"):
            assert select("syrk", x, output).func is backend._torch_syrk
        saved = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            assert select("syrk", x, output).func is backend._torch_syrk
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = saved
        set_op("syrk", "torch")
        try:
            assert select("syrk", x, output).func is backend._torch_syrk
        finally:
            set_op("syrk")
        assert select("syrk", x, output) is selected
    restored = select("syrk", x, output)
    assert restored.func is original.func
    assert restored.keywords == original.keywords


def test_cached_builtin_selection_bypasses_dynamic_external_implementations():
    x, output = torch.randn(7, 13), torch.empty(7, 7)
    select("syrk", x, output)
    available = [False]

    def dynamic(*args, **kwargs):
        pass

    record = ImplRecord(
        "syrk",
        "dynamic_test",
        dynamic,
        Spec.always(),
        priority=-1,
        available=lambda: available[0],
    )
    register_impl(record)
    try:
        assert select("syrk", x, output).func is backend._torch_syrk
        available[0] = True
        assert select("syrk", x, output).func is dynamic
        available[0] = False
        assert select("syrk", x, output).func is backend._torch_syrk
    finally:
        unregister_impl("syrk", "dynamic_test")
    assert select("syrk", x, output).func is backend._torch_syrk
