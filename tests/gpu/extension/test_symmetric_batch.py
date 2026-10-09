"""Batched symmetric BLAS isolation, layouts and Graph replay."""

import pytest
import torch

from astrai.extension.backend.newton_schulz import symm_out, syrk_out
from astrai.extension.kernel import newton_schulz as cuda
from astrai.extension.policy import newton_schulz as plan
from astrai.extension.runtime.dispatch import ExplicitSelectionError

CUDA_AVAILABLE = torch.cuda.is_available() and cuda.is_available()
CUDA_ONLY = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="symmetric CUDA extension unavailable"
)


def _layout(x, column):
    if column:
        return x.transpose(-2, -1).contiguous().transpose(-2, -1)
    return x.contiguous()


def _reference(operation, symmetric, x, addend, alpha, beta):
    results = []
    for index in range(x.size(0)):
        left = x[index] if operation == "syrk" else symmetric[index]
        right = x[index].T if operation == "syrk" else x[index]
        if beta:
            results.append(
                torch.addmm(addend[index], left, right, alpha=alpha, beta=beta)
            )
        else:
            results.append(alpha * (left @ right))
    return torch.stack(results)


@CUDA_ONLY
@pytest.mark.parametrize("operation", ["syrk", "symm"])
@pytest.mark.parametrize("input_column", [False, True])
@pytest.mark.parametrize("output_column", [False, True])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (-0.7, 0.3)])
def test_cuda_batch_layouts_and_independent_addend(
    operation, input_column, output_column, alpha, beta
):
    torch.manual_seed(73)
    x = _layout(
        torch.randn(3, 192, 320, device="cuda", dtype=torch.bfloat16),
        input_column,
    )
    x.div_(x.norm(dim=(-2, -1), keepdim=True))
    symmetric = torch.randn(3, 192, 192, device=x.device, dtype=x.dtype)
    symmetric = (symmetric + symmetric.transpose(-2, -1)) / 2
    symmetric.div_(symmetric.norm(dim=(-2, -1), keepdim=True))
    shape = symmetric.shape if operation == "syrk" else x.shape
    output = _layout(torch.empty(shape, device=x.device, dtype=x.dtype), output_column)
    addend = torch.randn(shape, device=x.device, dtype=x.dtype)
    if operation == "syrk":
        addend = (addend + addend.transpose(-2, -1)) / 2
    addend = _layout(addend, not output_column)
    addend.div_(addend.norm(dim=(-2, -1), keepdim=True))
    options = dict(
        backend="cuda",
        tile="64x64x64_W16x32_S2",
        addend=addend,
        alpha=alpha,
        beta=beta,
    )
    if operation == "syrk":
        syrk_out(x, output, **options)
        assert torch.equal(output, output.transpose(-2, -1))
    else:
        symm_out(symmetric, x, output, raster=-2, **options)
    torch.testing.assert_close(
        output,
        _reference(operation, symmetric, x, addend, alpha, beta),
        atol=0.0001,
        rtol=0.02,
    )


@CUDA_ONLY
@pytest.mark.parametrize(
    "operation,tile", [("syrk", "64x64x32_W16x32_S2"), ("symm", "64x64x32_W16x32_S2")]
)
def test_cuda_batch_results_do_not_mix_matrices(operation, tile):
    x = torch.zeros(3, 64, 128, device="cuda", dtype=torch.bfloat16)
    eye = torch.eye(64, device=x.device, dtype=x.dtype)
    x[1, :, :64] = eye * 0.5
    x[2, :, :64] = eye * -0.25
    symmetric = torch.stack((eye, eye * 2, eye * 3))
    output = torch.empty_like(symmetric if operation == "syrk" else x)
    if operation == "syrk":
        syrk_out(x, output, backend="cuda", tile=tile)
    else:
        symm_out(symmetric, x, output, backend="cuda", tile=tile)
    expected = torch.stack(
        [
            x[index] @ x[index].T
            if operation == "syrk"
            else symmetric[index] @ x[index]
            for index in range(3)
        ]
    )
    assert torch.equal(output, expected)
    assert not torch.equal(output[1], output[2])


@CUDA_ONLY
@pytest.mark.parametrize("operation", ["syrk", "symm"])
def test_cuda_batch_graph_replay_reads_changed_inputs(operation):
    torch.manual_seed(79)
    x = torch.randn(3, 64, 128, device="cuda", dtype=torch.bfloat16)
    x.div_(x.norm(dim=(-2, -1), keepdim=True))
    symmetric = torch.eye(64, device=x.device, dtype=x.dtype).repeat(3, 1, 1)
    shape = symmetric.shape if operation == "syrk" else x.shape
    addend = torch.ones(shape, device=x.device, dtype=x.dtype) * 0.001
    output = torch.empty_like(addend)

    def run():
        options = dict(
            backend="cuda",
            tile="64x64x64_W16x32_S2",
            addend=addend,
            alpha=-0.7,
            beta=0.3,
        )
        if operation == "syrk":
            syrk_out(x, output, **options)
        else:
            symm_out(symmetric, x, output, **options)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.copy_(torch.randn_like(x))
    x.div_(x.norm(dim=(-2, -1), keepdim=True))
    symmetric.mul_(0.25)
    addend.fill_(0.002)
    output.zero_()
    graph.replay()
    torch.testing.assert_close(
        output,
        _reference(operation, symmetric, x, addend, -0.7, 0.3),
        atol=0.0001,
        rtol=0.02,
    )


@CUDA_ONLY
@pytest.mark.parametrize("invalid", ["empty", "slice", "batch_slice", "expanded"])
def test_unsupported_cuda_batch_falls_back_without_reading_wrong_strides(invalid):
    if invalid == "empty":
        x = torch.empty(0, 64, 128, device="cuda", dtype=torch.bfloat16)
    elif invalid == "slice":
        x = torch.randn(3, 64, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    elif invalid == "batch_slice":
        x = torch.randn(6, 64, 128, device="cuda", dtype=torch.bfloat16)[::2]
    else:
        x = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16).expand(
            3, -1, -1
        )
    output = torch.empty(x.size(0), 64, 64, device=x.device, dtype=x.dtype)
    assert not plan.supports(x)
    syrk_out(x, output)
    expected = torch.bmm(x, x.transpose(-2, -1))
    assert torch.equal(output, expected)
    with pytest.raises(ExplicitSelectionError):
        syrk_out(x, output, backend="cuda", tile="64x64x64_W16x32_S2")
    with pytest.raises(RuntimeError, match="dense"):
        cuda.syrk_out(x, output, tile="64x64x64_W16x32_S2")


@CUDA_ONLY
def test_measured_batch_plan_does_not_apply_to_other_batch_sizes():
    x = torch.randn(4, 64, 128, device="cuda", dtype=torch.bfloat16)
    major, minor = torch.cuda.get_device_capability()
    row = dict(
        operation="syrk",
        cc=major * 10 + minor,
        rows=64,
        cols=128,
        batch_size=4,
        backend="cuda",
        tile="64x64x64_W16x32_S2",
    )
    with plan.override([row]):
        assert plan.probe("syrk", x).backend == "cuda"
        assert plan.probe("syrk", x[:3]).backend == "torch"
        assert plan.probe("syrk", x[:1]).backend == "torch"
        assert plan.probe("syrk", x[0]).backend == "torch"
