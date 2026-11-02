"""Explicit quantize outputs, fused scale publication and boundary checks."""

import itertools

import pytest
import torch

from astrai.extension.kernel.quantize import quantize, quantize_scaled
from astrai.extension.runtime.loader import get_module
from tests.support.capabilities import skip_no_fp8, skip_no_kernel

F4 = torch.float8_e4m3fn
F5 = torch.float8_e5m2
I8 = torch.int8
OUTPUTS = [(dtype, None) for dtype in (F4, F5, I8)]
OUTPUTS += [(None, dtype) for dtype in (F4, F5, I8)]
OUTPUTS += list(itertools.product((F4, F5, I8), repeat=2))


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("case", ["history", "margin"])
def test_scaled_validation_precedes_input_copy(case):
    x = torch.ones(17, 33, device="cuda").t()
    inv_scale = torch.ones(1, device="cuda")
    ring = _ring()
    kwargs = dict(hist_len=4, hist_idx=0, dtype=I8)
    if case == "history":
        kwargs["hist_idx"] = 4
    else:
        kwargs["margin"] = 1024
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        with pytest.raises(RuntimeError, match="hist_idx|margin"):
            quantize_scaled(x, inv_scale, ring, **kwargs)
    assert not {event.key for event in prof.key_averages()} & {
        "aten::contiguous",
        "aten::clone",
        "aten::copy_",
    }


def _cast(x, dtype):
    if dtype == I8:
        return torch.nan_to_num(x, nan=0.0).round().clamp(-127, 127).to(I8)
    return x.to(dtype)


def _ring(n=4):
    return torch.zeros(n + 4 + get_module("quantize").K_FOLD_SLOTS + 2, device="cuda")


@pytest.mark.cuda
@skip_no_fp8
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("dtype,tdtype", OUTPUTS)
@pytest.mark.parametrize("shape", [(37, 67), (2, 5, 65)])
def test_requested_outputs_match_cast(input_dtype, dtype, tdtype, shape):
    torch.manual_seed(17)
    # Include a noncontiguous last axis and flattenable leading dimensions.
    x = torch.randn(*shape, device="cuda", dtype=input_dtype).transpose(-1, -2)
    inv_scale = torch.tensor([0.25], device="cuda")
    out, out_t, amax = quantize(x, inv_scale, dtype=dtype, tdtype=tdtype)
    ref = x.float() * inv_scale
    assert amax is None
    if dtype is None:
        assert out is None
    else:
        assert out.shape == x.shape
        assert torch.equal(out.view(torch.uint8), _cast(ref, dtype).view(torch.uint8))
    if tdtype is None:
        assert out_t is None
    else:
        expected = _cast(ref.reshape(-1, x.shape[-1]).t().contiguous(), tdtype)
        assert torch.equal(out_t.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.cuda
@skip_no_fp8
@pytest.mark.parametrize("external", [False, True])
def test_publish_override_and_next_round_multiplier(external):
    ring = _ring()
    ring[:4].fill_(1)
    inv_scale = torch.tensor([2.0], device="cuda")
    next_scale = torch.zeros(1, device="cuda") if external else ring[-2:-1]
    next_inv_scale = torch.zeros(1, device="cuda") if external else ring[-1:]
    x = torch.full((37, 67), 112.0, device="cuda", dtype=torch.bfloat16)
    out, out_t, amax = quantize_scaled(
        x,
        inv_scale,
        ring,
        hist_len=4,
        hist_idx=2,
        dtype=F4,
        tdtype=F5,
        next_scale=next_scale,
        next_inv_scale=next_inv_scale,
    )
    assert amax.data_ptr() == ring[6:7].data_ptr()
    assert float(amax) == 112
    assert float(next_scale) == 0.25
    assert float(next_inv_scale) == 4
    # Publication must leave pair zero and the caller's current multiplier intact.
    assert float(ring[4]) == 0 and float(ring[5]) == 0
    assert float(inv_scale) == 2
    if external:
        assert torch.equal(ring[-2:], torch.zeros(2, device="cuda"))
    torch.testing.assert_close(
        out.float(), torch.full_like(x.float(), 224), rtol=0, atol=0
    )
    assert out_t.shape == (67, 37)
    next_out, _, _ = quantize(x, next_inv_scale, dtype=F4)
    torch.testing.assert_close(
        next_out.float(), torch.full_like(x.float(), 448), rtol=0, atol=0
    )


@pytest.mark.cuda
@skip_no_fp8
def test_transpose_only_default_range_and_margin():
    ring = _ring()
    x = torch.full((2, 3, 7), 112.0, device="cuda", dtype=torch.float16)
    out, out_t, amax = quantize_scaled(
        x,
        torch.ones(1, device="cuda"),
        ring,
        hist_len=4,
        hist_idx=1,
        tdtype=F5,
        margin=1,
    )
    assert out is None and out_t.shape == (7, 6)
    assert float(amax) == 112
    expected = torch.tensor([112.0 / 57344 / 2], device="cuda")
    torch.testing.assert_close(ring[4:5], expected, rtol=0, atol=0)
    torch.testing.assert_close(ring[5:6], expected.reciprocal(), rtol=0, atol=0)


@pytest.mark.cuda
@skip_no_fp8
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "dtype or tdtype"),
        ({"dtype": torch.float32}, "output dtype"),
        ({"tdtype": torch.float64}, "output dtype"),
    ],
)
def test_invalid_output_requests(kwargs, match):
    x = torch.ones(2, 3, device="cuda")
    with pytest.raises(RuntimeError, match=match):
        quantize(x, torch.ones(1, device="cuda"), **kwargs)


@pytest.mark.cuda
@skip_no_fp8
@pytest.mark.parametrize(
    "case,match",
    [
        ("input_dtype", "no kernel"),
        ("scalar_input", "1D"),
        ("transpose_1d", "2D"),
        ("inv_scale_dtype", "inv_scale"),
        ("inv_scale_size", "inv_scale"),
        ("inv_scale_cpu", "inv_scale"),
        ("ring_strided", "contiguous"),
        ("ring_cpu", "ring_state"),
        ("ring_small", "too small"),
        ("hist_zero", "hist_len"),
        ("hist_large", "hist_len"),
        ("hist_wraparound", "hist_len"),
        ("idx_negative", "hist_idx"),
        ("idx_large", "hist_idx"),
        ("idx_wraparound", "hist_idx"),
        ("next_scale_only", "provided together"),
        ("next_inv_scale_only", "provided together"),
        ("next_cpu", "next_scale"),
        ("max_zero", "quant_max"),
        ("max_nan", "quant_max"),
        ("margin_large", "margin"),
    ],
)
def test_invalid_scaled_parameters(case, match):
    x = torch.ones(2, 3, device="cuda")
    inv_scale = torch.ones(1, device="cuda")
    ring = _ring()
    args = dict(hist_len=4, hist_idx=0, dtype=F4)
    if case == "input_dtype":
        x = x.double()
    elif case == "scalar_input":
        x = x[0, 0]
    elif case == "transpose_1d":
        x = x[0]
        args = dict(hist_len=4, hist_idx=0, tdtype=F4)
    elif case == "inv_scale_dtype":
        inv_scale = inv_scale.double()
    elif case == "inv_scale_size":
        inv_scale = torch.ones(2, device="cuda")
    elif case == "inv_scale_cpu":
        inv_scale = inv_scale.cpu()
    elif case == "ring_strided":
        ring = torch.zeros(ring.numel() * 2, device="cuda")[::2]
    elif case == "ring_cpu":
        ring = ring.cpu()
    elif case == "ring_small":
        ring = ring[:4]
    elif case == "hist_zero":
        args["hist_len"] = 0
    elif case == "hist_large":
        args["hist_len"] = 2**31
    elif case == "hist_wraparound":
        args["hist_len"] = 2**32 + 4
    elif case == "idx_negative":
        args["hist_idx"] = -1
    elif case == "idx_large":
        args["hist_idx"] = 4
    elif case == "idx_wraparound":
        args["hist_idx"] = 2**32
    elif case == "next_scale_only":
        args["next_scale"] = torch.ones(1, device="cuda")
    elif case == "next_inv_scale_only":
        args["next_inv_scale"] = torch.ones(1, device="cuda")
    elif case == "next_cpu":
        args.update(next_scale=torch.ones(1), next_inv_scale=torch.ones(1))
    elif case == "max_zero":
        args["quant_max"] = 0
    elif case == "max_nan":
        args["quant_max"] = float("nan")
    elif case == "margin_large":
        args["margin"] = 1024
    with pytest.raises(RuntimeError, match=match):
        quantize_scaled(x, inv_scale, ring, **args)


@pytest.mark.cuda
@skip_no_fp8
def test_other_device_parameters_rejected():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    x = torch.ones(2, 3, device="cuda:0")
    inv_scale = torch.ones(1, device="cuda:0")
    with pytest.raises(RuntimeError, match="ring_state"):
        quantize_scaled(
            x, inv_scale, _ring().to("cuda:1"), hist_len=4, hist_idx=0, dtype=F4
        )
    with pytest.raises(RuntimeError, match="next_scale"):
        quantize_scaled(
            x,
            inv_scale,
            _ring(),
            hist_len=4,
            hist_idx=0,
            dtype=F4,
            next_scale=torch.ones(1, device="cuda:1"),
            next_inv_scale=torch.ones(1, device="cuda:1"),
        )


@pytest.mark.cuda
@skip_no_fp8
def test_nondefault_stream_and_graph_replay():
    stream = torch.cuda.Stream()
    x = torch.ones(64, 128, device="cuda", dtype=torch.bfloat16)
    inv_scale = torch.ones(1, device="cuda")
    ring = _ring()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            quantize_scaled(
                x, inv_scale, ring, hist_len=4, hist_idx=0, dtype=F4, tdtype=F5
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out, out_t, amax = quantize_scaled(
                x,
                inv_scale,
                ring,
                hist_len=4,
                hist_idx=0,
                dtype=F4,
                tdtype=F5,
            )
        x.fill_(2)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert float(amax) == 2
    assert torch.equal(out.float(), x.float())
    assert torch.equal(out_t.float(), x.t().float())


@pytest.mark.cuda
@skip_no_fp8
@pytest.mark.parametrize("hist_len", [1, 65])
@pytest.mark.parametrize("pairs", [1, 2])
def test_ring_offsets_and_capacity_boundaries(hist_len, pairs):
    scratch_slots = get_module("quantize").K_FOLD_SLOTS
    size = hist_len + 4 + scratch_slots + 2 * (pairs - 1)
    storage = torch.full((size + 2,), -17.0, device="cuda")
    ring = storage[1:-1]
    ring.zero_()
    ring[:hist_len].fill_(7)
    inv_scale = torch.ones(1, device="cuda")
    x = torch.full((37, 67), 14.0, device="cuda")
    _, _, amax = quantize_scaled(
        x,
        inv_scale,
        ring,
        hist_len=hist_len,
        hist_idx=hist_len - 1,
        dtype=F4,
    )
    assert amax.data_ptr() == ring[hist_len + 2 : hist_len + 3].data_ptr()
    assert float(amax) == 14
    expected_history = torch.full((hist_len,), 7.0, device="cuda")
    expected_history[-1] = 14
    torch.testing.assert_close(ring[:hist_len], expected_history, rtol=0, atol=0)
    torch.testing.assert_close(
        ring[hist_len : hist_len + 1],
        torch.tensor([14.0 / 448], device="cuda"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ring[hist_len + 1 : hist_len + 2],
        ring[hist_len : hist_len + 1].reciprocal(),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(ring[hist_len + 3 :]) == 0
    assert float(storage[0]) == float(storage[-1]) == -17
    # One float short must fail before the kernel can write outside this view.
    with pytest.raises(RuntimeError, match="too small"):
        quantize_scaled(
            x,
            inv_scale,
            ring[: hist_len + 3 + scratch_slots],
            hist_len=hist_len,
            hist_idx=hist_len - 1,
            dtype=F4,
        )


@pytest.mark.cuda
@skip_no_fp8
@pytest.mark.parametrize(
    "shape,match",
    [
        ((65535 * 32 + 1, 1), "CUDA grid limit"),
        ((2**31, 1), "too large"),
        ((1, 2**31), "too large"),
    ],
)
def test_invalid_extent_rejected_before_contiguous_copy(shape, match):
    x = torch.ones(1, 1, device="cuda").expand(shape)
    with pytest.raises(RuntimeError, match=match):
        quantize(x, torch.ones(1, device="cuda"), dtype=F4)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(7, 19), (2, 3, 19), (2, 3, 4, 19), (64, 128)])
def test_int8_per_row_multipliers(input_dtype, shape):
    torch.manual_seed(91)
    x = torch.randn(*shape, device="cuda", dtype=input_dtype).transpose(-1, -2)
    rows = x.float().reshape(-1, x.shape[-1])
    scale = rows.abs().amax(-1).clamp_min(1e-12) / 127
    inv_scale = scale.reciprocal().reshape(-1, 1)
    out, out_t, amax = quantize(x, inv_scale, dtype=I8, tdtype=I8)
    expected = _cast(rows * inv_scale, I8)
    assert out.shape == x.shape and out.is_contiguous() and amax is None
    assert torch.equal(out.reshape(rows.shape), expected)
    assert torch.equal(out_t, expected.t().contiguous())
    torch.testing.assert_close(
        out.float().reshape(rows.shape) * scale[:, None],
        rows,
        rtol=0,
        atol=float(scale.max()) * 0.501,
    )


@pytest.mark.cuda
@skip_no_fp8
@pytest.mark.parametrize("dtype,tdtype", [(F4, I8), (I8, F5), (F5, F4)])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(3, 5, 8), (64, 128)])
def test_row_multipliers_apply_to_original_rows_for_mixed_outputs(
    dtype, tdtype, input_dtype, shape
):
    numel = 1
    for size in shape:
        numel *= size
    x = torch.linspace(-60, 60, numel, device="cuda").reshape(shape).to(input_dtype)
    rows = x.numel() // x.shape[-1]
    inv_scale = torch.linspace(0.125, 1.5, rows, device="cuda").reshape(-1, 1)
    out, out_t, _ = quantize(x, inv_scale, dtype=dtype, tdtype=tdtype)
    ref = x.float().reshape(rows, x.shape[-1]) * inv_scale
    assert torch.equal(
        out.view(torch.uint8).reshape(ref.shape), _cast(ref, dtype).view(torch.uint8)
    )
    assert torch.equal(
        out_t.view(torch.uint8), _cast(ref.t().contiguous(), tdtype).view(torch.uint8)
    )


@pytest.mark.cuda
@skip_no_kernel
def test_int8_rounding_saturation_and_nonfinite_values():
    x = torch.tensor(
        [
            [
                -1000,
                -127.5,
                -126.5,
                -2.5,
                -1.5,
                -0.5,
                0.5,
                1.5,
                2.5,
                126.5,
                127.5,
                1000,
                float("nan"),
                float("inf"),
                -float("inf"),
            ]
        ],
        device="cuda",
    )
    out, out_t, _ = quantize(x, torch.ones(1, device="cuda"), dtype=I8, tdtype=I8)
    expected = torch.tensor(
        [[-127, -127, -126, -2, -2, 0, 0, 2, 2, 126, 127, 127, 0, 127, -127]],
        device="cuda",
        dtype=I8,
    )
    assert torch.equal(out, expected)
    assert torch.equal(out_t, expected.t())


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("shape", [(0,), (0, 17), (2, 0, 17), (3, 0)])
@pytest.mark.parametrize("multiplier", ["scalar", "row", "column", "matrix"])
def test_empty_int8_inputs(shape, multiplier):
    x = torch.empty(shape, device="cuda")
    rows = 1
    for size in shape[:-1]:
        rows *= size
    cols = shape[-1]
    inv_shape = {
        "scalar": (1,),
        "row": (rows, 1),
        "column": (cols,),
        "matrix": (rows, cols),
    }[multiplier]
    inv_scale = torch.ones(inv_shape, device="cuda")
    kwargs = {"dtype": I8}
    if x.ndim >= 2:
        kwargs["tdtype"] = I8
    out, out_t, amax = quantize(x, inv_scale, **kwargs)
    assert out.shape == x.shape and out.numel() == 0 and amax is None
    if out_t is not None:
        assert out_t.shape == (1, 0) and out_t.numel() == 0


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize(
    "case",
    [
        "length",
        "row_extent",
        "column_extent",
        "matrix_extent",
        "excess_rank",
        "cpu",
        "dtype",
    ],
)
def test_invalid_multiplier_broadcasts(case):
    x = torch.ones(2, 3, 5, device="cuda")
    inv_scale = torch.ones(6, 1, device="cuda")
    if case == "length":
        inv_scale = inv_scale.flatten()
    elif case == "row_extent":
        inv_scale = inv_scale[:5]
    elif case == "column_extent":
        inv_scale = torch.ones(1, 6, device="cuda")
    elif case == "matrix_extent":
        inv_scale = torch.ones(6, 4, device="cuda")
    elif case == "excess_rank":
        inv_scale = inv_scale.reshape(2, 3, 1)
    elif case == "cpu":
        inv_scale = inv_scale.cpu()
    elif case == "dtype":
        inv_scale = inv_scale.half()
    with pytest.raises(RuntimeError, match="inv_scale"):
        quantize(x, inv_scale, dtype=I8)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("axis", ["row", "column"])
def test_multiplier_stride_overflow_rejected_on_empty_views(axis):
    shape = (3, 0) if axis == "row" else (0, 3)
    stride = (2**31, 1) if axis == "row" else (3, 2**31)
    x = torch.empty(shape, device="cuda")
    inv_scale = torch.empty(0, device="cuda").as_strided(shape, stride)
    with pytest.raises(RuntimeError, match="inv_scale"):
        quantize(x, inv_scale, dtype=I8)


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("transpose_only", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_scaled_int8_default_range_and_history(transpose_only, external):
    ring = _ring()
    ring[:4] = torch.tensor([12.0, 21.0, 7.0, 17.0], device="cuda")
    scale = torch.zeros(1, device="cuda") if external else ring[4:5]
    recip = torch.zeros(1, device="cuda") if external else ring[5:6]
    x = torch.tensor([[-3.0, 31.75, 1.0], [9.0, -2.0, 0.0]], device="cuda")
    kwargs = dict(hist_len=4, hist_idx=1, margin=1)
    kwargs["tdtype" if transpose_only else "dtype"] = I8
    if external:
        kwargs.update(next_scale=scale, next_inv_scale=recip)
    out, out_t, amax = quantize_scaled(
        x, torch.tensor([2.0], device="cuda"), ring, **kwargs
    )
    expected = _cast(x * 2, I8)
    assert torch.equal(
        out_t if transpose_only else out, expected.t() if transpose_only else expected
    )
    if transpose_only:
        assert out is None
    else:
        assert out_t is None
    assert amax.data_ptr() == ring[6:7].data_ptr() and float(amax) == 31.75
    torch.testing.assert_close(
        ring[:4], torch.tensor([12.0, 31.75, 7.0, 17.0], device="cuda"), rtol=0, atol=0
    )
    torch.testing.assert_close(
        scale, torch.tensor([0.125], device="cuda"), rtol=0, atol=0
    )
    torch.testing.assert_close(
        recip, torch.tensor([8.0], device="cuda"), rtol=0, atol=0
    )
    next_out, _, _ = quantize(x, recip, dtype=I8)
    assert torch.equal(next_out, _cast(x * recip, I8))
    if external:
        assert float(ring[4]) == 0 and float(ring[5]) == 0


@pytest.mark.cuda
@skip_no_kernel
def test_scaled_int8_range_override_empty_fold_and_scalar_only():
    ring = _ring()
    ring[:4].fill_(64)
    _, _, amax = quantize_scaled(
        torch.empty(0, 3, device="cuda"),
        torch.ones(1, device="cuda"),
        ring,
        hist_len=4,
        hist_idx=2,
        dtype=I8,
        tdtype=I8,
        quant_max=128,
    )
    assert float(amax) == 0 and float(ring[2]) == 0
    assert float(ring[4]) == 0.5 and float(ring[5]) == 2
    with pytest.raises(RuntimeError, match="inv_scale"):
        quantize_scaled(
            torch.ones(2, 3, device="cuda"),
            torch.ones(2, device="cuda"),
            ring,
            hist_len=4,
            hist_idx=0,
            dtype=I8,
        )


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_int8_one_dimensional_input(input_dtype):
    x = torch.linspace(-12, 12, 19, device="cuda").to(input_dtype)
    inv_scale = torch.tensor([[0.5]], device="cuda")
    out, out_t, amax = quantize(x, inv_scale, dtype=I8)
    assert torch.equal(out, _cast(x.float() * 0.5, I8))
    assert out.shape == x.shape and out_t is None and amax is None
    with pytest.raises(RuntimeError, match="2D"):
        quantize(x, inv_scale, tdtype=I8)


@pytest.mark.cuda
@skip_no_kernel
def test_int8_zero_rows_with_row_multipliers():
    x = torch.zeros(2, 3, 17, device="cuda")
    scale = torch.full((6,), 1e-12 / 127, device="cuda")
    out, out_t, amax = quantize(
        x, scale.reciprocal().reshape(-1, 1), dtype=I8, tdtype=I8
    )
    assert out.shape == x.shape and out_t.shape == (17, 6) and amax is None
    assert torch.count_nonzero(out) == 0 and torch.count_nonzero(out_t) == 0


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["pure", "row", "column", "matrix", "scaled"])
@pytest.mark.parametrize(
    "dtype,tdtype",
    [
        pytest.param(I8, I8, marks=skip_no_kernel),
        pytest.param(F4, I8, marks=skip_no_fp8),
    ],
)
def test_shared_quantize_stream_and_cuda_graph_replay(mode, dtype, tdtype):
    stream = torch.cuda.Stream()
    x = torch.linspace(-24, 24, 64 * 128, device="cuda").reshape(64, 128)
    if mode == "row":
        inv_scale = torch.linspace(0.25, 2, 64, device="cuda").reshape(-1, 1)
    elif mode == "column":
        inv_scale = torch.linspace(0.25, 2, 256, device="cuda")[::2]
    elif mode == "matrix":
        inv_scale = torch.linspace(0.25, 2, 64 * 256, device="cuda").reshape(64, 256)[
            :, ::2
        ]
    else:
        inv_scale = torch.tensor([0.5], device="cuda")
    ring = _ring() if mode == "scaled" else None
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            if mode == "scaled":
                quantize_scaled(
                    x,
                    inv_scale,
                    ring,
                    hist_len=4,
                    hist_idx=0,
                    dtype=dtype,
                    tdtype=tdtype,
                )
            else:
                quantize(x, inv_scale, dtype=dtype, tdtype=tdtype)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if mode == "scaled":
                out, out_t, amax = quantize_scaled(
                    x,
                    inv_scale,
                    ring,
                    hist_len=4,
                    hist_idx=0,
                    dtype=dtype,
                    tdtype=tdtype,
                )
            else:
                out, out_t, amax = quantize(x, inv_scale, dtype=dtype, tdtype=tdtype)
        x.fill_(4)
        inv_scale.mul_(2)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    ref = x * inv_scale
    assert torch.equal(out.view(torch.uint8), _cast(ref, dtype).view(torch.uint8))
    assert torch.equal(
        out_t.view(torch.uint8), _cast(ref.t().contiguous(), tdtype).view(torch.uint8)
    )
    if mode == "scaled":
        assert float(amax) == 4 and float(ring[0]) == 4
        expected_scale = torch.tensor(
            [4.0 / (127 if dtype == I8 else 448)], device="cuda"
        )
        torch.testing.assert_close(ring[4:5], expected_scale, rtol=0, atol=0)
        torch.testing.assert_close(
            ring[5:6], expected_scale.reciprocal(), rtol=0, atol=0
        )
    else:
        assert amax is None


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,tdtype",
    [
        pytest.param(I8, I8, marks=skip_no_kernel),
        pytest.param(F4, I8, marks=skip_no_fp8),
    ],
)
@pytest.mark.parametrize("shape", [(37, 67), (64, 128)])
@pytest.mark.parametrize("layout", ["row", "column_1d", "column_2d", "matrix"])
@pytest.mark.parametrize("storage", ["strided", "expanded"])
def test_multiplier_tensor_shape_and_storage_strides(
    dtype, tdtype, shape, layout, storage
):
    rows, cols = shape
    x = torch.linspace(-60, 60, rows * cols, device="cuda").reshape(shape)
    if layout == "row":
        if storage == "strided":
            inv_scale = torch.empty(rows * 2, 2, device="cuda")[::2, :1]
            inv_scale.copy_(
                torch.linspace(0.125, 1.5, rows, device="cuda").reshape(-1, 1)
            )
        else:
            inv_scale = torch.tensor([[0.75]], device="cuda").expand(rows, 1)
    elif layout == "column_1d":
        if storage == "strided":
            inv_scale = torch.empty(cols * 2, device="cuda")[::2]
            inv_scale.copy_(torch.linspace(0.125, 1.5, cols, device="cuda"))
        else:
            inv_scale = torch.tensor([0.75], device="cuda").expand(cols)
    elif layout == "column_2d":
        if storage == "strided":
            inv_scale = torch.empty(2, cols * 2, device="cuda")[:1, ::2]
            inv_scale.copy_(
                torch.linspace(0.125, 1.5, cols, device="cuda").reshape(1, -1)
            )
        else:
            inv_scale = torch.tensor([[0.75]], device="cuda").expand(1, cols)
    elif storage == "strided":
        inv_scale = torch.empty(rows * 2, cols * 2, device="cuda")[::2, ::2]
        inv_scale.copy_(
            torch.linspace(0.125, 1.5, rows * cols, device="cuda").reshape(shape)
        )
    else:
        inv_scale = (
            torch.linspace(0.125, 1.5, cols, device="cuda")
            .reshape(1, -1)
            .expand(rows, cols)
        )
    out, out_t, amax = quantize(x, inv_scale, dtype=dtype, tdtype=tdtype)
    ref = x * inv_scale
    assert amax is None
    assert torch.equal(out.view(torch.uint8), _cast(ref, dtype).view(torch.uint8))
    assert torch.equal(
        out_t.view(torch.uint8), _cast(ref.t().contiguous(), tdtype).view(torch.uint8)
    )
