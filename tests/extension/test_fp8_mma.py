"""FP8 primitives: kernel-level (CUDA) and policy-level (CPU-verifiable) tests.

The kernel-level tests exercise the two stateless primitives (``quantize`` for
bf16/fp16/fp32 -> FP8, ``quant_gemm`` for the pre-quantized GEMM with transposed
operands); the policy-level tests (recipes, autocast context, per-tensor
meta) run without a GPU. The primitives themselves are CUDA-only
(attention-style direct wrappers — no torch.library dispatch layer).
"""

import threading

import pytest
import torch
import torch.nn.functional as F

import astrai.extension.quantize as f8mod
from astrai.extension.ops.gemm import quant_gemm
from astrai.extension.ops.quantize import quantize, quantize_dual
from astrai.extension.quantize import (
    FP8Recipe,
    FP8TensorMeta,
    _ScaleRing,
    fp8_autocast,
    fp8_format_pair,
    fp8_linear_enable,
    fp8_linear_enabled,
    fp8_state,
)
from tests.conftest import skip_no_fp8


def _scale(tensor):
    return (tensor.abs().amax().float() / 448.0).clamp_min(1e-12)


def _quantize(tensor, scale, fmt=torch.float8_e4m3fn):
    """Reference quantize: multiply by the reciprocal (the kernel's exact
    arithmetic — a plain divide flips fp8 boundary cases by one ulp)."""
    return (tensor.float() * scale.reciprocal()).to(fmt).float()


# --------------------------------------------------------------------------
# Kernel-level (CUDA)
# --------------------------------------------------------------------------


@skip_no_fp8
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(16, 8, 32), (17, 9, 33), (31, 15, 64), (32, 48, 96)],
)
def test_fp8_mm_matches_explicit_quantization(m, n, k):
    torch.manual_seed(m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    scale_a = _scale(a)
    scale_b = _scale(b)
    a8, _ = quantize(a, scale_a.reciprocal(), torch.float8_e4m3fn)
    b8, _ = quantize(b, scale_b.reciprocal(), torch.float8_e4m3fn)
    out = quant_gemm(a8, b8, a_scale=scale_a * scale_b, trans_b=False)
    expected = (_quantize(a, scale_a) @ _quantize(b, scale_b) * scale_a * scale_b).to(
        torch.bfloat16
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)


@skip_no_fp8
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("fmt", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quantize_input_dtypes(in_dtype, fmt):
    """quantize accepts bf16/fp16/fp32 inputs; bytes and amax match the
    explicit (value * multiplier) reference."""
    torch.manual_seed(3)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32) * 0.5
    x = x.to(in_dtype)
    scale = torch.tensor([0.5], device="cuda")
    x8, amax = quantize(x, scale, fmt)
    out_dtype = fmt
    assert x8.dtype == out_dtype
    assert x8.shape == x.shape
    assert amax is None  # no ring => pure scale+cast, no fused amax
    ref = (x.float() * 0.5).to(out_dtype)
    assert torch.equal(x8, ref)


@skip_no_fp8
def test_quantize_e5m2_format():
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    x8, amax = quantize(x, torch.tensor([10.0], device="cuda"), torch.float8_e5m2)
    assert x8.dtype == torch.float8_e5m2
    assert amax is None  # no ring => no fused amax


@skip_no_fp8
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
def test_quant_gemm_transposed_operands(trans_a, trans_b):
    """quant_gemm handles all four operand layouts via trans_a/trans_b."""
    torch.manual_seed(17)
    m, n, k = 19, 13, 37
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)  # A [M][K]
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)  # B^T [N][K]
    sa, sb = _scale(a), _scale(b)
    a8, _ = quantize(a, sa.reciprocal(), torch.float8_e4m3fn)
    b8, _ = quantize(b, sb.reciprocal(), torch.float8_e4m3fn)
    a_op = a8.t().contiguous() if trans_a else a8
    b_op = b8 if trans_b else b8.t().contiguous()

    out = quant_gemm(a_op, b_op, a_scale=sa * sb, trans_a=trans_a, trans_b=trans_b)
    assert out.shape == (m, n)
    expected = (_quantize(a, sa) @ _quantize(b, sb).t() * sa * sb).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)


@skip_no_fp8
@pytest.mark.parametrize("bias_on", [False, True])
def test_quant_gemm_fused_bias(bias_on):
    """Epilogue-fused bias matches the unfused out + bias reference (single
    fp32 rounding vs the reference's double rounding keeps it within 1 ulp),
    including N-tail columns and batched broadcast."""
    torch.manual_seed(31)
    m, n, k = 19, 13, 37  # odd n exercises the guarded bias loads
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    sa, sb = _scale(a), _scale(b)
    a8, _ = quantize(a, sa.reciprocal(), torch.float8_e4m3fn)
    b8, _ = quantize(b, sb.reciprocal(), torch.float8_e4m3fn)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    out = quant_gemm(
        a8, b8, a_scale=sa * sb, trans_b=True, bias=bias if bias_on else None
    )
    base = (_quantize(a, sa) @ _quantize(b, sb).t() * sa * sb).to(torch.bfloat16)
    expected = base + bias if bias_on else base
    # bias is O(1) against O(sqrt(k)) accumulators: absolute tolerance rules
    torch.testing.assert_close(out, expected, atol=0.13, rtol=0.01)

    # Batched broadcast: bias applies to every batch slice (each slice gets
    # its own reference from its own operand values).
    ab = torch.randn(3, m, k, device="cuda", dtype=torch.bfloat16)
    ab8, _ = quantize(ab, sa.reciprocal(), torch.float8_e4m3fn)
    outb = quant_gemm(ab8, b8, a_scale=sa * sb, trans_b=True, bias=bias)
    assert outb.shape == (3, m, n)
    for i in range(3):
        expected_b = (_quantize(ab[i], sa) @ _quantize(b, sb).t() * sa * sb).to(
            torch.bfloat16
        ) + bias
        torch.testing.assert_close(outb[i], expected_b, atol=0.13, rtol=0.01)


@skip_no_fp8
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
def test_quant_gemm_batched(trans_a, trans_b):
    """3D operands run as one bmm launch: all four layouts, odd shapes."""
    torch.manual_seed(23)
    batch, m, n, k = 4, 19, 13, 37
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(batch, n, k, device="cuda", dtype=torch.bfloat16)
    sa, sb = _scale(a), _scale(b)
    a8, _ = quantize(a, sa.reciprocal(), torch.float8_e4m3fn)
    b8, _ = quantize(b, sb.reciprocal(), torch.float8_e4m3fn)
    a_op = a8.transpose(-2, -1).contiguous() if trans_a else a8
    b_op = b8 if trans_b else b8.transpose(-2, -1).contiguous()

    out = quant_gemm(a_op, b_op, a_scale=sa * sb, trans_a=trans_a, trans_b=trans_b)
    assert out.shape == (batch, m, n)
    # flags + transposed buffers reconstruct the original operands: the math
    # is always A_orig @ B_orig^T regardless of the layout combination.
    expected = (_quantize(a, sa) @ _quantize(b, sb).transpose(-2, -1) * sa * sb).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)


@skip_no_fp8
def test_quant_gemm_batched_broadcast():
    """A size-1 batch broadcasts across the other operand (matmul rules),
    and a 2D operand broadcasts across a 3D one."""
    torch.manual_seed(29)
    batch, m, n, k = 3, 16, 8, 32
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1, n, k, device="cuda", dtype=torch.bfloat16)
    sa, sb = _scale(a), _scale(b)
    a8, _ = quantize(a, sa.reciprocal(), torch.float8_e4m3fn)
    b8, _ = quantize(b, sb.reciprocal(), torch.float8_e4m3fn)

    out = quant_gemm(a8, b8, a_scale=sa * sb, trans_b=True)
    assert out.shape == (batch, m, n)
    expected = (_quantize(a, sa) @ _quantize(b, sb).transpose(-2, -1) * sa * sb).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)

    # 2D weight broadcast over 3D activations
    w8 = b8[0]
    out2 = quant_gemm(a8, w8, a_scale=sa * sb, trans_b=True)
    assert out2.shape == (batch, m, n)
    torch.testing.assert_close(out2, expected, atol=0.125, rtol=0.01)


@skip_no_fp8
def test_quant_gemm_col_major_view_zero_copy():
    """An inner-transposed view (.t() of a contiguous buffer) folds into the
    layout tag with no device copy — the only allocation is the output."""
    torch.manual_seed(31)
    m, n, k = 64, 64, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    sa, sb = _scale(a), _scale(b)
    a8, _ = quantize(a, sa.reciprocal(), torch.float8_e4m3fn)
    b8, _ = quantize(b, sb.reciprocal(), torch.float8_e4m3fn)

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = quant_gemm(a8.t(), b8, a_scale=sa * sb, trans_a=True, trans_b=True)
    torch.cuda.synchronize()
    grew = torch.cuda.memory_allocated() - before
    assert grew == out.numel() * out.element_size()  # no operand copy

    expected = (_quantize(a, sa) @ _quantize(b, sb).t() * sa * sb).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=0.125, rtol=0.01)


@skip_no_fp8
def test_delayed_scaling_forward_uses_snapshot_scale():
    """The delayed scale for step N is computed from amax(steps < N); the
    forward must snapshot the scale before the ring update, so a changing
    amax across steps does not leak the next-step scale into the output."""
    torch.manual_seed(11)
    dev = torch.device("cuda")
    state = f8mod.fp8_state()
    state.reset()
    state.default_recipe = FP8Recipe(history_len=1, margin=0)
    state.default_format = (torch.float8_e4m3fn, torch.float8_e4m3fn)
    try:
        m, n, k = 32, 16, 64
        x1 = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.5
        # Smaller amax than x1: the delayed scale (amax(x1)/448) still covers
        # x2 without fp8 saturation, while the next-step scale would differ.
        x2 = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.35
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.5
        bias = torch.zeros(n, device=dev, dtype=torch.bfloat16)

        f8mod.fp8_linear_forward(x1, w, bias)  # step 1: seeds the rings
        out2, _, _ = f8mod.fp8_linear_forward(x2, w, bias)  # amax changes
        torch.cuda.synchronize()

        # The delayed scale for step 2 is amax(x1)/448 (history_len=1); the
        # GEMM must use that same scale for dequant as the quantize used.
        sx = _scale(x1)
        sw = _scale(w)
        qx = _quantize(x2, sx)
        qw = _quantize(w, sw)
        expected = (qx @ qw.t() * sx * sw + bias).to(torch.bfloat16)
        torch.testing.assert_close(out2, expected, atol=0.125, rtol=0.01)
    finally:
        state.reset()


@skip_no_fp8
def test_fp8_linear_forward_and_backward():
    """The composed strategy path: forward quantize+GEMM+bias, backward
    dX/dW GEMMs on transposed operands (E5M2 in hybrid)."""
    torch.manual_seed(7)
    m, n, k = 19, 13, 37
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    state = f8mod.fp8_state()
    state.reset()
    state.default_recipe = FP8Recipe(dynamic=True)
    try:
        out, _, _ = f8mod.fp8_linear_forward(x, weight, bias)

        sx, sw = _scale(x), _scale(weight)
        qx = _quantize(x, sx)
        qw = _quantize(weight, sw)
        expected_out = (qx @ qw.t() * sx * sw + bias).to(torch.bfloat16)
        torch.testing.assert_close(out, expected_out, atol=0.125, rtol=0.01)

        # backward through the aten::linear integration (hybrid E5M2). The
        # incoming gradient is 2*out of the *fp8* forward (bf16-rounded), not
        # 2*exact — derive the reference from the actual output.
        xr = x.detach().clone().requires_grad_()
        wr = weight.detach().clone().requires_grad_()
        br = bias.detach().clone().requires_grad_()
        with fp8_autocast(enabled=True):
            loss = F.linear(xr, wr, br).float().pow(2).sum()
        loss.backward()

        g = (2 * out.float()).to(torch.bfloat16).float()  # actual grad wrt out
        # the dynamic path measures current-step amax in the bwd fmt (E5M2);
        # amax must be taken in fp32 — a bf16-rounded scale flips E5M2
        # boundary rounding (2-bit mantissa) and the reference drifts.
        e5 = 57344.0
        sg = (g.abs().amax() / e5).clamp_min(1e-12)
        sw5 = (weight.abs().amax().float() / e5).clamp_min(1e-12)
        sx5 = (x.abs().amax().float() / e5).clamp_min(1e-12)
        expected_grad_x = (
            _quantize(g, sg, torch.float8_e5m2)
            @ _quantize(weight, sw5, torch.float8_e5m2)
            * sg
            * sw5
        ).to(torch.bfloat16)
        expected_grad_w = (
            _quantize(g, sg, torch.float8_e5m2).t()
            @ _quantize(x, sx5, torch.float8_e5m2)
            * sg
            * sx5
        ).to(torch.bfloat16)
        torch.testing.assert_close(xr.grad, expected_grad_x, atol=0.5, rtol=0.05)
        torch.testing.assert_close(wr.grad, expected_grad_w, atol=0.5, rtol=0.05)
        torch.testing.assert_close(
            br.grad, g.sum(0).to(torch.bfloat16), atol=0.5, rtol=0.05
        )
    finally:
        state.reset()


@skip_no_fp8
def test_fp8_linear_backward_outside_autocast():
    """aten::linear records an fp8 autograd node inside fp8_autocast; the
    backward runs fp8 kernels even after the context exits (loss.backward()
    placement is free), instead of falling back to bf16 mm."""
    torch.manual_seed(5)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        96, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    xr, wr, br = (t.detach().clone().requires_grad_() for t in (x, weight, bias))

    calls = {"fwd": 0}
    orig = f8mod.fp8_linear_forward

    def spy(*args, **kwargs):
        calls["fwd"] += 1
        return orig(*args, **kwargs)

    f8mod.fp8_linear_forward = spy
    try:
        with fp8_autocast(enabled=True):
            out = F.linear(x, weight, bias)
        assert type(out.grad_fn).__name__ == "_LinearFp8Backward"
        out.float().pow(2).sum().backward()  # outside the autocast region
    finally:
        f8mod.fp8_linear_forward = orig
        f8mod.fp8_state().reset()

    assert calls["fwd"] == 1  # fp8 kernels, not the bf16 fallback
    ref = F.linear(xr, wr, br)
    ref.float().pow(2).sum().backward()

    # E5M2 backward quantization noise: compare directions/norms (the
    # torchao/TE style) rather than elementwise against the bf16 reference.
    def _direction(a, b):
        cos = torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        )
        return cos > 0.99 and 0.9 < a.float().norm() / b.float().norm() < 1.1

    assert _direction(x.grad, xr.grad)
    assert _direction(weight.grad, wr.grad)
    assert _direction(bias.grad, br.grad)


@skip_no_fp8
def test_quant_gemm_matches_scaled_mm():
    torch.manual_seed(11)
    m, n, k = 512, 4096, 4096
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    sa = _scale(a)
    sb = _scale(b)
    a8, _ = quantize(a, sa.reciprocal(), torch.float8_e4m3fn)
    b8, _ = quantize(b, sb.reciprocal(), torch.float8_e4m3fn)
    out = quant_gemm(a8, b8, a_scale=sa * sb, trans_b=False)
    assert out.dtype == torch.bfloat16
    assert out.shape == (m, n)

    ref = (a8.float().double() @ b8.float().double() * sa * sb).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=6.0, rtol=0.05)

    try:
        torch._scaled_mm(a8, b8, sa, sb, out_dtype=torch.bfloat16)
    except (RuntimeError, NotImplementedError):
        return
    torch.testing.assert_close(
        out,
        torch._scaled_mm(a8, b8, sa, sb, out_dtype=torch.bfloat16),
        atol=2.0,
        rtol=0.01,
    )


# --------------------------------------------------------------------------
# Policy-level (CPU-verifiable)
# --------------------------------------------------------------------------


def test_recipe_scale_from_history():
    """Delayed: max over the window + margin; dynamic: current amax."""
    hist = torch.tensor([1.0, 2.0, 0.5])
    d = FP8Recipe(history_len=3, margin=0)
    assert torch.allclose(
        d.scale_from_history(hist, torch.float8_e4m3fn), torch.tensor(2.0 / 448.0)
    )
    d_m = FP8Recipe(history_len=3, margin=2)
    assert torch.allclose(
        d_m.scale_from_history(hist, torch.float8_e4m3fn),
        torch.tensor(2.0 / 448.0 / 4.0),
    )
    dyn = FP8Recipe(dynamic=True)
    amax = torch.tensor([0.25])
    assert torch.allclose(
        dyn.scale_from_history(amax, torch.float8_e4m3fn),
        torch.tensor(0.25 / 448.0),
    )
    assert torch.allclose(
        dyn.scale_from_history(amax, torch.float8_e5m2),
        torch.tensor(0.25 / 57344.0),
    )


def test_fp8_format_pair():
    """A format spec normalizes to (fwd, bwd) fp8 dtypes; torch.dtype is the
    canonical key — no custom format enum."""
    assert fp8_format_pair("hybrid") == (torch.float8_e4m3fn, torch.float8_e5m2)
    assert fp8_format_pair(torch.float8_e4m3fn) == (
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
    )
    assert fp8_format_pair(torch.float8_e5m2) == (
        torch.float8_e5m2,
        torch.float8_e5m2,
    )


def test_fp8_autocast_context():
    """fp8_autocast pushes and restores the thread-local active config."""
    state = fp8_state()
    state.reset()
    try:
        with fp8_autocast(enabled=True, fp8_format="hybrid", update_interval=8):
            cfg = f8mod._active_config.get()
            assert cfg is not None and cfg.enabled
            assert not cfg.recipe.dynamic
            assert cfg.recipe.history_len == 8
            assert cfg.fp8_format == (torch.float8_e4m3fn, torch.float8_e5m2)
            with fp8_autocast(
                enabled=True,
                recipe=FP8Recipe(dynamic=True),
                fp8_format=torch.float8_e4m3fn,
            ):
                inner = f8mod._active_config.get()
                assert inner.recipe.dynamic
                assert inner.fp8_format == (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fn,
                )
            assert f8mod._active_config.get() is cfg  # restored on exit
        assert f8mod._active_config.get() is None
        assert not fp8_linear_enabled()
    finally:
        state.reset()


def test_fp8_tensor_meta_delayed_update():
    """Meta seeds from data; hist/scale are packed views of one state buffer."""
    recipe = FP8Recipe(history_len=4, margin=0)
    meta = FP8TensorMeta(
        _ScaleRing(torch.device("cpu"), recipe),
        _ScaleRing(torch.device("cpu"), recipe),
        _ScaleRing(torch.device("cpu"), recipe),
    )
    w = torch.randn(8, 8)
    meta.w.seed(w, torch.float8_e4m3fn)
    assert meta.w.initialized
    torch.testing.assert_close(meta.w.scale, (w.abs().amax() / 448.0).reshape(1))
    # [hist | scale | legacy | amax | done] packing: views alias one buffer.
    assert meta.w.state.numel() == 4 + 4
    assert meta.w.hist.data_ptr() == meta.w.state.data_ptr()
    assert meta.w.scale.data_ptr() == meta.w.state[4:].data_ptr()
    meta.w.advance()
    assert meta.w.idx == 1

    # fold_args hands the kernel the buffer, the slot and the recipe constants
    args = meta.w.fold_args(torch.float8_e4m3fn)
    assert args["ring_state"] is meta.w.state and args["hist_idx"] == 1
    assert args["fp8_max"] == 448.0 and args["pow2_margin"] == 1.0


@skip_no_fp8
@pytest.mark.parametrize("fmt", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quantize_dual_and_transposed_orientations(fmt):
    """quantize_dual yields both orientations from one read; quantize's
    transposed switch keeps the 2-tuple arity with the [cols][rows] layout."""
    torch.manual_seed(11)
    x = torch.randn(37, 67, device="cuda", dtype=torch.bfloat16) * 3
    mult = _scale(x).reciprocal()
    x8, amax = quantize(x, mult, fmt)
    x8T, _ = quantize(x, mult, fmt, transposed=True)
    d8, d8T, _ = quantize_dual(x, mult, fmt)
    assert x8T.shape == (67, 37)
    assert torch.equal(x8.view(torch.uint8), d8.view(torch.uint8))
    assert torch.equal(x8T.view(torch.uint8), d8T.view(torch.uint8))
    assert torch.equal(x8T.t().contiguous().view(torch.uint8), x8.view(torch.uint8))
    assert amax is None  # no ring => no fused amax


@skip_no_fp8
@pytest.mark.parametrize(
    "fmt,fmax",
    [(torch.float8_e4m3fn, 448.0), (torch.float8_e5m2, 57344.0)],
)
@pytest.mark.parametrize("margin", [0, 1])
def test_quantize_ring_fold_matches_host_update(fmt, fmax, margin):
    """The in-kernel delayed-scaling fold matches a host-side reference."""
    dev = torch.device("cuda")
    n, idx = 4, 2
    torch.manual_seed(3)
    x = torch.randn(128, 96, dtype=torch.bfloat16, device=dev) * 3
    mult = torch.tensor([0.01], device=dev)
    pow2m = float(2**margin)

    # Reference: legacy quantize + the host fold (amax measured out-of-band,
    # as dynamic scaling does — the no-ring kernel no longer returns one).
    x8_ref = quantize(x, mult, fmt)[0]
    hist = torch.full((n,), 1.0, device=dev)
    hist[idx] = x.float().abs().amax().reshape(1)
    scale = (hist.max() / fmax / pow2m).clamp_min(1e-12).reshape(1)

    # Fused: same window, fold inside the quantize kernel's last block.
    ring = torch.zeros(n + 4, device=dev)
    ring[:n].fill_(1.0)
    x8, _ = quantize(
        x,
        mult,
        fmt,
        ring_state=ring,
        hist_idx=idx,
        fp8_max=fmax,
        pow2_margin=pow2m,
    )
    assert torch.equal(x8.view(torch.uint8), x8_ref.view(torch.uint8))
    torch.testing.assert_close(ring[:n], hist, rtol=0, atol=0)
    torch.testing.assert_close(ring[n : n + 1], scale, rtol=0, atol=0)
    assert float(ring[n + 2]) == 0.0  # amax slot self-cleaned
    assert int(ring[n + 3].view(torch.int32)) == 0  # done counter reset


@skip_no_fp8
@pytest.mark.parametrize(
    "fmt,fmax",
    [(torch.float8_e4m3fn, 448.0), (torch.float8_e5m2, 57344.0)],
)
def test_quantize_ring_fold_tall_dual_grid(fmt, fmax):
    """The delayed-scaling fold counts BOTH grid dims: the dual (tiled)
    kernel launches a 2D grid, and a tall tensor with a late global max must
    fold the full amax, not a round-local partial."""
    dev = torch.device("cuda")
    n, idx = 4, 2
    torch.manual_seed(7)
    x = torch.randn(8192, 256, dtype=torch.bfloat16, device=dev) * 3
    x[8000, 7] = 100.0  # late row block: a gridDim.x-only fold misses it
    mult = torch.tensor([1.0], device=dev)

    x8_ref = quantize(x, mult, fmt)[0]
    hist = torch.full((n,), 1.0, device=dev)
    hist[idx] = x.float().abs().amax().reshape(1)
    scale = (hist.max() / fmax).clamp_min(1e-12).reshape(1)

    ring = torch.zeros(n + 4, device=dev)
    ring[:n].fill_(1.0)
    d8, d8T, _ = quantize_dual(
        x, mult, fmt, ring_state=ring, hist_idx=idx, fp8_max=fmax, pow2_margin=1.0
    )
    assert torch.equal(d8.view(torch.uint8), x8_ref.view(torch.uint8))
    torch.testing.assert_close(ring[:n], hist, rtol=0, atol=0)
    torch.testing.assert_close(ring[n : n + 1], scale, rtol=0, atol=0)
    assert float(ring[n + 2]) == 0.0
    assert int(ring[n + 3].view(torch.int32)) == 0


@skip_no_fp8
def test_dynamic_recipe_backward():
    """fp8 dynamic scaling runs its backward through quantize_dual without a
    ring (amax measured inline); the fold must not reference the None meta."""
    torch.manual_seed(5)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    state = fp8_state()
    state.reset()
    try:
        with fp8_autocast(enabled=True, recipe=FP8Recipe(dynamic=True)):
            out = F.linear(x, w)
            out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert w.grad is not None and torch.isfinite(w.grad).all()
    finally:
        state.reset()


@skip_no_fp8
def test_quantize_amax_presence():
    """No ring => pure scale+cast (amax None); ring => the delayed-scaling
    fold fills a self-cleaned amax slot."""
    torch.manual_seed(9)
    x = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
    mult = _scale(x).reciprocal()
    x8, amax = quantize(x, mult, torch.float8_e4m3fn)
    assert amax is None
    d8, d8T, amax_dual = quantize_dual(x, mult, torch.float8_e4m3fn)
    assert amax_dual is None
    assert torch.equal(d8.view(torch.uint8), x8.view(torch.uint8))
    assert torch.equal(d8T.view(torch.uint8), x8.t().contiguous().view(torch.uint8))
    # ring: the in-kernel fold writes the amax slot, then self-cleans it.
    ring = torch.zeros(8, device="cuda")
    ring[:4].fill_(1.0)
    x8r, amax_ring = quantize(
        x,
        mult,
        torch.float8_e4m3fn,
        ring_state=ring,
        hist_idx=2,
        fp8_max=448.0,
        pow2_margin=1.0,
    )
    assert amax_ring is not None
    assert torch.equal(x8.view(torch.uint8), x8r.view(torch.uint8))
    assert float(amax_ring) == 0.0  # amax slot self-cleaned by the fold
    assert int(ring[7].view(torch.int32)) == 0  # done counter reset


# --------------------------------------------------------------------------
# torch-autocast parity: context semantics (nesting, thread locality, switch)
# --------------------------------------------------------------------------


def _linear():
    """Shared helper: a small bf16 linear operand set on CUDA (grad-tracking
    so aten::linear records an autograd node)."""
    torch.manual_seed(31)
    x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return x, w


@skip_no_fp8
def test_nested_disabled_region_redispatches_bf16():
    """A nested fp8_autocast(enabled=False) region temporarily restores the
    bf16 aten::linear path (torch's nested-disable semantics), and fp8
    resumes when it exits."""
    x, w = _linear()
    with fp8_autocast(enabled=True):
        F.linear(x, w)
        with fp8_autocast(enabled=False):
            out_bf16 = F.linear(x, w)
            assert type(out_bf16.grad_fn).__name__ != "_LinearFp8Backward"
            assert out_bf16.dtype == torch.bfloat16
        out_again = F.linear(x, w)
        assert type(out_again.grad_fn).__name__ == "_LinearFp8Backward"


@skip_no_fp8
def test_global_switch_routes_without_region():
    """fp8_linear_enable(True) routes aten::linear to fp8 outside any region
    (the persistent default); disabling restores bf16."""
    x, w = _linear()
    state = fp8_state()
    try:
        fp8_linear_enable(True)
        out = F.linear(x, w)
        assert type(out.grad_fn).__name__ == "_LinearFp8Backward"
        fp8_linear_enable(False)
        out = F.linear(x, w)
        assert type(out.grad_fn).__name__ != "_LinearFp8Backward"
    finally:
        state.reset()


def test_autocast_state_is_thread_local():
    """torch parity: the active config is thread-local — another thread does
    not see an open region (CPU-only check of the flag, no kernels)."""
    seen = {}
    with fp8_autocast(enabled=True):
        assert fp8_linear_enabled()
        t = threading.Thread(target=lambda: seen.update(enabled=fp8_linear_enabled()))
        t.start()
        t.join()
    assert seen["enabled"] is False
    assert not fp8_linear_enabled()
