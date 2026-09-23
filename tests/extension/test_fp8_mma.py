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
from astrai.extension.loader import get_module
from astrai.extension.ops.gemm import quant_gemm
from astrai.extension.ops.quantize import quantize, quantize_dual

try:
    from astrai.extension.ops.quantize import K_FOLD_SLOTS
except RuntimeError:
    # The binding must stay import-safe on boxes without the extension;
    # every K_FOLD_SLOTS use sits inside kernel-level tests that skip
    # via skip_no_fp8/skip_no_kernel when the kernel is not built.
    K_FOLD_SLOTS = None
from astrai.extension.quantize import (
    FP8Recipe,
    fp8_autocast,
    fp8_format_pair,
    fp8_linear_enable,
    fp8_linear_enabled,
    fp8_load_state_dict,
    fp8_state_dict,
)
from tests.conftest import skip_no_fp8


def _gemm():
    """The gemm kernel module — the composed fp8 linear + its debug hooks."""
    return get_module("gemm")


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
    gemm = _gemm()
    gemm.fp8_reset()
    try:
        m, n, k = 32, 16, 64
        x1 = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.5
        # Smaller amax than x1: the delayed scale (amax(x1)/448) still covers
        # x2 without fp8 saturation, while the next-step scale would differ.
        x2 = torch.randn(m, k, device=dev, dtype=torch.bfloat16) * 0.35
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.5
        bias = torch.zeros(n, device=dev, dtype=torch.bfloat16)

        # history_len=1, symmetric E4M3 pair, update_rings=True (the training
        # bookkeeping without autograd — the old pure-function mode).
        kw = (True, False, False, 1, 0, torch.float8_e4m3fn, torch.float8_e4m3fn)
        gemm.fp8_linear(x1, w, bias, *kw)  # step 1: seeds the rings
        out2 = gemm.fp8_linear(x2, w, bias, *kw)  # amax changes
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
        gemm.fp8_reset()


@skip_no_fp8
def test_fp8_linear_forward_and_backward():
    """The composed strategy path: forward quantize+GEMM+bias, backward
    dX/dW GEMMs on transposed operands (E5M2 in hybrid)."""
    torch.manual_seed(7)
    m, n, k = 19, 13, 37
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    gemm = _gemm()
    gemm.fp8_reset()
    try:
        out = gemm.fp8_linear(
            x,
            weight,
            bias,
            True,
            False,
            True,
            16,
            0,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )

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
        gemm.fp8_reset()


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

    gemm = _gemm()
    gemm.fp8_reset()
    gemm.fp8_debug_reset_stats()
    try:
        with fp8_autocast(enabled=True):
            out = F.linear(x, weight, bias)
        assert out.grad_fn is not None  # an fp8 node owns the backward
        out.float().pow(2).sum().backward()  # outside the autocast region
    finally:
        stats = gemm.fp8_debug_stats()
        gemm.fp8_reset()

    # One fwd + two bwd GEMMs: fp8 kernels, not the bf16 fallback.
    assert stats["gemm"] == 3, stats
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


@skip_no_fp8
@pytest.mark.parametrize(
    ("margin", "hist_scale"),
    [(0, 1.0), (2, 0.25)],  # scale = amax/448/2**margin
)
def test_recipe_scale_from_history(margin, hist_scale):
    """The scale formula (max over the window / finfo / 2**margin) as the C++
    op applies it at seed time: a no-grad forward seeds the rings, and the
    published scale must equal the closed form on a known amax."""
    torch.manual_seed(3)
    dev = torch.device("cuda")
    gemm = _gemm()
    gemm.fp8_reset()
    try:
        hist = [1.0, 2.0, 0.5]
        # A ring prefilled with a known window (via the snapshot path), then a
        # no-grad forward reads it without folding.
        n, k = 8, 64
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.1
        x = torch.randn(4, k, device=dev, dtype=torch.bfloat16) * 0.1
        gemm.fp8_linear(
            x,
            w,
            None,
            False,
            False,
            False,
            3,
            margin,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
        )
        sd = gemm.fp8_state_dict()
        entry = sd["entries"][0]
        entry["w"]["state"][:3] = torch.tensor(hist, device=dev)
        gemm.fp8_reset()
        gemm.fp8_load_state_dict(sd)
        # An update-rings forward folds from the restored window: the last
        # block republishes scale = max(hist)/448/2**margin in-kernel (the
        # current x's amax lands at hist[idx] but stays below the window max).
        gemm.fp8_linear(
            x,
            w,
            None,
            True,
            False,
            False,
            3,
            margin,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
        )
        scale = gemm.fp8_debug_meta(w, 3, margin)["w"]["scale"]
        expected = 2.0 / 448.0 * hist_scale
        torch.testing.assert_close(scale, torch.full_like(scale, expected))
    finally:
        gemm.fp8_reset()


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
    f8mod.fp8_reset()
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
        f8mod.fp8_reset()


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
    ring = torch.zeros(n + 4 + K_FOLD_SLOTS, device=dev)
    ring[:n].fill_(1.0)
    x8, amax = quantize(
        x,
        mult,
        fmt,
        ring_state=ring,
        hist_idx=idx,
        hist_len=n,
        fp8_max=fmax,
        pow2_margin=pow2m,
    )
    assert torch.equal(x8.view(torch.uint8), x8_ref.view(torch.uint8))
    torch.testing.assert_close(ring[:n], hist, rtol=0, atol=0)
    torch.testing.assert_close(ring[n : n + 1], scale, rtol=0, atol=0)
    # The reported amax is the round's peak (hist[idx] is host-computed above).
    torch.testing.assert_close(amax.reshape(()), hist[idx], rtol=0, atol=0)
    assert float(ring[n + 4 :].abs().sum()) == 0.0  # fold scratch self-cleaned
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

    ring = torch.zeros(n + 4 + K_FOLD_SLOTS, device=dev)
    ring[:n].fill_(1.0)
    d8, d8T, amax = quantize_dual(
        x,
        mult,
        fmt,
        ring_state=ring,
        hist_idx=idx,
        hist_len=n,
        fp8_max=fmax,
        pow2_margin=1.0,
    )
    assert torch.equal(d8.view(torch.uint8), x8_ref.view(torch.uint8))
    torch.testing.assert_close(ring[:n], hist, rtol=0, atol=0)
    torch.testing.assert_close(ring[n : n + 1], scale, rtol=0, atol=0)
    torch.testing.assert_close(amax.reshape(()), hist[idx], rtol=0, atol=0)
    assert float(ring[n + 4 :].abs().sum()) == 0.0
    assert int(ring[n + 3].view(torch.int32)) == 0


@skip_no_fp8
def test_dynamic_recipe_backward():
    """fp8 dynamic scaling runs its backward through quantize_dual without a
    ring (amax measured inline); the fold must not reference the None meta."""
    torch.manual_seed(5)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(96, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    f8mod.fp8_reset()
    try:
        with fp8_autocast(enabled=True, recipe=FP8Recipe(dynamic=True)):
            out = F.linear(x, w)
            out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert w.grad is not None and torch.isfinite(w.grad).all()
    finally:
        f8mod.fp8_reset()


@skip_no_fp8
def test_quantize_amax_presence():
    """No ring => pure scale+cast (amax None); ring => the delayed-scaling
    fold reports the round's raw-domain amax on that output."""
    torch.manual_seed(9)
    x = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
    mult = _scale(x).reciprocal()
    x8, amax = quantize(x, mult, torch.float8_e4m3fn)
    assert amax is None
    d8, d8T, amax_dual = quantize_dual(x, mult, torch.float8_e4m3fn)
    assert amax_dual is None
    assert torch.equal(d8.view(torch.uint8), x8.view(torch.uint8))
    assert torch.equal(d8T.view(torch.uint8), x8.t().contiguous().view(torch.uint8))
    # ring: the fold reports the round's amax and self-cleans its scratch.
    ring = torch.zeros(8 + K_FOLD_SLOTS, device="cuda")
    ring[:4].fill_(1.0)
    x8r, amax_ring = quantize(
        x,
        mult,
        torch.float8_e4m3fn,
        ring_state=ring,
        hist_idx=2,
        hist_len=4,
        fp8_max=448.0,
        pow2_margin=1.0,
    )
    assert amax_ring is not None
    assert torch.equal(x8.view(torch.uint8), x8r.view(torch.uint8))
    # The round's amax — the window holds only 1.0, so a stale slot would read 0.
    torch.testing.assert_close(
        amax_ring.reshape(()), x.float().abs().amax(), rtol=0, atol=0
    )
    assert float(ring[4 + 4 :].abs().sum()) == 0.0  # fold scratch self-cleaned
    assert int(ring[7].view(torch.int32)) == 0  # done counter reset


@skip_no_fp8
def test_quantize_empty_input_ring_still_publishes():
    """An empty tensor still fires one block so the delayed-scaling fold
    publishes: hist records the round's amax 0, scale comes off the window,
    the fold reports that 0 on the amax sink and self-cleans its scratch."""
    dev = torch.device("cuda")
    n, idx = 4, 1
    ring = torch.zeros(n + 4 + K_FOLD_SLOTS, device=dev)
    ring[:n].fill_(1.0)
    x = torch.empty(0, 1536, dtype=torch.bfloat16, device=dev)
    mult = torch.tensor([1.0], device=dev)
    x8, amax = quantize(
        x,
        mult,
        torch.float8_e4m3fn,
        ring_state=ring,
        hist_idx=idx,
        hist_len=n,
        fp8_max=448.0,
        pow2_margin=1.0,
    )
    assert x8.shape == (0, 1536)
    assert amax is not None and float(amax) == 0.0  # the empty round's amax
    assert float(ring[idx]) == 0.0
    expected_scale = (ring[:n].max() / 448.0).clamp_min(1e-12).reshape(1)
    torch.testing.assert_close(ring[n : n + 1], expected_scale, rtol=0, atol=0)
    assert float(ring[n + 4 :].abs().sum()) == 0.0  # fold scratch self-cleaned
    assert int(ring[n + 3].view(torch.int32)) == 0  # done counter reset


@skip_no_fp8
def test_quantize_ring_requires_hist_len():
    """A ring without hist_len is refused, not guessed: the composed ring's
    trailing scale pair makes the window unrecoverable from numel, and the
    inference that used to run silently read the scale slots as history."""
    dev = torch.device("cuda")
    ring = torch.zeros(4 + 4 + K_FOLD_SLOTS, device=dev)
    x = torch.randn(32, 64, dtype=torch.bfloat16, device=dev)
    mult = _scale(x).reciprocal()
    with pytest.raises(RuntimeError, match="hist_len"):
        quantize(x, mult, torch.float8_e4m3fn, ring_state=ring, hist_idx=0)


@skip_no_fp8
def test_quantize_ring_scratch_self_clean_across_reuse():
    """Two rounds on one ring with a shrinking amax: the fold scratch must
    zero itself each round so round 2's scale tracks round 2's amax, not a
    leftover from round 1's larger blocks."""
    dev = torch.device("cuda")
    n, idx = 4, 0
    ring = torch.zeros(n + 4 + K_FOLD_SLOTS, device=dev)
    ring[:n].fill_(0.1)
    big = torch.randn(64, 96, dtype=torch.bfloat16, device=dev) * 50
    small = torch.randn(64, 96, dtype=torch.bfloat16, device=dev) * 0.05
    mult = torch.tensor([1.0], device=dev)
    kw = dict(ring_state=ring, hist_idx=idx, hist_len=n, fp8_max=448.0, pow2_margin=1.0)

    quantize(big, mult, torch.float8_e4m3fn, **kw)
    assert float(ring[n + 4 :].abs().sum()) == 0.0
    scale1 = float(ring[n])

    quantize(small, mult, torch.float8_e4m3fn, **kw)
    assert float(ring[n + 4 :].abs().sum()) == 0.0
    amax2 = small.float().abs().amax().reshape(1)
    torch.testing.assert_close(ring[idx].reshape(1), amax2, rtol=0, atol=0)
    expected2 = (torch.tensor(max(0.1, float(amax2)), device=dev) / 448.0).clamp_min(
        1e-12
    )
    torch.testing.assert_close(ring[n : n + 1], expected2.reshape(1), rtol=0, atol=0)
    assert float(ring[n]) < scale1  # round 1's amax did not leak


@skip_no_fp8
@pytest.mark.parametrize(
    ("rows", "cols"),
    [(32, 256), (33, 255), (33, 257), (37, 260), (64, 512), (65, 64)],
)
def test_quantize_tile_boundary_shapes(rows, cols):
    """Bitwise parity across RM/T/Dual on shapes around the tile geometry:
    exact multiples, col/row tails beyond the block width, and rows%4!=0
    (which must fall back to the scalar transposed store)."""
    torch.manual_seed(rows * 1000 + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 3
    mult = _scale(x).reciprocal()
    ref = (x.float() * mult).to(torch.float8_e4m3fn)
    x8, _ = quantize(x, mult, torch.float8_e4m3fn)
    x8T, _ = quantize(x, mult, torch.float8_e4m3fn, transposed=True)
    d8, d8T, _ = quantize_dual(x, mult, torch.float8_e4m3fn)
    assert x8T.shape == (cols, rows)
    assert torch.equal(x8.view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(d8.view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(x8T.view(torch.uint8), d8T.view(torch.uint8))
    assert torch.equal(x8T.t().contiguous().view(torch.uint8), ref.view(torch.uint8))


@skip_no_fp8
def test_quantize_fp32_transposed_and_dual():
    """fp32 inputs through T/Dual stay bitwise-identical to the explicit
    reference (the fp32 traits path had transposed/dual coverage only via
    indirect consumers before)."""
    torch.manual_seed(13)
    x = torch.randn(96, 320, device="cuda", dtype=torch.float32) * 3
    mult = _scale(x).reciprocal()
    ref = (x * mult).to(torch.float8_e4m3fn)
    x8, _ = quantize(x, mult, torch.float8_e4m3fn)
    x8T, _ = quantize(x, mult, torch.float8_e4m3fn, transposed=True)
    d8, d8T, _ = quantize_dual(x, mult, torch.float8_e4m3fn)
    assert torch.equal(x8.view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(d8.view(torch.uint8), ref.view(torch.uint8))
    assert torch.equal(x8T.view(torch.uint8), d8T.view(torch.uint8))
    assert torch.equal(x8T.t().contiguous().view(torch.uint8), ref.view(torch.uint8))


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


def _fp8_gemms():
    """How many fp8 GEMMs the composed path has launched so far."""
    return int(_gemm().fp8_debug_stats()["gemm"])


@skip_no_fp8
def test_nested_disabled_region_redispatches_bf16():
    """A nested fp8_autocast(enabled=False) region temporarily restores the
    bf16 aten::linear path (torch's nested-disable semantics), and fp8
    resumes when it exits. Discriminated by the fp8 GEMM counter, not the
    grad_fn type name (the C++ node's Python name is an implementation
    detail)."""
    x, w = _linear()
    _gemm().fp8_reset()
    _gemm().fp8_debug_reset_stats()
    try:
        with fp8_autocast(enabled=True):
            F.linear(x, w)
            n_fp8 = _fp8_gemms()
            assert n_fp8 == 1, n_fp8
            with fp8_autocast(enabled=False):
                out_bf16 = F.linear(x, w)
                assert _fp8_gemms() == n_fp8  # no fp8 GEMM: bf16 fallback
                assert out_bf16.dtype == torch.bfloat16
            out_again = F.linear(x, w)
            assert _fp8_gemms() == n_fp8 + 1  # fp8 resumed
    finally:
        _gemm().fp8_reset()


@skip_no_fp8
def test_global_switch_routes_without_region():
    """fp8_linear_enable(True) routes aten::linear to fp8 outside any region
    (the persistent default); disabling restores bf16."""
    x, w = _linear()
    _gemm().fp8_reset()
    _gemm().fp8_debug_reset_stats()
    try:
        fp8_linear_enable(True)
        F.linear(x, w)
        n_fp8 = _fp8_gemms()
        assert n_fp8 == 1, n_fp8
        fp8_linear_enable(False)
        F.linear(x, w)
        assert _fp8_gemms() == n_fp8  # bf16 fallback, no fp8 GEMM
    finally:
        _gemm().fp8_reset()


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


# --------------------------------------------------------------------------
# Delayed-scaling plumbing: the kernel-published reciprocal and the
# version-keyed weight cast cache (host-overhead work, 2026-09-21)
# --------------------------------------------------------------------------


@skip_no_fp8
@pytest.mark.parametrize("fmt", ["hybrid", torch.float8_e5m2])
@pytest.mark.parametrize("margin", [0, 1])
def test_ring_publishes_reciprocal_bitwise(fmt, margin):
    """The fold publishes the next scale *and* its reciprocal in one block:
    the recip slot stays bit-identical to ``torch.reciprocal(scale)`` step
    after step and ring after ring. That identity is what lets the policy
    hand the kernels the ring's recip instead of computing one per call
    (``__frcp_rn`` and ATen's 1/x are both correctly rounded)."""
    torch.manual_seed(17)
    dev = torch.device("cuda")
    gemm = _gemm()
    gemm.fp8_reset()
    recipe = FP8Recipe(history_len=2, margin=margin)
    n, k = 16, 64
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.3
    w.requires_grad_(True)
    bias = torch.zeros(n, device=dev, dtype=torch.bfloat16)
    try:
        with fp8_autocast(enabled=True, recipe=recipe, fp8_format=fmt):
            for step, mag in enumerate((0.5, 0.9, 0.2, 0.7)):
                # Growing then shrinking amax: consecutive steps publish
                # different scales, so a stale recipient slot cannot hide.
                x = torch.randn(8 + step, k, device=dev, dtype=torch.bfloat16) * mag
                F.linear(x, w, bias).float().pow(2).sum().backward()
                meta = gemm.fp8_debug_meta(w, recipe.history_len, recipe.margin)
                for name in ("x", "w", "g"):
                    ring = meta[name]
                    assert ring["initialized"]
                    assert torch.equal(
                        ring["scale_recip"].flatten(),
                        torch.reciprocal(ring["scale"]).flatten(),
                    ), f"step {step} ring {name}: published reciprocal drifted"
    finally:
        gemm.fp8_reset()


@skip_no_fp8
def test_weight_cast_cache_reuses_and_invalidates():
    """The weight cast is reused while the weight's version counter is
    unchanged (one cast per optimizer step, not per micro-batch), and
    invalidated by an in-place update and a checkpoint restore. Reuse must
    be numerically inert: identical inputs under an unchanged weight give
    bit-identical outputs."""
    torch.manual_seed(23)
    dev = torch.device("cuda")
    gemm = _gemm()
    gemm.fp8_reset()
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    w = torch.randn(32, 64, device=dev, dtype=torch.bfloat16)
    bias = torch.zeros(32, device=dev, dtype=torch.bfloat16)
    try:

        def run():
            gemm.fp8_debug_reset_stats()
            return gemm.fp8_linear(
                x,
                w,
                bias,
                True,
                False,
                False,
                16,
                0,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            )

        o1 = run()
        # Casts: one dual pass for x and one for w — each yields both
        # orientations, the transposed side in the backward format.
        stats = gemm.fp8_debug_stats()
        assert stats["quantize"] == 2 and stats["cast_miss"] == 1, stats
        o2 = run()
        # Cache hit: only x is cast again.
        stats = gemm.fp8_debug_stats()
        assert stats["quantize"] == 1, stats
        assert stats["cast_hit"] == 1 and stats["cast_miss"] == 0, stats
        assert torch.equal(o1, o2)

        with torch.no_grad():
            w.add_(0.01)  # the optimizer-step shape: in-place => version bump
        o3 = run()
        stats = gemm.fp8_debug_stats()
        assert stats["quantize"] == 2 and stats["cast_miss"] == 1, stats
        assert not torch.equal(o2, o3)

        # The ring's history also folds on a miss only: a hit leaves both the
        # window and the write index untouched.
        meta = gemm.fp8_debug_meta(w)
        idx, hist = meta["w"]["idx"], meta["w"]["hist"].clone()
        run()
        stats = gemm.fp8_debug_stats()
        assert stats["quantize"] == 1 and stats["cast_hit"] == 1, stats
        meta = gemm.fp8_debug_meta(w)
        assert meta["w"]["idx"] == idx and torch.equal(meta["w"]["hist"], hist)
        assert meta["cast_version"] == w._version

        fp8_load_state_dict(fp8_state_dict())
        run()
        stats = gemm.fp8_debug_stats()
        assert stats["quantize"] == 2  # a restore re-publishes scales
    finally:
        gemm.fp8_reset()


@skip_no_fp8
@pytest.mark.parametrize(("m", "k"), [(64, 128), (17, 33), (31, 96), (8, 8), (40, 130)])
def test_quantize_dual_mixed_formats_bitwise(m, k):
    """A mixed-format dual pass (E4M3 row-major, E5M2 transposed — the hybrid
    training pair) is bit-identical to the two single-format casts it
    replaces: the conversions are elementwise, only the read is shared.
    Odd/unaligned shapes exercise the predicated boundary path."""
    torch.manual_seed(m + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    mult = (x.abs().amax().float() / 448.0).clamp_min(1e-12).reciprocal()
    d8, d8T, _ = quantize_dual(
        x, mult, torch.float8_e4m3fn, transposed_fmt=torch.float8_e5m2
    )
    ref8, _ = quantize(x, mult, torch.float8_e4m3fn)
    ref8T, _ = quantize(x, mult, torch.float8_e5m2, transposed=True)
    assert d8.dtype == torch.float8_e4m3fn and d8T.dtype == torch.float8_e5m2
    assert d8.shape == (m, k) and d8T.shape == (k, m)
    assert torch.equal(d8, ref8)
    assert torch.equal(d8T, ref8T)
