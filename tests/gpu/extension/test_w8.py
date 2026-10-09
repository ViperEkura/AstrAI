"""Quantized-GEMM primitives: W8A16 / W8A8 / W16A16 kernel-level tests plus
the int8 policy layer (quantizers). Kernel tests exercise the stateless
wrappers in ``astrai.extension.kernel.gemm`` against torch references built
from the same quantized values; policy tests check the quantizers'
contracts.
"""

import pytest
import torch
import torch.nn.functional as F

from astrai.extension import plan, quantize_act_int8, quantize_weight_int8
from astrai.extension.kernel.gemm import quant_gemm
from tests.support.capabilities import skip_no_kernel

# bf16-output comparisons: the kernel's dequant and fp32 accumulation are
# exact, so the residual is output rounding plus accumulation-order
# differences against the torch reference — the same tolerance class the
# C harness uses.
ATOL, RTOL = 0.25, 0.02


def _rand(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m, k, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    return x, w


def _quant_weight(w):
    w8, ws = quantize_weight_int8(w)
    w_ref = (w8.float() * ws.unsqueeze(1)).to(torch.bfloat16)
    return w8, ws, w_ref


@skip_no_kernel
@pytest.mark.parametrize(("m", "n", "k"), [(64, 64, 64), (256, 384, 512), (17, 9, 33)])
class TestQuantGemm:
    def test_w16a16(self, m, n, k):
        x, w = _rand(m, n, k)
        out = quant_gemm(x, w)
        ref = F.linear(x, w)
        torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)

    def test_w16a16_layouts(self, m, n, k):
        x, w = _rand(m, n, k)
        ref = F.linear(x, w).float()
        base = quant_gemm(x, w)
        # crosswise activation storage [K][M] (trans_a)
        assert torch.allclose(
            quant_gemm(x.t().contiguous().t(), w).float(),
            base.float(),
            atol=ATOL,
            rtol=RTOL,
        )
        # transposed-view zero copy: w.t() storage [K][N], trans_b=False
        out_t = quant_gemm(x, w.t(), trans_b=False)
        torch.testing.assert_close(out_t.float(), ref, atol=ATOL, rtol=RTOL)
        # crosswise weight storage [K][N] (trans_b=False, contiguous)
        out_c = quant_gemm(x, w.t().contiguous(), trans_b=False)
        torch.testing.assert_close(out_c.float(), ref, atol=ATOL, rtol=RTOL)

    def test_w8a16(self, m, n, k):
        x, w = _rand(m, n, k)
        w8, ws, w_ref = _quant_weight(w)
        out = quant_gemm(x, w8, b_scale=ws)
        ref = F.linear(x, w_ref)
        torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)

    def test_w8a16_per_tensor_scale(self, m, n, k):
        x, w = _rand(m, n, k)
        w8, _, _ = _quant_weight(w)
        s = (w.float().abs().amax() / 127.0).reshape(1).float()
        out = quant_gemm(x, w8, b_scale=s)
        # Rebuild the reference with the scalar scale.
        w_pt = (w8.float() * s.item()).to(torch.bfloat16)
        ref = F.linear(x, w_pt)
        torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)

    def test_w8a8(self, m, n, k):
        x, w = _rand(m, n, k)
        w8, ws, w_ref = _quant_weight(w)
        x8, xs = quantize_act_int8(x)
        x_ref = (x8.float() * xs.unsqueeze(1)).reshape(x.shape).to(torch.bfloat16)
        out = quant_gemm(x8.reshape(m, k), w8, a_scale=xs, b_scale=ws)
        ref = F.linear(x_ref, w_ref)
        torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)

    def test_w8a8_per_tensor_activation_scale(self, m, n, k):
        x, w = _rand(m, n, k)
        w8, ws, w_ref = _quant_weight(w)
        s = (x.float().abs().amax() / 127.0).reshape(1).float()
        x8 = torch.round(x.float() / s.item()).clamp_(-127, 127).to(torch.int8)
        x_ref = (x8.float() * s.item()).to(torch.bfloat16)
        out = quant_gemm(x8, w8, a_scale=s, b_scale=ws)
        ref = F.linear(x_ref, w_ref)
        torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)

    def test_bias_fusion(self, m, n, k):
        x, w = _rand(m, n, k)
        w8, ws, w_ref = _quant_weight(w)
        bias = torch.randn(n, device="cuda").to(torch.bfloat16)
        fused = quant_gemm(x, w8, b_scale=ws, bias=bias)
        ref = F.linear(x, w_ref).float() + bias.float()
        torch.testing.assert_close(fused.float(), ref, atol=ATOL, rtol=RTOL)

    def test_batched_broadcast(self, m, n, k):
        x, w = _rand(m, n, k)
        w8, ws, w_ref = _quant_weight(w)
        xb = x.unsqueeze(0).expand(3, -1, -1)
        out = quant_gemm(xb, w8, b_scale=ws)
        assert out.shape == (3, m, n)
        ref = F.linear(x, w_ref)
        for b in range(3):
            torch.testing.assert_close(
                out[b].float(), ref.float(), atol=ATOL, rtol=RTOL
            )


@skip_no_kernel
@pytest.mark.parametrize(
    ("trans_a", "trans_b"), [(False, True), (True, True), (False, False), (True, False)]
)
@pytest.mark.parametrize("k", [127, 128, 129])
def test_w8a8_k128_swizzle_and_tail(trans_a, trans_b, k):
    m, n = 65, 129
    x = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    w = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
    xs = torch.ones(m, device="cuda")
    ws = torch.ones(n, device="cuda")
    a = x.T.contiguous() if trans_a else x
    b = w if trans_b else w.T.contiguous()
    with plan.override(planner="hybrid", rows="0 0 0 0 2 -1 2 2 0 128"):
        selected = plan.probe(m, n, k, torch.int8, torch.int8, trans_a, trans_b)
        assert (selected.source, selected.k_tile) == ("override", 128)
        out = quant_gemm(a, b, xs, ws, trans_a, trans_b)
    ref = ((x.float() @ w.float().T) * xs[:, None] * ws[None, :]).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@skip_no_kernel
class TestQuantGemmValidation:
    def test_unsupported_dtype_pair_rejected(self):
        x = torch.randn(8, 8, device="cuda", dtype=torch.float16)
        w = torch.randn(8, 8, device="cuda", dtype=torch.float16)
        # The dispatch switch's default arm is the single validator; its
        # message names the unsupported pair and the supported set.
        with pytest.raises(RuntimeError, match="unsupported operand dtype pair"):
            quant_gemm(x, w)

    def test_int8_without_scale_rejected(self):
        x8 = torch.zeros(8, 8, device="cuda", dtype=torch.int8)
        w8 = torch.zeros(8, 8, device="cuda", dtype=torch.int8)
        s = torch.ones(8, device="cuda")
        with pytest.raises(RuntimeError, match="a_scale is required"):
            quant_gemm(x8, w8, b_scale=s)

    def test_bf16_with_scale_rejected(self):
        x = torch.randn(8, 8, device="cuda").to(torch.bfloat16)
        w = torch.randn(8, 8, device="cuda").to(torch.bfloat16)
        s = torch.ones(8, device="cuda")
        with pytest.raises(RuntimeError, match="bf16 operand"):
            quant_gemm(x, w, b_scale=s)

    def test_scale_extent_rejected(self):
        x = torch.randn(8, 16, device="cuda").to(torch.bfloat16)
        w8 = torch.zeros(8, 16, device="cuda", dtype=torch.int8)
        bad = torch.ones(7, device="cuda")  # n=8
        with pytest.raises(RuntimeError, match="b_scale"):
            quant_gemm(x, w8, b_scale=bad)

    def test_scale_dtype_rejected(self):
        x = torch.randn(8, 16, device="cuda").to(torch.bfloat16)
        w8 = torch.zeros(8, 16, device="cuda", dtype=torch.int8)
        bad = torch.ones(8, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="float32"):
            quant_gemm(x, w8, b_scale=bad)

    def test_rank_rejected_before_shape_access(self):
        x = torch.ones(8, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(8, 8, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="2D or 3D"):
            quant_gemm(x, w)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
    def test_scale_and_bias_must_share_operand_device(self):
        x = torch.ones(8, 8, device="cuda:0", dtype=torch.bfloat16)
        w8 = torch.ones(8, 8, device="cuda:0", dtype=torch.int8)
        other_scale = torch.ones(8, device="cuda:1")
        with pytest.raises(RuntimeError, match="same device"):
            quant_gemm(x, w8, b_scale=other_scale)

        w = torch.ones(8, 8, device="cuda:0", dtype=torch.bfloat16)
        other_bias = torch.ones(8, device="cuda:1", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="same device"):
            quant_gemm(x, w, bias=other_bias)


@skip_no_kernel
class TestQuantizers:
    def test_weight_quantizer_contract(self):
        w = (torch.randn(64, 128, device="cuda") * 0.5).to(torch.bfloat16)
        w8, ws = quantize_weight_int8(w)
        assert w8.dtype == torch.int8 and w8.shape == w.shape
        assert w8.is_contiguous()
        assert ws.dtype == torch.float32 and ws.shape == (64,)
        deq = w8.float() * ws.unsqueeze(1)
        # round-to-nearest per channel: every element sits within half a
        # quantization step (plus fp32 division slop) of its source.
        err = (deq - w.float()).abs()
        assert bool((err <= ws.unsqueeze(1) * 0.5 * 1.001 + 1e-9).all())

    def test_act_quantizer_contract(self):
        x = (torch.randn(3, 5, 64, device="cuda") * 0.5).to(torch.bfloat16)
        q, s = quantize_act_int8(x)
        assert q.dtype == torch.int8 and q.shape == x.shape
        assert s.dtype == torch.float32 and s.shape == (15,)
        assert q.int().abs().max().item() <= 127


@skip_no_kernel
class TestQuantGemmPositiveGeometry:
    """Reject zero dimensions before allocation or planner selection."""

    @pytest.mark.parametrize("mode", ["hybrid", "table", "heuristic", "model"])
    @pytest.mark.parametrize("shape", [(0, 64, 32), (8, 0, 32), (8, 64, 0)])
    @pytest.mark.parametrize("batched", [False, True])
    def test_zero_dimension_rejected(self, mode, shape, batched):
        from astrai.extension import plan

        m, n, k = shape
        prefix = (3,) if batched else ()
        a = torch.empty((*prefix, m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
        with plan.override(planner=mode, table_off=True):
            with pytest.raises(
                RuntimeError, match="M, N and K must be greater than zero"
            ):
                quant_gemm(a, b)

    def test_zero_k_with_bias_rejected(self):
        a = torch.empty((4, 0), device="cuda", dtype=torch.bfloat16)
        b = torch.empty((5, 0), device="cuda", dtype=torch.bfloat16)
        bias = torch.arange(5, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="M, N and K must be greater than zero"):
            quant_gemm(a, b, bias=bias)
