"""FP8 training-quality gate: the fp8 path must stay numerically faithful.

Every other test checks semantics (bitwise parity of a refactor, ring
lifecycle, kernel boundaries). This file is the qualitative net: the fp8
training path must keep producing outputs and gradients close to bf16 —
tight on single steps (where no chaos amplifies the quantization error),
loose on a real training trajectory (where the trajectory itself diverges
legitimately; only collapse/NaN is a failure).

Thresholds were set from measured values with ~2x margin (2026-09-21,
production shapes: single-step out rel-fro 3.8e-2 / cos 0.9993, grad cos
0.998 / norm ratio 0.994, 30-step AdamW final-loss ratio 1.11):
- single-step out: rel-fro < 8e-2, cos > 0.995
- gradients: cos > 0.99, norm ratio in [0.95, 1.05]
- training smoke: final loss < 10% of the first, fp8/bf16 final ratio in
  [0.5, 2.0], everything finite (the disaster detectors).
"""

import pytest
import torch
import torch.nn.functional as F

from astrai.extension import quantize as f8mod
from tests.conftest import skip_no_fp8

DEV = "cuda"


@pytest.fixture(autouse=True)
def _clean_fp8_state():
    f8mod.fp8_reset()
    yield
    f8mod.fp8_reset()


def _operands(seed=1, n=1536, k=1536, m=2048, x_scale=0.5):
    torch.manual_seed(seed)
    w = (torch.randn(n, k, device=DEV, dtype=torch.bfloat16) * 0.02).requires_grad_()
    x = (torch.randn(m, k, device=DEV, dtype=torch.bfloat16) * x_scale).requires_grad_()
    return x, w


def _fwd_bwd(x, w, fp8):
    # Fresh leaves: clones of a leaf are non-leaves and keep no .grad.
    x = x.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    if fp8:
        with f8mod.fp8_autocast(enabled=True):
            out = F.linear(x, w)
    else:
        out = F.linear(x, w)
    out.float().pow(2).sum().backward()
    return out.detach(), x.grad.clone(), w.grad.clone()


@skip_no_fp8
def test_single_step_output_parity():
    """One fp8 step on production-shape operands keeps the output close to
    bf16: relative Frobenius error and cosine within the measured envelope
    (the e4m3 cast accumulates ~4% over a K=1536 contraction)."""
    x, w = _operands()
    ref, _, _ = _fwd_bwd(x, w, fp8=False)
    got, _, _ = _fwd_bwd(x, w, fp8=True)

    rel = ((ref.float() - got.float()).norm() / ref.float().norm()).item()
    cos = F.cosine_similarity(
        ref.float().flatten(), got.float().flatten(), dim=0
    ).item()
    assert rel < 8e-2, f"single-step relative error {rel:.4f} exceeds the envelope"
    assert cos > 0.995, f"single-step cosine {cos:.5f} below the envelope"


@skip_no_fp8
def test_single_step_gradients_parity():
    """dX and dW keep their direction and scale: cosine > 0.99, norm ratio
    within +-5% (the e5m2 backward format is coarser than e4m3, so this is
    deliberately looser than an elementwise comparison but still catches a
    broken quantize, a lost scale or a swapped operand)."""
    x, w = _operands()
    _, gx_ref, gw_ref = _fwd_bwd(x, w, fp8=False)
    _, gx, gw = _fwd_bwd(x, w, fp8=True)

    for name, a, b in (("grad_x", gx, gx_ref), ("grad_w", gw, gw_ref)):
        assert torch.isfinite(a.float()).all(), f"{name} non-finite"
        cos = F.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()
        ratio = (a.float().norm() / b.float().norm()).item()
        assert cos > 0.99, f"{name} cosine {cos:.5f} below the envelope"
        assert 0.95 < ratio < 1.05, f"{name} norm ratio {ratio:.4f} drifted"


@skip_no_fp8
def test_training_smoke_no_divergence():
    """A 30-step AdamW trajectory on a 3-layer MLP must learn under fp8 and
    stay in the same basin as bf16 — the disaster detector (NaN, collapse,
    or a scale blow-up), not a trajectory-equality check: quantization noise
    is legitimately amplified by the optimizer over 30 steps."""
    steps = 30

    def run(fp8):
        torch.manual_seed(0)
        layers = [
            torch.nn.Linear(512, 1024, bias=False, device=DEV, dtype=torch.bfloat16),
            torch.nn.Linear(1024, 1024, bias=False, device=DEV, dtype=torch.bfloat16),
            torch.nn.Linear(1024, 512, bias=False, device=DEV, dtype=torch.bfloat16),
        ]
        torch.manual_seed(100)
        x = torch.randn(64, 512, device=DEV, dtype=torch.bfloat16)
        torch.manual_seed(200)
        target = x @ (torch.randn(512, 512, device=DEV, dtype=torch.bfloat16) * 0.1)
        opt = torch.optim.AdamW(
            [p for layer in layers for p in layer.parameters()], lr=3e-3
        )

        def forward(inp):
            h = F.gelu(F.linear(inp, layers[0].weight))
            h = F.gelu(F.linear(h, layers[1].weight))
            return F.linear(h, layers[2].weight)

        losses = []
        for _ in range(steps):
            if fp8:
                with f8mod.fp8_autocast(enabled=True):
                    out = forward(x)
            else:
                out = forward(x)
            loss = F.mse_loss(out.float(), target.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    bf16 = run(False)
    fp8 = run(True)

    assert all(map(torch.isfinite, torch.tensor(fp8))), "fp8 loss went non-finite"
    assert fp8[-1] < 0.1 * fp8[0], f"fp8 did not learn: {fp8[0]:.3f} -> {fp8[-1]:.3f}"
    ratio = fp8[-1] / bf16[-1]
    assert 0.5 < ratio < 2.0, (
        f"fp8 diverged from the bf16 basin: {fp8[-1]:.4f} vs {bf16[-1]:.4f}"
    )
