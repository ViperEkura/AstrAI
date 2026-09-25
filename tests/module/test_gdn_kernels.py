"""Validation for the Gated DeltaNet CUDA kernels.

The preparation kernel and the output-stage backward are checked against torch
equivalents — the torch path for the layout work, autograd for the backward. The
reference operators the module trains on are covered by ``test_gdn_ops.py``.
"""

import pytest
import torch
import torch.nn.functional as F

from astrai.model.components.attention import GDN
from astrai.model.components.gdn_ops import _to_heads_fp32, l2norm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernels need CUDA"
)

CHUNK = 64


def chunk_local_cumsum(g, chunk_size=CHUNK):
    """Cumsum of the gate inside each chunk, on the ``[B, H, T]`` layout.

    Kept here rather than imported: this is the definition the kernel is checked
    against, and a shared helper would let a bug in it hide on both sides.
    """
    total = g.shape[-1]
    padded = -(-total // chunk_size) * chunk_size
    if padded != total:
        g = F.pad(g, (0, padded - total))
    shaped = g.reshape(*g.shape[:-1], padded // chunk_size, chunk_size)
    return shaped.cumsum(dim=-1).reshape(*g.shape[:-1], padded)


def prep_available():
    from astrai.extension.loader import is_available

    return torch.cuda.is_available() and is_available("gated_deltanet")


def make_prep_layer(device):
    """Head dim 128: the shape the CUDA preparation kernel is written for."""
    return (
        GDN(
            dim=256,
            n_heads=2,
            gdn_num_key_heads=2,
            gdn_num_value_heads=2,
            gdn_key_head_dim=128,
            gdn_value_head_dim=128,
            gdn_conv_kernel_size=4,
        )
        .to(device)
        .to(torch.bfloat16)
    )


@pytest.mark.skipif(not prep_available(), reason="gated deltanet kernel not built")
def test_prep_kernel_matches_the_torch_preparation(device):
    """Normalization and the gate scan, checked against the torch steps directly."""
    from astrai.extension.ops.gdn import gdn_fwd

    layer = make_prep_layer(device)
    x = torch.randn(1, 128, 256, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        q, k, v = (t.detach() for t in layer._project_qkv(x))
        g, beta = (t.detach() for t in layer._gates(x))
        q_hat, k_hat, v_hat, g_hat, beta_hat = gdn_fwd(q, k, v, g, beta, 1e-6, 64)
        q_ref = l2norm(_to_heads_fp32(q)).to(torch.bfloat16)
        k_ref = l2norm(_to_heads_fp32(k)).to(torch.bfloat16)
        v_ref = _to_heads_fp32(v).to(torch.bfloat16)
        g_ref = chunk_local_cumsum(_to_heads_fp32(g), CHUNK)
        beta_ref = _to_heads_fp32(beta)
    assert v_hat.shape == v_ref.shape and torch.equal(v_hat, v_ref)
    assert torch.equal(beta_hat, beta_ref)
    torch.testing.assert_close(g_hat, g_ref, atol=1e-5, rtol=1e-4)
    # bf16 output, so the tolerance is one output ULP: the norm's summation order
    # differs between the reduction tree and torch's, which flips the last bit.
    torch.testing.assert_close(q_hat, q_ref, atol=4e-3, rtol=4e-2)
    torch.testing.assert_close(k_hat, k_ref, atol=4e-3, rtol=4e-2)


@pytest.mark.skipif(not prep_available(), reason="gated deltanet kernel not built")
def test_prep_rejects_tensors_that_do_not_carry_the_projection_layout(device):
    """A contiguous [B, T, H, D] tensor would be read as the wrong elements."""
    from astrai.extension.ops.gdn import gdn_fwd

    layer = make_prep_layer(device)
    x = torch.randn(1, 64, 256, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        q, k, v = layer._project_qkv(x)
        g, beta = layer._gates(x)
        with pytest.raises(ValueError, match="projection layout"):
            gdn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), g, beta, 1e-6, 64)


@pytest.mark.skipif(not prep_available(), reason="gated deltanet kernel not built")
def test_gated_deltanet_bwd_matches_autograd(device):
    """The output-stage backward against autograd of the same stage.

    Every gradient is checked, and the case is chosen with a dense state so the
    two axes of h are distinguishable: a symmetric state hides an axis mix-up in
    the d_qe product, which is exactly the bug this test caught.
    """
    from astrai.extension.loader import get_module

    torch.manual_seed(0)
    batch, heads, seq_len, dim = 2, 2, 128, 128
    chunk = 64
    chunks = seq_len // chunk
    gen = torch.Generator().manual_seed(0)
    rnd = lambda *d: torch.randn(*d, generator=gen, dtype=torch.float32)

    q = rnd(batch, seq_len, heads, dim).to(device, torch.bfloat16)
    k = rnd(batch, seq_len, heads, dim).to(device, torch.bfloat16)
    g = (-0.5 * torch.rand(batch, seq_len, heads, generator=gen)).to(device)
    qh, kh = (_to_heads_fp32(t) for t in (q, k))
    gh = chunk_local_cumsum(_to_heads_fp32(g), chunk)
    qh, kh = l2norm(qh), l2norm(kh)
    scale = dim**-0.5
    h = (rnd(batch, heads, chunks, dim, dim) * 0.1).to(device, torch.bfloat16)
    v_new = (rnd(batch, heads, seq_len, dim) * 0.1).to(device, torch.bfloat16)
    do = (rnd(batch, heads, seq_len, dim) * 0.1).to(device, torch.bfloat16)

    gb = gh.reshape(batch, heads, chunks, chunk)
    decay = (gb.unsqueeze(-1) - gb.unsqueeze(-2)).tril().exp().tril()
    qr = qh.to(torch.bfloat16).float().requires_grad_(True)
    kr = kh.to(torch.bfloat16).float().requires_grad_(True)
    vr = v_new.float().clone().requires_grad_(True)
    hr = h.float().clone().requires_grad_(True)
    gr = gh.clone().requires_grad_(True)
    gbr = gr.reshape(batch, heads, chunks, chunk)
    dec = (gbr.unsqueeze(-1) - gbr.unsqueeze(-2)).tril().exp().tril()
    out = torch.zeros(batch, heads, seq_len, dim, device=device)
    for i in range(chunks):
        rows = slice(i * chunk, (i + 1) * chunk)
        attn = qr[:, :, rows] @ kr[:, :, rows].transpose(-1, -2) * dec[:, :, i]
        out[:, :, rows] = (
            (qr[:, :, rows] * gbr[:, :, i].exp()[..., None]) @ hr[:, :, i]
            + attn @ vr[:, :, rows]
        ) * scale
    torch.autograd.backward(out, do.float())

    dq, dk, dv, dh, dgh = get_module("gated_deltanet").gated_deltanet_bwd(
        qh.to(torch.bfloat16),
        kh.to(torch.bfloat16),
        v_new,
        h,
        gh.contiguous(),
        do,
        scale,
    )
    torch.cuda.synchronize()
    for name, got, want in (
        ("dq", dq, qr.grad),
        ("dk", dk, kr.grad),
        ("dv_new", dv, vr.grad),
        ("dh", dh, hr.grad.sum(2)),
        ("dgh", dgh, gr.grad),
    ):
        torch.testing.assert_close(
            got.float(),
            want.float(),
            atol=4e-3,
            rtol=4e-2,
            msg=lambda m, n=name: f"{n}: {m}",
        )
