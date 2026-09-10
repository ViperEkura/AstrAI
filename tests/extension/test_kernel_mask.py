"""Kernel-level mask dimension support (2D, 3D, 4D)."""

import math

import torch

from astrai.extension.ops.attention import attn_prefill
from tests.conftest import skip_no_kernel
from tests.extension.conftest import D


def _reference(q, k, v, mask):
    """fp32 masked GQA attention reference (True=keep)."""
    b, s_q, h, d = q.shape
    rep = h // k.shape[2]
    qf = q.float().transpose(1, 2)
    kf = k.float().repeat_interleave(rep, dim=2).transpose(1, 2)
    vf = v.float().repeat_interleave(rep, dim=2).transpose(1, 2)
    scores = qf @ kf.transpose(-1, -2) / math.sqrt(d)
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    # 4D [batch, 1, q_len, kv_len] broadcasts over heads as-is
    scores = scores.masked_fill(~mask, float("-inf"))
    return (scores.softmax(dim=-1) @ vf).transpose(1, 2).to(q.dtype)


@skip_no_kernel
def test_kernel_accepts_2d_mask():
    """2D mask [batch, kv_len] gates the softmax, not just parses."""
    torch.manual_seed(11)
    batch, q_len, n_heads, n_kv_heads = 1, 8, 4, 1
    kv_len = 8
    q = torch.randn(batch, q_len, n_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(batch, kv_len, dtype=torch.bool, device="cuda")
    mask[:, 4:] = False

    out = attn_prefill(q, k, v, mask=mask, is_causal=False)
    assert out.shape == (batch, q_len, n_heads, D)
    torch.testing.assert_close(out, _reference(q, k, v, mask), atol=0.05, rtol=0.05)


@skip_no_kernel
def test_kernel_accepts_3d_mask():
    """3D mask [batch, q_len, kv_len] applies per-query-row gating."""
    torch.manual_seed(12)
    batch, q_len, n_heads, n_kv_heads = 1, 8, 4, 1
    kv_len = 8
    q = torch.randn(batch, q_len, n_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(batch, q_len, kv_len, dtype=torch.bool, device="cuda")
    mask[:, 0, 5:] = False  # differs per query row: only the 3D path can apply it
    mask[:, 1, 6:] = False

    out = attn_prefill(q, k, v, mask=mask, is_causal=False)
    assert out.shape == (batch, q_len, n_heads, D)
    torch.testing.assert_close(out, _reference(q, k, v, mask), atol=0.05, rtol=0.05)


@skip_no_kernel
def test_kernel_accepts_4d_mask():
    """4D mask [batch, 1, q_len, kv_len] broadcasts over heads and gates."""
    torch.manual_seed(13)
    batch, q_len, n_heads, n_kv_heads = 1, 8, 4, 1
    kv_len = 8
    q = torch.randn(batch, q_len, n_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(batch, 1, q_len, kv_len, dtype=torch.bool, device="cuda")
    mask[:, :, :, 4:] = False

    out = attn_prefill(q, k, v, mask=mask, is_causal=False)
    assert out.shape == (batch, q_len, n_heads, D)
    torch.testing.assert_close(out, _reference(q, k, v, mask), atol=0.05, rtol=0.05)


@skip_no_kernel
def test_4d_mask_matches_no_mask_when_all_true():
    """A 4D all-True mask should produce the same output as no mask."""
    batch, q_len, n_heads, n_kv_heads = 1, 8, 4, 1
    kv_len = 8
    q = torch.randn(batch, q_len, n_heads, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, kv_len, n_kv_heads, D, device="cuda", dtype=torch.bfloat16)

    out_no_mask = attn_prefill(q, k, v, mask=None, is_causal=False)
    mask = torch.ones(batch, 1, q_len, kv_len, dtype=torch.bool, device="cuda")
    out_with_mask = attn_prefill(q, k, v, mask=mask, is_causal=False)

    diff = (out_no_mask.float() - out_with_mask.float()).abs().max().item()
    assert diff == 0.0, f"4D all-True mask diff: {diff}"
