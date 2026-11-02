"""Kernel-level mask dimension support (2D, 3D, 4D)."""

import math

import pytest
import torch

from astrai.extension.kernel.attention import attn_prefill
from tests.support.capabilities import skip_no_kernel
from tests.support.models import D


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


@pytest.mark.cuda
@skip_no_kernel
@pytest.mark.parametrize(
    "dimensions,all_true", [(2, False), (3, False), (4, False), (4, True)]
)
def test_mask_layouts_match_reference(dimensions, all_true):
    torch.manual_seed(9 + dimensions)
    q = torch.randn(1, 8, 4, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 1, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    shape = (1, 8) if dimensions == 2 else (1, 8, 8)
    mask = torch.ones(shape, dtype=torch.bool, device="cuda")
    if not all_true:
        if dimensions == 3:
            mask[:, 0, 5:] = False
            mask[:, 1, 6:] = False
        else:
            mask[..., 4:] = False
    if dimensions == 4:
        mask = mask[:, None]
    actual = attn_prefill(q, k, v, mask=mask, is_causal=False)
    if all_true:
        expected = attn_prefill(q, k, v, mask=None, is_causal=False)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        torch.testing.assert_close(
            actual, _reference(q, k, v, mask), atol=0.05, rtol=0.05
        )
