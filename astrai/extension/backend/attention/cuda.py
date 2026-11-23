"""Native dense attention and fixed-split rollout autograd."""

import torch
from torch.autograd.function import once_differentiable

from astrai.extension.backend.attention import torch as torch_attention
from astrai.extension.kernel import attention as kernel
from astrai.extension.kernel.attention import ROLLOUT_KV_SPLIT_SIZE
from astrai.extension.runtime.loader import is_available

HEAD_DIMS = (32, 64, 128, 256)


def supports(spec):
    return (
        spec.device.type == "cuda"
        and spec.dtype in (torch.bfloat16, torch.float16)
        and spec.head_dim in HEAD_DIMS
        and (spec.scale is None or spec.scale > 0)
        and (
            spec.mask_kind == "none"
            or (spec.mask_kind == str(torch.bool) and spec.mask_last_stride == 1)
        )
        and is_available("attention")
    )


def forward(q, k, v, mask=None, is_causal=False, *, scale=None):
    return kernel.attention(q, k, v, mask, is_causal, scale=scale)


class _ConsistentPrefill(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask, is_causal, scale):
        ctx.save_for_backward(q, k, v, mask)
        ctx.is_causal = is_causal
        ctx.scale = scale
        return kernel.attention(
            q, k, v, mask, is_causal, scale=scale, kv_split_size=ROLLOUT_KV_SPLIT_SIZE
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, mask = ctx.saved_tensors
        # Differentiate scaled dot-product attention with the same mask and
        # scale. The reference backward is independent of the tile schedule
        # and does not differentiate intermediate BF16 rounding.
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_(True) for t in (q, k, v))
            result = torch_attention.forward(
                *inputs, mask, ctx.is_causal, scale=ctx.scale
            )
            grads = torch.autograd.grad(result, inputs, grad_output)
        return *grads, None, None, None


def forward_consistent(q, k, v, mask=None, is_causal=False, *, scale=None):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _ConsistentPrefill.apply(q, k, v, mask, is_causal, scale)
    return kernel.attention(
        q, k, v, mask, is_causal, scale=scale, kv_split_size=ROLLOUT_KV_SPLIT_SIZE
    )


def forward_varlen(q, k, v, qs, ks, is_causal, scale):
    return kernel.attention_varlen(q, k, v, qs, ks, is_causal, scale)


def forward_cache(
    q, kc, vc, k, v, past, pages, qs, mask, causal, workspace=(None, None, None)
):
    return kernel.attention_with_kvcache(
        q, kc, vc, k, v, past, pages, qs, mask, causal, workspace=workspace
    )


def forward_cache_consistent(
    q, kc, vc, k, v, past, pages, qs, mask, causal, workspace=(None, None, None)
):
    return kernel.attention_with_kvcache(
        q,
        kc,
        vc,
        k,
        v,
        past,
        pages,
        qs,
        mask,
        causal,
        workspace=workspace,
        kv_split_size=ROLLOUT_KV_SPLIT_SIZE,
    )
