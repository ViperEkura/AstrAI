"""Packed attention layout and PyTorch varlen adapter regressions."""

from types import SimpleNamespace

import pytest
import torch

from astrai.attention_layout import VarlenAttentionLayout
from astrai.extension.backend.attention import torch as native
from astrai.extension.backend.attention import varlen
from astrai.extension.backend.attention.dispatcher import attention


def _layout(device: str) -> VarlenAttentionLayout:
    return VarlenAttentionLayout(
        cu_seqlens=torch.tensor([0, 3, 5, 7], dtype=torch.int32, device=device),
        indices=torch.tensor([0, 1, 2, 3, 4, 5, 6], device=device),
        max_seqlen=3,
        position_ids=torch.tensor([[0, 1, 2, 0, 1], [0, 1, 0, 0, 0]], device=device),
    )


def _reference_varlen(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
    assert cu_q.tolist() == cu_k.tolist() == [0, 3, 5, 7]
    assert (max_q, max_k) == (3, 3)
    assert kwargs["window_size"] == (-1, 0)
    assert kwargs["enable_gqa"]
    outputs = []
    for start, end in zip(cu_q[:-1].tolist(), cu_q[1:].tolist()):
        keys = k[start:end].repeat_interleave(2, dim=1)
        values = v[start:end].repeat_interleave(2, dim=1)
        queries = q[start:end].transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)
        scale = kwargs["scale"] or queries.size(-1) ** -0.5
        scores = (queries @ keys.transpose(-2, -1)) * scale
        causal = torch.ones(end - start, end - start, dtype=torch.bool).tril()
        probs = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        outputs.append((probs @ values).transpose(0, 1))
    return torch.cat(outputs)


def test_4d_varlen_gather_scatter_and_gradients(monkeypatch):
    monkeypatch.setattr(varlen, "varlen_attn", _reference_varlen)
    layout = _layout("cpu")
    torch.manual_seed(7)
    q = torch.randn(2, 5, 2, 8, requires_grad=True)
    k = torch.randn(2, 5, 1, 8, requires_grad=True)
    v = torch.randn(2, 5, 1, 8, requires_grad=True)
    actual = varlen.forward_4d(q, k, v, layout, None)
    expected = native.forward(q, k, v, None, 0, layout)
    valid = layout.indices
    torch.testing.assert_close(
        actual.flatten(0, 1).index_select(0, valid),
        expected.flatten(0, 1).index_select(0, valid),
    )
    assert not actual.flatten(0, 1)[7:].any()
    actual.square().sum().backward(retain_graph=True)
    grads = [tensor.grad.clone() for tensor in (q, k, v)]
    for tensor in (q, k, v):
        tensor.grad = None
    expected.flatten(0, 1).index_select(0, valid).square().sum().backward()
    for grad, tensor in zip(grads, (q, k, v)):
        torch.testing.assert_close(grad, tensor.grad)


def test_3d_varlen_uses_kv_cache_offsets(monkeypatch):
    captured = {}

    def record(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
        captured.update(
            k=k, cu_q=cu_q, cu_k=cu_k, max_q=max_q, max_k=max_k, kwargs=kwargs
        )
        return q

    monkeypatch.setattr(varlen, "varlen_attn", record)
    q = torch.randn(3, 2, 8)
    k = torch.randn(3, 1, 8)
    v = torch.randn(3, 1, 8)
    cache = SimpleNamespace(
        k_buffer=torch.zeros(1, 5, 1, 8),
        v_buffer=torch.zeros(1, 5, 1, 8),
        out_cache_loc=torch.tensor([0, 1, 3]),
        req_to_token=torch.tensor([[0, 1, 2], [3, 4, 0]]),
        req_pool_indices=torch.tensor([0, 1]),
        max_len=3,
        seq_lens=torch.tensor([3, 2]),
        qo_indptr=torch.tensor([0, 2, 3], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    out = varlen.forward_3d(q, k, v, cache, 0, True, None)
    torch.testing.assert_close(out, q)
    torch.testing.assert_close(captured["k"][[0, 1, 3]], k)
    assert captured["cu_q"].tolist() == [0, 2, 3]
    assert captured["cu_k"].tolist() == [0, 3, 5]
    assert (captured["max_q"], captured["max_k"]) == (2, 3)
    assert captured["kwargs"]["window_size"] == (-1, 0)


@pytest.mark.parametrize("head_dim", [0, 8, 30, 32, 192, 256, 264, 320])
@pytest.mark.parametrize("fwd", [None, "prefill", "decode"])
def test_varlen_supports_head_dimension(monkeypatch, head_dim, fwd):
    monkeypatch.setattr(
        torch.backends.cuda, "is_flash_attention_available", lambda: True
    )
    spec = SimpleNamespace(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        head_dim=head_dim,
        mask_kind="varlen" if fwd is None else "none",
        ndim=4 if fwd is None else 3,
        fwd=fwd,
        kv_cache_kind="none" if fwd is None else "KVCache",
    )
    assert varlen.supports(spec) == (head_dim in (8, 32, 192, 256))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_dim", [32, 256, 36, 320])
def test_builtin_varlen_matches_dense_sft_attention(monkeypatch, head_dim):
    if not torch.backends.cuda.is_flash_attention_available():
        pytest.skip("PyTorch FlashAttention kernel unavailable")

    def no_compile(*args, **kwargs):
        raise AssertionError("varlen attention must not call torch.compile")

    monkeypatch.setattr(torch, "compile", no_compile)
    layout = _layout("cuda")
    torch.manual_seed(9)
    q = torch.randn(
        2, 5, 2, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        2, 5, 1, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        2, 5, 1, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    actual = attention(q, k, v, attn_mask=layout)
    expected = native.forward(q, k, v, None, 0, layout)
    valid = layout.indices
    torch.testing.assert_close(
        actual.flatten(0, 1).index_select(0, valid),
        expected.flatten(0, 1).index_select(0, valid),
        rtol=2e-2,
        atol=2e-2,
    )
    actual.flatten(0, 1).index_select(0, valid).float().square().sum().backward(
        retain_graph=True
    )
    grads = [tensor.grad.float().clone() for tensor in (q, k, v)]
    for tensor in (q, k, v):
        tensor.grad = None
    expected.flatten(0, 1).index_select(0, valid).float().square().sum().backward()
    for grad, tensor in zip(grads, (q, k, v)):
        torch.testing.assert_close(grad, tensor.grad.float(), rtol=5e-2, atol=5e-2)
