import pytest
import torch

from astrai.extension import attention_varlen, op_backend
from astrai.extension.runtime.loader import is_available


def compare_segments(device, dtype, dim, causal):
    torch.manual_seed(37)
    qs = torch.tensor([0, 2, 7, 24, 24], dtype=torch.int32, device=device)
    ks = torch.tensor([0, 5, 7, 7, 11], dtype=torch.int32, device=device)
    q, k, v = [
        torch.randn(n, h, dim, device=device, dtype=dtype)
        for n, h in ((24, 4), (11, 2), (11, 2))
    ]
    out = attention_varlen(q, k, v, cu_seqlens_q=qs, cu_seqlens_k=ks, is_causal=causal)
    ref_inputs = [t.float() for t in (q, k, v)]
    with op_backend(attention="torch"):
        reference = attention_varlen(
            *ref_inputs, cu_seqlens_q=qs, cu_seqlens_k=ks, is_causal=causal
        )
    torch.testing.assert_close(out.float(), reference, rtol=0.035, atol=0.045)
    assert not out[7:].count_nonzero()
    if causal:
        assert not out[2:5].count_nonzero()
    k[:5].zero_()
    v[:5].zero_()
    changed = attention_varlen(
        q, k, v, cu_seqlens_q=qs, cu_seqlens_k=ks, is_causal=causal
    )
    torch.testing.assert_close(out[2:], changed[2:], rtol=0, atol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_varlen_cpu(causal):
    compare_segments("cpu", torch.float32, 32, causal)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available("attention"),
    reason="native CUDA required",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [32, 64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_varlen_native(dtype, dim, causal):
    with op_backend(attention="cuda"):
        compare_segments("cuda", dtype, dim, causal)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available("attention"),
    reason="native CUDA required",
)
def test_varlen_shared_offsets_across_tiles():
    torch.manual_seed(5)
    ptr = torch.tensor([0, 75, 209, 311], device="cuda", dtype=torch.int32)
    x = [
        torch.randn(311, h, 64, device="cuda", dtype=torch.bfloat16) for h in (12, 2, 2)
    ]
    out = attention_varlen(*x, cu_seqlens_q=ptr, is_causal=True, scale=0.17)
    ref = [t.float() for t in x]
    with op_backend(attention="torch"):
        expected = attention_varlen(*ref, cu_seqlens_q=ptr, is_causal=True, scale=0.17)
    torch.testing.assert_close(out.float(), expected, rtol=0.04, atol=0.065)


def test_varlen_requires_grad():
    q = torch.ones(3, 1, 8, requires_grad=True)
    offsets = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="forward only"):
        attention_varlen(q, q, q, cu_seqlens_q=offsets)
    with torch.no_grad():
        assert attention_varlen(q, q, q, cu_seqlens_q=offsets).shape == q.shape


def test_varlen_invalid_endpoints():
    q = torch.ones(3, 1, 8)
    with pytest.raises(ValueError, match="invalid segment"):
        attention_varlen(q, q, q, cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32))
