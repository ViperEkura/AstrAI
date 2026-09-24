"""Attention element-type contract.

The kernels are dtype-agnostic at the ABI (void* pointers, the element type
being a compile-time kernel parameter) and instantiated per element type; each
entry switches on q's ``at::ScalarType`` over the family's instantiation list
(``ASTRAI_ATTN_DTYPE_LIST``) and refuses anything else.  These tests pin the two
halves of that contract: the instantiated type runs, and any other precision
fails loudly instead of being reinterpreted as the instantiated one — the
failure mode that matters, since a void* carries no type of its own.
"""

import pytest
import torch

from astrai.extension.ops.attention import (
    attn_decode,
    attn_paged_decode,
    attn_paged_prefill,
    attn_prefill,
)
from tests.conftest import skip_no_kernel
from tests.extension.conftest import D

Q_LEN, KV_LEN, N_HEADS, N_KV_HEADS = 8, 16, 4, 2


def _contig(dtype):
    torch.manual_seed(3)
    q = torch.randn(1, Q_LEN, N_HEADS, D, device="cuda", dtype=dtype)
    k = torch.randn(1, KV_LEN, N_KV_HEADS, D, device="cuda", dtype=dtype)
    return q, k, k.clone()


def _paged(dtype):
    batch, seq_len, heads, kv_heads = 2, 32, N_HEADS, N_KV_HEADS
    torch.manual_seed(4)
    q = torch.randn(batch, heads, D, device="cuda", dtype=dtype)
    k = torch.randn(batch, kv_heads, D, device="cuda", dtype=dtype)
    k_cache = torch.randn(batch * seq_len, kv_heads, D, device="cuda", dtype=dtype)
    v_cache = torch.randn(batch * seq_len, kv_heads, D, device="cuda", dtype=dtype)
    req_to_token = torch.zeros(batch, seq_len, dtype=torch.int32, device="cuda")
    for i in range(batch):
        req_to_token[i] = torch.arange(
            i * seq_len, (i + 1) * seq_len, dtype=torch.int32, device="cuda"
        )
    req_pool = torch.arange(batch, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * seq_len
    return q, k, k.clone(), k_cache, v_cache, req_to_token, req_pool, kv_indptr


def _call_prefill(dtype):
    q, k, v = _contig(dtype)
    return attn_prefill(q, k, v, is_causal=True)


def _call_decode(dtype):
    q, k, v = _contig(dtype)
    return attn_decode(q[:, :1], k, v, is_causal=True)


def _call_paged_decode(dtype):
    q, k, v, k_cache, v_cache, rtt, pool, indptr = _paged(dtype)
    return attn_paged_decode(
        q, k_cache, v_cache, rtt, pool, indptr, new_k=k, new_v=v, is_causal=True
    )


ENTRIES = {
    "prefill": _call_prefill,
    "decode": _call_decode,
    "paged_decode": _call_paged_decode,
}


@skip_no_kernel
@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_instantiated_dtype_runs(entry):
    """The build's instantiated element type (bf16) runs and returns it."""
    out = ENTRIES[entry](torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


@skip_no_kernel
@pytest.mark.parametrize("entry", sorted(ENTRIES))
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_other_dtype_rejected(entry, dtype):
    """A precision with no kernel instantiation is refused, not reinterpreted.

    The switch the entry takes on q's scalar type decides the instantiation, so
    a missing entry must surface as an error naming what *is* instantiated —
    silently running the bf16 kernel on these bytes would return finite garbage.
    """
    with pytest.raises(
        RuntimeError, match=r"has no kernel for .*\(instantiated: BFloat16\)"
    ):
        ENTRIES[entry](dtype)
