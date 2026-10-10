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

from astrai.extension.kernel.attention import (
    attn_decode,
    attn_paged_decode,
    attn_paged_prefill,
    attn_prefill,
)
from tests.support.capabilities import skip_no_kernel
from tests.support.models import D

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


@skip_no_kernel
def test_uninstantiated_head_dim_raises():
    q = torch.zeros(1, 1, N_HEADS, 96, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, KV_LEN, N_KV_HEADS, 96, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="head_dim 96 has no kernel instantiation"):
        attn_decode(q, k, k)


def _reference(q, k, v, mask=None, causal=False, scale=None):
    # Explicit lower-right causality also covers queries longer than K/V.
    heads = q.size(2) // k.size(2)
    k = k.repeat_interleave(heads, dim=2)
    v = v.repeat_interleave(heads, dim=2)
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
    if causal:
        allowed = (
            torch.arange(q.size(1), device=q.device)[:, None] + k.size(1) - q.size(1)
            >= torch.arange(k.size(1), device=q.device)[None, :]
        )
        mask = allowed if mask is None else allowed & mask
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        scale=scale,
    ).transpose(1, 2)


@skip_no_kernel
@pytest.mark.parametrize("q_len,kv_len", [(13, 7), (7, 13), (7, 7), (137, 7)])
@pytest.mark.parametrize("causal", [False, True])
def test_prefill_signed_query_positions(q_len, kv_len, causal):
    torch.manual_seed(19)
    q = torch.randn(2, q_len, 6, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, kv_len, 2, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    out = attn_prefill(q, k, v, is_causal=causal, scale=0.17)
    expected = _reference(q, k, v, causal=causal, scale=0.17)
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    if causal and q_len > kv_len:
        assert torch.count_nonzero(out[:, : q_len - kv_len]) == 0


@skip_no_kernel
@pytest.mark.parametrize("kv_len", [15, 127, 513])
def test_decode_mask_broadcast_and_temporary_workspace(kv_len):
    torch.manual_seed(20)
    # Preserve noncontiguous batch/head strides in the output.
    q = torch.randn(2, 1, 12, 64, device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    k = torch.randn(2, kv_len, 2, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    mask = torch.ones(1, 6, 1, kv_len, device="cuda", dtype=torch.bool)
    mask[:, 0] = False
    mask[..., 2::3] = False
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.current_stream())
        out = attn_decode(q, k, v, mask=mask, scale=0.13)
        scratch = torch.empty(2, 6, 32, 64, device="cuda", dtype=torch.float32)
        scratch.fill_(999)
    torch.cuda.current_stream().wait_stream(stream)
    expected = _reference(q, k, v, mask=mask, scale=0.13)
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
    assert torch.count_nonzero(out[:, :, 0]) == 0


@skip_no_kernel
@pytest.mark.parametrize("dim,width", [(64, 31), (256, 31), (64, 513), (256, 513)])
def test_paged_decode_graph_replays_split_append(dim, width):
    torch.manual_seed(21)
    batch, heads, kv_heads = 2, 6, 2
    length_cases = ((15, 7), (9, 13)) if width == 31 else ((257, 499), (479, 193))
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(batch * width, kv_heads, dim, device="cuda", dtype=q.dtype)
    v_cache = torch.randn_like(k_cache)
    table = torch.arange(batch * width, device="cuda", dtype=torch.int32).view(
        batch, width
    )
    requests = torch.arange(batch, device="cuda", dtype=torch.int32)
    indptr = torch.tensor(
        [0, length_cases[0][0], sum(length_cases[0])], device="cuda", dtype=torch.int32
    )
    new_k = torch.randn(batch, kv_heads, dim, device="cuda", dtype=q.dtype)
    new_v = torch.randn_like(new_k)
    mask = torch.ones(1, width, device="cuda", dtype=torch.bool)
    mask[:, 1::4] = False
    output = torch.empty_like(q)

    def call():
        return attn_paged_decode(
            q,
            k_cache,
            v_cache,
            table,
            requests,
            indptr,
            new_k=new_k,
            new_v=new_v,
            mask=mask,
            is_causal=False,
            out_buf=output,
            scale=0.13,
        )

    warm = torch.cuda.Stream()
    with torch.cuda.stream(warm):
        warm.wait_stream(torch.cuda.current_stream())
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for lengths in length_cases:
        indptr.copy_(
            torch.tensor(
                [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
            )
        )
        new_k.normal_()
        new_v.normal_()
        graph.replay()
        for b, length in enumerate(lengths):
            slot = b * width + length - 1
            torch.testing.assert_close(k_cache[slot], new_k[b], rtol=0, atol=0)
            torch.testing.assert_close(v_cache[slot], new_v[b], rtol=0, atol=0)
            expected = _reference(
                q[b : b + 1, None],
                k_cache[b * width : b * width + length][None],
                v_cache[b * width : b * width + length][None],
                mask=mask[:, :length],
                scale=0.13,
            )
            torch.testing.assert_close(
                captured[b], expected[0, 0], atol=0.02, rtol=0.02
            )


@skip_no_kernel
@pytest.mark.parametrize("dim,length,masked", [(64, 13, False), (256, 193, True)])
@pytest.mark.parametrize("layout", ["aligned", "k_offset", "v_offset", "outer_strides"])
def test_paged_decode_append_strided_sources(dim, length, masked, layout):
    torch.manual_seed(24)
    batch, heads, kv_heads, width = 2, 6, 2, length + 7
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    lengths = (length, length - 6)
    head_stride = dim + (1 if layout == "outer_strides" else 8)
    batch_stride = kv_heads * head_stride + (1 if layout == "outer_strides" else 8)
    k_offset, v_offset = {
        "aligned": (8, 16),
        "k_offset": (1, 8),
        "v_offset": (8, 1),
        "outer_strides": (0, 0),
    }[layout]

    def source(offset):
        storage_size = offset + batch_stride + (kv_heads - 1) * head_stride + dim
        storage = torch.randn(storage_size, device="cuda", dtype=q.dtype)
        return storage.as_strided(
            (batch, kv_heads, dim), (batch_stride, head_stride, 1), offset
        )

    new_k, new_v = source(k_offset), source(v_offset)
    pool_size = batch * width + 7
    k_cache = torch.randn(pool_size, kv_heads, dim, device="cuda", dtype=q.dtype)
    v_cache = torch.randn_like(k_cache)
    table = torch.randperm(pool_size, device="cuda")[: batch * width].to(torch.int32)
    table = table.view(batch, width)
    requests = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    indptr = torch.tensor(
        [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
    )
    expected_k, expected_v = k_cache.clone(), v_cache.clone()
    for b, count in enumerate(lengths):
        slot = table[requests[b], count - 1]
        expected_k[slot] = new_k[b]
        expected_v[slot] = new_v[b]
    mask = None
    if masked:
        mask = torch.ones(batch, width, device="cuda", dtype=torch.bool)
        mask[:, 1::3] = False

    out = attn_paged_decode(
        q,
        k_cache,
        v_cache,
        table,
        requests,
        indptr,
        new_k=new_k,
        new_v=new_v,
        mask=mask,
        scale=0.17,
    )

    # Check every cache slot: only the two append destinations may change.
    torch.testing.assert_close(k_cache, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(v_cache, expected_v, atol=0, rtol=0)
    for b, count in enumerate(lengths):
        slots = table[requests[b], :count].long()
        expected = _reference(
            q[b : b + 1, None],
            expected_k[slots][None],
            expected_v[slots][None],
            mask=None if mask is None else mask[b : b + 1, :count],
            scale=0.17,
        )
        torch.testing.assert_close(out[b], expected[0, 0], atol=0.02, rtol=0.02)


@skip_no_kernel
@pytest.mark.parametrize("unaligned_cache", ["k", "v"])
def test_paged_decode_single_token_unaligned_cache_graph(unaligned_cache):
    torch.manual_seed(26)
    heads, kv_heads, dim, pool_size, slot = 6, 2, 64, 3, 1
    q = torch.randn(1, heads, dim, device="cuda", dtype=torch.bfloat16)
    new_k = torch.randn(1, kv_heads, dim, device="cuda", dtype=q.dtype)
    new_v = torch.randn_like(new_k)
    offsets = (1, 8) if unaligned_cache == "k" else (8, 1)
    pool_elements = pool_size * kv_heads * dim
    storage = [
        torch.full((pool_elements + 16,), value, device="cuda", dtype=q.dtype)
        for value in (-11, 13)
    ]
    caches = [
        backing[offset : offset + pool_elements].view(pool_size, kv_heads, dim)
        for backing, offset in zip(storage, offsets)
    ]
    expected_storage = [backing.clone() for backing in storage]
    table = torch.tensor([[slot]], device="cuda", dtype=torch.int32)
    requests = torch.zeros(1, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)

    def call():
        return attn_paged_decode(
            q,
            caches[0],
            caches[1],
            table,
            requests,
            indptr,
            new_k=new_k,
            new_v=new_v,
            out_buf=output,
            scale=0.17,
        )

    # A single appended token needs no cache reads; only its write destination
    # is unaligned. This does not extend the existing cached-token load contract.
    warm = torch.cuda.Stream()
    with torch.cuda.stream(warm):
        warm.wait_stream(torch.cuda.current_stream())
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for _ in range(2):
        new_k.normal_()
        new_v.normal_()
        graph.replay()
        torch.testing.assert_close(
            captured, new_v.repeat_interleave(heads // kv_heads, dim=1), atol=0, rtol=0
        )
        for backing, expected, offset, new in zip(
            storage, expected_storage, offsets, (new_k, new_v)
        ):
            start = offset + slot * kv_heads * dim
            expected[start : start + kv_heads * dim] = new.flatten()
            # Include storage guards and untouched cache slots in the comparison.
            torch.testing.assert_close(backing, expected, atol=0, rtol=0)


@skip_no_kernel
@pytest.mark.parametrize("dim", [32, 64, 128, 256])
def test_paged_decode_combines_empty_and_partial_splits(dim):
    torch.manual_seed(25)
    batch, heads, kv_heads, width = 2, 6, 2, 97
    lengths = (3, 81)
    q = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(batch * width, kv_heads, dim, device="cuda", dtype=q.dtype)
    v_cache = torch.randn_like(k_cache)
    table = torch.arange(batch * width, device="cuda", dtype=torch.int32).view(
        batch, width
    )
    requests = torch.arange(batch, device="cuda", dtype=torch.int32)
    indptr = torch.tensor([0, 3, 84], device="cuda", dtype=torch.int32)
    # The capacity supplies seven KV tiles: production planning allows three
    # splits. Short requests and this mask leave both whole and partial rows empty.
    mask = torch.zeros(batch, heads, 1, width, device="cuda", dtype=torch.bool)
    mask[0, 1:, :, :3] = True
    mask[1, 1:, :, 70:80] = True
    mask[1, 2, :, :2] = True

    out = attn_paged_decode(
        q, k_cache, v_cache, table, requests, indptr, mask=mask, scale=0.19
    )

    for b, count in enumerate(lengths):
        expected = _reference(
            q[b : b + 1, None],
            k_cache[None, b * width : b * width + count],
            v_cache[None, b * width : b * width + count],
            mask=mask[b : b + 1, :, :, :count],
            scale=0.19,
        )
        torch.testing.assert_close(out[b], expected[0, 0], atol=0.02, rtol=0.02)
    assert torch.count_nonzero(out[:, 0]) == 0


@skip_no_kernel
def test_split_workspace_pair_is_required():
    q, k, v, kc, vc, table, requests, indptr = _paged(torch.bfloat16)
    workspace = torch.empty(2, N_HEADS, 32, D, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="provided together"):
        attn_paged_decode(q, kc, vc, table, requests, indptr, o_part_buf=workspace)


@skip_no_kernel
@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_native_scale_never_silently_uses_default(scale):
    q, k, v = _contig(torch.bfloat16)
    with pytest.raises(RuntimeError, match="finite and positive"):
        attn_prefill(q, k, v, scale=scale)


@skip_no_kernel
@pytest.mark.parametrize("rank", [2, 3, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_prefill_mask_query_axis_broadcast(rank, causal):
    torch.manual_seed(22)
    q = torch.randn(2, 17, 6, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 23, 2, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    mask = torch.ones(1, 23, device="cuda", dtype=torch.bool)
    mask[:, 2::3] = False
    if rank == 3:
        mask = mask[:, None, :]
    elif rank == 4:
        mask = mask[:, None, None, :]
    out = attn_prefill(q, k, v, mask=mask, is_causal=causal)
    expected = _reference(q, k, v, mask=mask, causal=causal)
    torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)


@skip_no_kernel
@pytest.mark.parametrize(
    "mask_rows,expand_rows",
    [(1, False), (3, False), (3, True)],
    ids=["broadcast", "short", "expanded_short"],
)
@pytest.mark.parametrize("mask_cols", [7, 19])
def test_paged_prefill_short_mask_bounds(mask_rows, expand_rows, mask_cols):
    torch.manual_seed(23)
    q_lens, kv_lens = (5, 9), (11, 17)
    batch, width, heads, kv_heads, dim = 2, 19, 6, 2, 64
    q = torch.randn(sum(q_lens), heads, dim, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(batch * width, kv_heads, dim, device="cuda", dtype=q.dtype)
    v_cache = torch.randn_like(k_cache)
    table = torch.arange(batch * width, device="cuda", dtype=torch.int32).view(
        batch, width
    )
    requests = torch.arange(batch, device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor([0, 11, 28], device="cuda", dtype=torch.int32)
    qo_indptr = torch.tensor([0, 5, 14], device="cuda", dtype=torch.int32)
    tile_batches = requests.clone()
    tile_indices = torch.zeros(batch, device="cuda", dtype=torch.int32)
    row_mask = (torch.arange(mask_cols, device="cuda") % 3 != 1).view(
        1, 1, 1, mask_cols
    )
    if expand_rows:
        # A zero query stride does not make a non-singleton extent broadcast.
        mask = row_mask.expand(1, 1, mask_rows, mask_cols)
    else:
        mask = row_mask.repeat(1, 1, mask_rows, 1)
        if mask_rows > 1:
            mask[:, :, 1, 0] = False

    out = attn_paged_prefill(
        q,
        k_cache,
        v_cache,
        table,
        requests,
        kv_indptr,
        qo_indptr,
        tile_batches,
        tile_indices,
        mask=mask,
        is_causal=False,
    )
    start = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        full_mask = torch.zeros(1, 1, q_len, kv_len, device="cuda", dtype=torch.bool)
        valid_rows = q_len if mask_rows == 1 else mask_rows
        valid_cols = min(mask_cols, kv_len)
        full_mask[:, :, :valid_rows, :valid_cols] = mask[..., :valid_cols]
        expected = _reference(
            q[None, start : start + q_len],
            k_cache[None, b * width : b * width + kv_len],
            v_cache[None, b * width : b * width + kv_len],
            mask=full_mask,
        )
        torch.testing.assert_close(
            out[start : start + q_len], expected[0], atol=0.02, rtol=0.02
        )
        if mask_rows > 1:
            assert torch.count_nonzero(out[start + mask_rows : start + q_len]) == 0
        start += q_len
