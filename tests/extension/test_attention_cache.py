import pytest
import torch

from astrai.extension import attention, op_backend
from astrai.extension.runtime.loader import get_module, is_available


@pytest.mark.parametrize("regular", [False, True])
@pytest.mark.parametrize("page_size", [3, 4])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_paged_append_crosses_physical_blocks(device, regular, page_size):
    if device == "cuda" and (
        not torch.cuda.is_available() or not is_available("attention")
    ):
        pytest.skip("native CUDA required")
    torch.manual_seed(31)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pq = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    table = pq([[3, 0, 2], [4, 1, 5]])
    ks, vs = [
        torch.randn(6, page_size, 2, 64, device=device, dtype=dtype) for _ in range(2)
    ]
    oldk, oldv = ks.clone(), vs.clone()
    lengths = [3, 2] if not regular else [3, 3]
    offsets = [0, 3, sum(lengths)]
    q, k, v = [
        torch.randn(sum(lengths), h, 64, device=device, dtype=dtype) for h in (4, 2, 2)
    ]
    qin, kin, vin = (
        tuple(t.reshape(2, 3, *t.shape[1:]) for t in (q, k, v))
        if regular
        else (q, k, v)
    )
    with op_backend(attention="cuda" if device == "cuda" else "torch"):
        out = attention(
            qin,
            kin,
            vin,
            k_cache=ks,
            v_cache=vs,
            cache_seqlens=pq([3, 5]),
            page_table=table,
            cu_seqlens_q=None if regular else pq(offsets),
            is_causal=True,
        ).reshape_as(q)
    for i, (n, past) in enumerate(zip(lengths, [3, 5])):
        a, b = offsets[i : i + 2]
        pos = torch.arange(past + n, device=device)
        slot = table[i, pos // page_size].long() * page_size + pos % page_size
        oldk.view(-1, 2, 64)[slot[past:]] = k[a:b]
        oldv.view(-1, 2, 64)[slot[past:]] = v[a:b]
        keys = oldk.view(-1, 2, 64)[slot].repeat_interleave(2, 1)
        vals = oldv.view(-1, 2, 64)[slot].repeat_interleave(2, 1)
        mask = pos[None, :] <= past + torch.arange(n, device=device)[:, None]
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[a:b].float().transpose(0, 1),
            keys.float().transpose(0, 1),
            vals.float().transpose(0, 1),
            attn_mask=mask,
        ).transpose(0, 1)
        torch.testing.assert_close(out[a:b].float(), ref, rtol=0.03, atol=0.035)
    torch.testing.assert_close(ks, oldk, rtol=0, atol=0)
    torch.testing.assert_close(vs, oldv, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available("attention"),
    reason="native CUDA required",
)
def test_native_entries_and_regular_cache_layout():
    from astrai.extension.runtime.loader import get_module

    native = get_module("attention")
    assert {name for name in dir(native) if not name.startswith("_")} == {
        "attention",
        "attention_varlen",
        "attention_with_kvcache",
    }
    q = torch.randn(2, 3, 2, 64, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    kc = torch.zeros(2, 4, 2, 64, device=q.device, dtype=q.dtype)
    vc = torch.zeros_like(kc)
    pages = torch.tensor([[1], [0]], device=q.device, dtype=torch.int32)
    past = torch.zeros(2, device=q.device, dtype=torch.int32)
    out = native.attention_with_kvcache(q, kc, vc, k, v, past, pages, is_causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2),
        v.float().transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)
    torch.testing.assert_close(out.float(), expected, rtol=0.03, atol=0.035)
    torch.testing.assert_close(kc.flip(0)[:, :3], k, rtol=0, atol=0)
    torch.testing.assert_close(vc.flip(0)[:, :3], v, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("zero_heads", [False, True])
def test_invalid_cache_shape_is_rejected_before_write(device, zero_heads):
    if device == "cuda" and (
        not torch.cuda.is_available() or not is_available("attention")
    ):
        pytest.skip("native CUDA required")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    q = torch.randn(1, 2, 2, 64, device=device, dtype=dtype)
    kc = torch.randn(1, 4, 1, 64, device=device, dtype=dtype)
    vc = torch.randn_like(kc)
    before = kc.clone(), vc.clone()
    k_new = q[:, :, :1] if zero_heads else q
    if zero_heads:
        q = q[:, :, :0]
    with op_backend(attention="torch" if device == "cpu" else "cuda"):
        with pytest.raises((ValueError, RuntimeError)):
            attention(
                q,
                k_new,
                k_new,
                k_cache=kc,
                v_cache=vc,
                cache_seqlens=torch.zeros(1, device=device, dtype=torch.int32),
                page_table=torch.zeros(1, 1, device=device, dtype=torch.int32),
            )
    torch.testing.assert_close(kc, before[0])
    torch.testing.assert_close(vc, before[1])


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multiquery_cache_matches_token_decode(dtype):
    if not torch.cuda.is_available() or not is_available("attention"):
        pytest.skip("native CUDA attention required")

    # 17 tokens x 4 Q heads per KV head spans a full 64-row Q tile and a tail.
    batch, tokens, page = 2, 17, 16
    torch.manual_seed(137)
    q, k, v = [
        torch.randn(batch, tokens, heads, 128, device="cuda", dtype=dtype)
        for heads in (16, 4, 4)
    ]
    past = torch.tensor([15, 31], device="cuda", dtype=torch.int32)
    pages = torch.tensor([[4, 0, 2], [3, 5, 1]], device="cuda", dtype=torch.int32)
    kc = torch.randn(6, page, 4, 128, device="cuda", dtype=dtype)
    vc = torch.randn_like(kc)
    step_k, step_v = kc.clone(), vc.clone()
    with op_backend(attention="cuda"):
        out = attention(
            q,
            k,
            v,
            k_cache=kc,
            v_cache=vc,
            cache_seqlens=past,
            page_table=pages,
            is_causal=True,
        )
        steps = [
            attention(
                q[:, i],
                k[:, i],
                v[:, i],
                k_cache=step_k,
                v_cache=step_v,
                cache_seqlens=past + i,
                page_table=pages,
                is_causal=True,
            )
            for i in range(tokens)
        ]
    torch.testing.assert_close(out, torch.stack(steps, dim=1), rtol=0.03, atol=0.015)
    torch.testing.assert_close(kc, step_k, rtol=0, atol=0)
    torch.testing.assert_close(vc, step_v, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_bitwise_across_graphs(dtype):
    if not is_available("attention") or not torch.cuda.is_available():
        pytest.skip("native CUDA attention required")

    torch.manual_seed(91)
    device = "cuda"
    page_size, heads, kv_heads, dim = 16, 16, 4, 128
    lengths = (1, 64, 65, 2048, 4096, 4097, 8193, 16384)
    pages = max(lengths) // page_size
    k_cache = torch.randn(
        pages + 3, page_size, kv_heads, dim, device=device, dtype=dtype
    )
    v_cache = torch.randn_like(k_cache)
    table = torch.randperm(pages, device=device, dtype=torch.int32)
    q = torch.randn(1, heads, dim, device=device, dtype=dtype)
    k_new = torch.randn(1, kv_heads, dim, device=device, dtype=dtype)
    v_new = torch.randn_like(k_new)
    native = get_module("attention").attention_with_kvcache
    expected = {}
    # A graph per request capacity, followed by a larger graph with the same row
    # in a different batch slot. Other rows write a separate physical page.
    for batch, width, replay_lengths in [
        (1, (length + page_size - 1) // page_size, [length]) for length in lengths
    ] + [(1, pages, lengths), (4, pages, lengths)]:
        tables = torch.full((batch, width), pages, device=device, dtype=torch.int32)
        tables[:-1, 0] += torch.arange(batch - 1, device=device, dtype=torch.int32)
        tables[-1].copy_(table[:width])
        past = torch.zeros(batch, device=device, dtype=torch.int32)
        qs, ks, vs = [t.repeat(batch, 1, 1) for t in (q, k_new, v_new)]
        args = (qs, k_cache, v_cache, ks, vs, past, tables)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = native(*args)
        for length in replay_lengths:
            past[-1] = length - 1
            graph.replay()
            value = out[-1].view(torch.int16).clone()
            if length in expected:
                assert torch.equal(value, expected[length])
            else:
                expected[length] = value
