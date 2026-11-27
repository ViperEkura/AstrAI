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


@pytest.mark.parametrize("regular", [False, True])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@torch.inference_mode()
def test_torch_decode_with_spaced_cache_rows(device, regular):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    torch.manual_seed(12)
    dtype = torch.bfloat16 if device == "cuda" else torch.float64
    pages = torch.tensor(
        [[1, 2, -1], [5, 6, -1], [9, 10, -1]], device=device, dtype=torch.int32
    )
    past = torch.full((3,), 4, device=device, dtype=torch.int32)
    # Keep a storage offset, gaps between rows and an invalid unused page.
    kc, vc = [
        torch.randn(13, 3, 2, 16, device=device, dtype=dtype)[1:] for _ in range(2)
    ]
    q, k, v = [torch.randn(3, h, 16, device=device, dtype=dtype) for h in (6, 2, 2)]
    expected_k, expected_v = kc.clone(), vc.clone()
    expected_k[[2, 6, 10], 1] = k
    expected_v[[2, 6, 10], 1] = v
    histories = [
        torch.stack([t[i : i + 2].flatten(0, 1)[:5] for i in (1, 5, 9)])
        .repeat_interleave(3, dim=2)
        .transpose(1, 2)
        .double()
        for t in (expected_k, expected_v)
    ]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.double().unsqueeze(2), *histories
    ).squeeze(2)
    if regular:
        q, k, v = (t.unsqueeze(1) for t in (q, k, v))
    with op_backend(attention="torch"):
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
    assert out.shape == q.shape
    tol = 0.015 if device == "cuda" else 1e-10
    torch.testing.assert_close(
        out.reshape_as(expected).double(), expected, atol=tol, rtol=tol
    )
    torch.testing.assert_close(kc, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(vc, expected_v, atol=0, rtol=0)


@torch.inference_mode()
def test_torch_decode_preserves_writes_visible_in_request_order():
    # Row 1 overwrites row 0's history after row 0 has consumed it.
    kc = torch.zeros(3, 1, 1, 1)
    vc = torch.tensor([2.0, 0.0, 6.0]).view_as(kc)
    q = k = torch.zeros(2, 1, 1)
    v = torch.tensor([4.0, 10.0]).view_as(q)
    with op_backend(attention="torch"):
        out = attention(
            q,
            k,
            v,
            k_cache=kc,
            v_cache=vc,
            cache_seqlens=torch.tensor([1, 1], dtype=torch.int32),
            page_table=torch.tensor([[0, 1], [2, 0]], dtype=torch.int32),
        )
    torch.testing.assert_close(out.flatten(), torch.tensor([3.0, 8.0]))
    torch.testing.assert_close(vc.flatten(), torch.tensor([10.0, 4.0, 6.0]))


@pytest.mark.parametrize("external", [False, True])
@torch.inference_mode()
def test_torch_decode_preserves_query_aliasing_a_later_cache_write(external):
    kc = torch.zeros(4, 1, 1, 1)
    vc = torch.tensor([2.0, 0.0, 6.0, 0.0]).view_as(kc)
    # Both queries read slot 3. It is still zero when row 0 attends.
    last_slot = torch.from_numpy(vc.numpy()[3:]).view(1, 1, 1) if external else vc[3]
    q = last_slot.expand(2, 1, 1)
    k, v = torch.ones_like(q), torch.tensor([4.0, 8.0]).view_as(q)
    with op_backend(attention="torch"):
        out = attention(
            q,
            k,
            v,
            k_cache=kc,
            v_cache=vc,
            cache_seqlens=torch.tensor([1, 1], dtype=torch.int32),
            page_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        )
    torch.testing.assert_close(out[0], torch.full_like(out[0], 3.0))
    torch.testing.assert_close(vc.flatten(), torch.tensor([2.0, 4.0, 6.0, 8.0]))


@torch.inference_mode()
def test_torch_decode_validates_last_active_page_before_any_write():
    kc, vc = [torch.zeros(4, 1, 1, 4) for _ in range(2)]
    q = torch.ones(2, 1, 4)
    with (
        op_backend(attention="torch"),
        pytest.raises(ValueError, match="page index outside cache"),
    ):
        attention(
            q,
            q,
            q,
            k_cache=kc,
            v_cache=vc,
            cache_seqlens=torch.tensor([1, 1], dtype=torch.int32),
            page_table=torch.tensor([[0, 1], [2, 4]], dtype=torch.int32),
        )
    assert not kc.any() and not vc.any()


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

    # A chunk spans multiple physical pages with distinct request histories.
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


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_multiquery_split_graph_replay(dtype):
    if not torch.cuda.is_available() or not is_available("attention"):
        pytest.skip("native CUDA attention required")
    torch.manual_seed(127)
    page, tokens, heads, kv_heads, dim = 16, 65, 4, 2, 64
    device = "cuda"
    q, k, v = [
        torch.randn(1, tokens, h, dim, device=device, dtype=dtype)
        for h in (heads, kv_heads, kv_heads)
    ]
    n = 160
    kc, vc = [
        torch.randn(2 * n, page, kv_heads, dim, device=device, dtype=dtype)
        for _ in range(2)
    ]
    pages = torch.randperm(n, device=device, dtype=torch.int32)
    mask = torch.ones(1, 1, tokens, n * page, device=device, dtype=torch.bool)
    mask[..., 0, :] = False
    native = get_module("attention").attention_with_kvcache
    expected = {}
    for batch, width in [(1, 8), (1, n), (2, n)]:
        table = pages[:width].repeat(batch, 1)
        if batch == 2:
            table[0] += n
        past = torch.zeros(batch, device=device, dtype=torch.int32)
        args = [t.repeat(batch, 1, 1, 1) for t in (q, k, v)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = native(
                args[0],
                kc,
                vc,
                args[1],
                args[2],
                past,
                table,
                None,
                mask[..., : width * page],
                True,
            )
        for length in [15, 57, 401, 2031]:
            if length + tokens > width * page:
                continue
            past[-1] = length
            graph.replay()
            actual = out[-1].clone()
            assert torch.count_nonzero(actual[0]).item() == 0
            if length in expected:
                assert torch.equal(
                    actual.view(torch.int16), expected[length].view(torch.int16)
                )
            else:
                expected[length] = actual
                logical = pages[: (length + tokens + page - 1) // page].long()
                kk, vv = [
                    x[logical]
                    .flatten(0, 1)[: length + tokens]
                    .repeat_interleave(heads // kv_heads, dim=1)
                    for x in (kc, vc)
                ]
                visible = torch.arange(length + tokens, device=device)[None, :] <= (
                    length + torch.arange(tokens, device=device)[:, None]
                )
                visible[0] = False
                ref = torch.nn.functional.scaled_dot_product_attention(
                    q[0].transpose(0, 1).float(),
                    kk.transpose(0, 1).float(),
                    vv.transpose(0, 1).float(),
                    attn_mask=visible,
                ).transpose(0, 1)
                torch.testing.assert_close(actual.float(), ref, rtol=0.035, atol=0.012)
