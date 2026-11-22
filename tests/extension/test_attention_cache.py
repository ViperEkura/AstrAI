import pytest
import torch

from astrai.extension import attention_with_kvcache, op_backend
from astrai.extension.runtime.loader import is_available


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
        out = attention_with_kvcache(
            qin,
            ks,
            vs,
            k_new=kin,
            v_new=vin,
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
            attention_with_kvcache(
                q,
                kc,
                vc,
                k_new=k_new,
                v_new=k_new,
                cache_seqlens=torch.zeros(1, device=device, dtype=torch.int32),
                page_table=torch.zeros(1, 1, device=device, dtype=torch.int32),
            )
    torch.testing.assert_close(kc, before[0])
    torch.testing.assert_close(vc, before[1])
