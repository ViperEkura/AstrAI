import pytest
import torch

from astrai.extension import attention_with_kvcache, op_backend
from astrai.extension.runtime.loader import is_available


@pytest.mark.parametrize("regular", [False, True])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_paged_append_crosses_physical_blocks(device, regular):
    if device == "cuda" and (
        not torch.cuda.is_available() or not is_available("attention")
    ):
        pytest.skip("native CUDA required")
    torch.manual_seed(31)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pq = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    table = pq([[3, 0, 2], [4, 1, 5]])
    ks, vs = [torch.randn(6, 4, 2, 64, device=device, dtype=dtype) for _ in range(2)]
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
        slot = table[i, pos // 4].long() * 4 + pos % 4
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
