"""Direct native entry checks must protect callers without Python validation."""

import pytest
import torch

from astrai.extension.runtime.loader import get_module, is_available

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or not is_available("attention"),
        reason="native CUDA required",
    ),
]


@pytest.mark.parametrize("mode", ["dense", "varlen", "cache"])
@pytest.mark.parametrize("invalid", ["rank", "heads", "shape", "dtype", "device"])
def test_native_tensor_contracts(mode, invalid):
    q = torch.randn(1, 2, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, 2, 64, device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    if mode != "dense":
        q, k, v = (t[0] for t in (q, k, v))
    kc = torch.randn(2, 4, 2, 64, device=q.device, dtype=q.dtype)
    vc = torch.randn_like(kc)
    before = kc.clone(), vc.clone()
    offsets = torch.tensor([0, 2], device=q.device, dtype=torch.int32)
    past = torch.zeros(1, device=q.device, dtype=torch.int32)
    pages = torch.tensor([[1, 0]], device=q.device, dtype=torch.int32)
    if invalid == "rank":
        q = q[0]
    elif invalid == "heads":
        k, v = k[..., :0, :], v[..., :0, :]
    elif invalid == "shape":
        v = v[..., :1, :]
    elif invalid == "dtype":
        v = v.float()
    else:
        v = v.cpu()
    native = get_module("attention")
    with pytest.raises(RuntimeError):
        if mode == "dense":
            native.attention(q, k, v)
        elif mode == "varlen":
            native.attention_varlen(q, k, v, offsets)
        else:
            native.attention_with_kvcache(q, kc, vc, k, v, past, pages, offsets)
    torch.testing.assert_close(kc, before[0])
    torch.testing.assert_close(vc, before[1])


@pytest.mark.parametrize("mode", ["varlen", "cache"])
@pytest.mark.parametrize(
    "invalid", ["offset_dtype", "offset_rank", "offset_device", "grad"]
)
def test_native_segment_contracts(mode, invalid):
    q = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 64, device=q.device, dtype=q.dtype)
    v = torch.randn_like(k)
    offsets = torch.tensor([0, 2], device=q.device, dtype=torch.int32)
    kc = torch.randn(2, 4, 2, 64, device=q.device, dtype=q.dtype)
    vc = torch.randn_like(kc)
    before = kc.clone(), vc.clone()
    if invalid == "offset_dtype":
        offsets = offsets.long()
    elif invalid == "offset_rank":
        offsets = offsets[None]
    elif invalid == "offset_device":
        offsets = offsets.cpu()
    else:
        q.requires_grad_()
    native = get_module("attention")
    with torch.enable_grad(), pytest.raises(RuntimeError):
        if mode == "varlen":
            native.attention_varlen(q, k, v, offsets)
        else:
            native.attention_with_kvcache(
                q,
                kc,
                vc,
                k,
                v,
                torch.zeros(1, device=q.device, dtype=torch.int32),
                torch.tensor([[1, 0]], device=q.device, dtype=torch.int32),
                offsets,
            )
    torch.testing.assert_close(kc, before[0])
    torch.testing.assert_close(vc, before[1])
