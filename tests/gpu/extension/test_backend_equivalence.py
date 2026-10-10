"""Numerical equivalence between TorchNativeBackend and CudaBackend.

Covers training forward, inference prefill, inference decode (mixed
seq_lens with padding mask), and end-to-end scheduler.run_batch.
"""

import pytest
import torch

from astrai.extension import ATTN_BACKEND, attn_backend
from astrai.extension.kernel.attention import attn_paged_decode
from astrai.inference.core.cache import BlockPool, KVCacheManager
from astrai.inference.core.scheduler import Scheduler
from astrai.inference.worker.graph import CUDAGraphRunner
from astrai.inference.worker.workspace import InferenceWorkspace
from tests.support.capabilities import skip_no_kernel
from tests.support.models import D
from tests.support.tokenizers import FakeTokenizer


def _mk_task_cache(pool: BlockPool) -> KVCacheManager:
    return KVCacheManager(pool)


def _ws(pool: BlockPool) -> InferenceWorkspace:
    return InferenceWorkspace(
        pool.max_batch_size,
        pool.max_seq_len,
        max_q_heads=2,
        head_dim=64,
        device=pool.device,
        dtype=pool.dtype,
    )


@skip_no_kernel
def test_training_forward_matches_torch(cuda_model):
    """Training forward (kv_cache=None) resolves to a capable dense backend.

    CudaBackend cannot run training (requires a KV cache), so the default
    falls back by capability — flash when it can handle the call
    (mask-free/causal), otherwise torch SDPA.  The default path must not
    raise and must produce finite logits; explicitly selected torch SDPA
    must be deterministic across runs.
    """

    model, _ = cuda_model
    input_ids = torch.randint(0, 1000, (2, 16), device="cuda")

    with torch.no_grad():
        out_default = model(input_ids)

    with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
        with torch.no_grad():
            out_torch_a = model(input_ids)
        with torch.no_grad():
            out_torch_b = model(input_ids)

    assert out_default["logits"].shape == out_torch_a["logits"].shape
    assert torch.isfinite(out_default["logits"]).all()
    torch.testing.assert_close(
        out_torch_a["logits"], out_torch_b["logits"], atol=0, rtol=0
    )


@skip_no_kernel
def test_prefill_with_kv_cache_matches_torch(cuda_model):
    """Inference prefill with KV cache should match torch backend."""
    model, _ = cuda_model
    prompt_ids = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15]]
    device = "cuda"
    input_ids = torch.tensor(sum(prompt_ids, []), dtype=torch.long, device=device)
    position_ids = torch.cat([torch.arange(len(p), device=device) for p in prompt_ids])

    cache = BlockPool(
        n_layers=2,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=4,
        max_seq_len=64,
        device=device,
        dtype=torch.bfloat16,
    )

    task_cache = _mk_task_cache(cache)
    ws = _ws(cache)
    task_cache.alloc_slots("t1", prompt_ids[0])
    task_cache.alloc_slots("t2", prompt_ids[1])
    kv1 = task_cache.bind(["t1", "t2"], ws, start_pos=0)
    with torch.inference_mode():
        out_torch = model(
            input_ids, kv_cache=kv1, position_ids=position_ids, fwd="prefill"
        )

    task_cache.free_slots("t1")
    task_cache.free_slots("t2")
    task_cache.alloc_slots("t1", prompt_ids[0])
    task_cache.alloc_slots("t2", prompt_ids[1])
    kv2 = task_cache.bind(["t1", "t2"], ws, start_pos=0)
    with attn_backend(ATTN_BACKEND.CUDA):
        with torch.inference_mode():
            out_cuda = model(
                input_ids,
                kv_cache=kv2,
                position_ids=position_ids,
                fwd="prefill",
            )

    offset = 0
    for i, p in enumerate(prompt_ids):
        d = (
            (
                out_torch["logits"][offset : offset + len(p)].float()
                - out_cuda["logits"][offset : offset + len(p)].float()
            )
            .abs()
            .max()
            .item()
        )
        assert d == 0.0, f"Prefill diff for sample {i}: {d}"
        offset += len(p)


@skip_no_kernel
@pytest.mark.parametrize("mask_len", [4, 6])
def test_masked_cached_chunk_prefill_matches_torch(cuda_model, mask_len):
    model, _ = cuda_model
    device = "cuda"
    prompt_ids = [1, 2, 3, 4, 5, 6]
    ids = torch.tensor(prompt_ids, device=device)
    key_mask = torch.tensor([[True, False, True, True, False, True]], device=device)[
        :, :mask_len
    ]

    def run(backend):
        pool = BlockPool(
            n_layers=2,
            n_kv_heads=1,
            head_dim=D,
            max_batch_size=1,
            max_seq_len=64,
            device=device,
            dtype=torch.bfloat16,
        )
        task_cache = _mk_task_cache(pool)
        workspace = _ws(pool)
        assert task_cache.alloc_slots("t", prompt_ids)
        with attn_backend(backend), torch.inference_mode():
            model(
                ids[:3],
                input_mask=key_mask[:, :3],
                position_ids=torch.arange(3, device=device),
                kv_cache=task_cache.bind(["t"], workspace, start_pos=0, seq_ends=[3]),
                fwd="prefill",
            )
            chunk = model(
                ids[3:],
                input_mask=key_mask,
                position_ids=torch.arange(3, 6, device=device),
                kv_cache=task_cache.bind(["t"], workspace, start_pos=3, seq_ends=[6]),
                fwd="prefill",
            )
        return chunk["logits"]

    torch_out = run(ATTN_BACKEND.TORCH_NATIVE)
    cuda_out = run(ATTN_BACKEND.CUDA)
    torch.testing.assert_close(cuda_out, torch_out, atol=0.05, rtol=0.05)


@skip_no_kernel
def test_decode_mixed_seq_lens_matches_torch(cuda_model):
    """Decode with mixed seq_lens in batch — padding mask must produce correct output."""
    model, _ = cuda_model
    device = "cuda"

    prompt_ids = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15]]
    cache = BlockPool(
        n_layers=2,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=4,
        max_seq_len=64,
        device=device,
        dtype=torch.bfloat16,
    )

    # Prefill to populate cache
    input_ids = torch.tensor(sum(prompt_ids, []), dtype=torch.long, device=device)
    position_ids = torch.cat([torch.arange(len(p), device=device) for p in prompt_ids])

    task_cache = _mk_task_cache(cache)
    ws = _ws(cache)
    task_cache.alloc_slots("t1", prompt_ids[0])
    task_cache.alloc_slots("t2", prompt_ids[1])
    kv = task_cache.bind(["t1", "t2"], ws, start_pos=0)
    with torch.inference_mode():
        model(input_ids, kv_cache=kv, position_ids=position_ids, fwd="prefill")

    # Decode step — seq_lens are 9 and 7 (after extending)
    dec_ids = torch.tensor([99, 98], dtype=torch.long, device=device)
    dec_pos = torch.tensor([8, 6], dtype=torch.long, device=device)

    task_cache.extend_slots("t1", 8)
    task_cache.extend_slots("t2", 6)
    kv_t = task_cache.bind(["t1", "t2"], ws)
    with torch.inference_mode():
        out_torch = model(dec_ids, kv_cache=kv_t, position_ids=dec_pos, fwd="decode")

    kv_c = task_cache.bind(["t1", "t2"], ws)
    with attn_backend(ATTN_BACKEND.CUDA):
        with torch.inference_mode():
            out_cuda = model(dec_ids, kv_cache=kv_c, position_ids=dec_pos, fwd="decode")

    diff = (out_torch["logits"].float() - out_cuda["logits"].float()).abs().max().item()
    assert diff < 0.05, f"Decode diff (mixed seq_lens): {diff}"


@skip_no_kernel
def test_paged_decode_appends_new_kv_in_kernel():
    """Fused decode writes current-token K/V to each request's paged slot."""
    pool = BlockPool(
        n_layers=1,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=2,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
        page_size=8,
        n_tokens=128,
    )
    task_cache = _mk_task_cache(pool)
    ws = _ws(pool)
    task_cache.alloc_slots("t1", list(range(8)))
    task_cache.alloc_slots("t2", list(range(6)))
    task_cache.extend_slots("t1", 8)
    task_cache.extend_slots("t2", 6)
    kv_cache = task_cache.bind(["t1", "t2"], ws)

    q = torch.randn(2, 2, D, device="cuda", dtype=torch.bfloat16)
    new_k = torch.randn(2, 1, D, device="cuda", dtype=torch.bfloat16)
    new_v = torch.randn(2, 1, D, device="cuda", dtype=torch.bfloat16)
    out = attn_paged_decode(
        q,
        kv_cache.k_buffer[0],
        kv_cache.v_buffer[0],
        kv_cache.req_to_token,
        kv_cache.req_pool_indices,
        kv_cache.kv_indptr,
        new_k=new_k,
        new_v=new_v,
        is_causal=True,
        o_part_buf=kv_cache.decode_o_part,
        ml_part_buf=kv_cache.decode_ml_part,
        out_buf=kv_cache.decode_out,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        kv_cache.k_buffer[0, kv_cache.out_cache_loc], new_k, rtol=0, atol=0
    )
    torch.testing.assert_close(
        kv_cache.v_buffer[0, kv_cache.out_cache_loc], new_v, rtol=0, atol=0
    )
    assert torch.isfinite(out).all()


@skip_no_kernel
def test_decode_cuda_graph_replay_is_exact(cuda_model):
    """INT32 cache indices must remain graph-capturable and replay exactly."""
    model, _ = cuda_model
    device = "cuda"
    prompt_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    cache = BlockPool(
        n_layers=2,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=1,
        max_seq_len=64,
        device=device,
        dtype=torch.bfloat16,
    )
    task_cache = _mk_task_cache(cache)
    ws = _ws(cache)
    task_cache.alloc_slots("t1", prompt_ids)

    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    position_ids = torch.arange(len(prompt_ids), device=device)

    with attn_backend(ATTN_BACKEND.CUDA), torch.inference_mode():
        model(
            input_ids,
            position_ids=position_ids,
            kv_cache=task_cache.bind(["t1"], ws, start_pos=0),
            fwd="prefill",
        )

        task_cache.extend_slots("t1", len(prompt_ids))
        kv_cache = task_cache.bind(["t1"], ws)
        assert kv_cache.req_to_token.dtype == torch.int32
        assert kv_cache.req_pool_indices.dtype == torch.int32
        assert kv_cache.out_cache_loc.dtype == torch.int32

        decode_args = {
            "input_ids": torch.tensor([9], dtype=torch.long, device=device),
            "position_ids": torch.tensor([len(prompt_ids)], device=device),
            "kv_cache": kv_cache,
            "fwd": "decode",
        }
        graph = CUDAGraphRunner(enabled=True)
        graph.forward(model, key=(1,), **decode_args)
        graph.forward(model, key=(1,), **decode_args)
        first = graph.forward(model, key=(1,), **decode_args)["logits"].clone()
        slot = kv_cache.out_cache_loc[0]
        first_k = kv_cache.k_buffer[:, slot].clone()
        first_v = kv_cache.v_buffer[:, slot].clone()

        second = graph.forward(model, key=(1,), **decode_args)["logits"].clone()
        torch.cuda.synchronize()

    assert graph.has_graph((1,))
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(kv_cache.k_buffer[:, slot], first_k, rtol=0, atol=0)
    torch.testing.assert_close(kv_cache.v_buffer[:, slot], first_v, rtol=0, atol=0)


@skip_no_kernel
def test_run_batch_cuda_matches_torch_greedy(cuda_model):
    """Greedy decode (temperature=0) should produce identical tokens."""
    model, _ = cuda_model
    tokenizer = FakeTokenizer()

    prompts = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15, 16]]

    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    out_torch = sched.run_batch(prompts, max_tokens=5, temperature=0.0)
    sched.stop()

    cache_cuda = BlockPool(
        n_layers=2,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=4,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sched2 = Scheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
        cache=cache_cuda,
    )
    with attn_backend(ATTN_BACKEND.CUDA):
        out_cuda = sched2.run_batch(prompts, max_tokens=5, temperature=0.0)
    sched2.stop()

    assert out_torch == out_cuda, f"Torch={out_torch} != CUDA={out_cuda}"
