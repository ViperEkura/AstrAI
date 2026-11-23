"""Numerical equivalence between Torch and CUDA attention implementations.

Covers training forward, inference prefill, inference decode (mixed
seq_lens with padding mask), and end-to-end rollout run_batch.
"""

import pytest
import torch

from astrai.extension import attention, op_backend
from astrai.extension.runtime.loader import is_available
from astrai.inference.core.cache import BlockPool, KVCacheManager
from astrai.inference.core.engine import EngineCore
from astrai.inference.core.request import Request
from astrai.inference.worker.graph import CUDAGraphRunner
from astrai.inference.worker.workspace import InferenceWorkspace
from astrai.trainer.backend import run_batch
from tests.support.cache import appended_slots, bind_cache
from tests.support.capabilities import skip_no_cuda
from tests.support.models import D
from tests.support.tokenizers import FakeTokenizer


def _mk_task_cache(pool: BlockPool) -> KVCacheManager:
    return KVCacheManager(pool)


def _ws(pool: BlockPool) -> InferenceWorkspace:
    return InferenceWorkspace(
        pool.max_batch_size,
        max_q_heads=2,
        head_dim=64,
        device=pool.device,
        dtype=pool.dtype,
    )


@pytest.mark.cuda
@skip_no_cuda
def test_training_forward_matches_torch(cuda_model):
    """Training forward (kv_cache=None) resolves to a capable dense backend.

    CUDA attention cannot run training (requires a KV cache), so the default
    falls back by capability — flash when it can handle the call
    (mask-free/causal), otherwise torch SDPA.  The default path must not
    raise and must produce finite logits; explicitly selected torch SDPA
    must be deterministic across runs.
    """

    model, _ = cuda_model
    input_ids = torch.randint(0, 1000, (2, 16), device="cuda")

    with torch.no_grad():
        out_default = model(input_ids)

    with op_backend(attention="torch"):
        with torch.no_grad():
            out_torch_a = model(input_ids)
        with torch.no_grad():
            out_torch_b = model(input_ids)

    assert out_default["logits"].shape == out_torch_a["logits"].shape
    assert torch.isfinite(out_default["logits"]).all()
    torch.testing.assert_close(
        out_torch_a["logits"], out_torch_b["logits"], atol=0, rtol=0
    )


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_available("attention")),
    reason="attention CUDA kernel unavailable",
)
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
    req_t1 = Request("t1", prompt_ids[0])
    task_cache.alloc([req_t1])[0]
    req_t2 = Request("t2", prompt_ids[1])
    task_cache.alloc([req_t2])[0]
    kv1 = bind_cache(task_cache, ws, [req_t1, req_t2], start_pos=0)
    with torch.inference_mode():
        out_torch = model(input_ids, kv_cache=kv1, position_ids=position_ids)

    task_cache.free([req_t1])
    task_cache.free([req_t2])
    req_t1 = Request("t1", prompt_ids[0])
    task_cache.alloc([req_t1])[0]
    req_t2 = Request("t2", prompt_ids[1])
    task_cache.alloc([req_t2])[0]
    kv2 = bind_cache(task_cache, ws, [req_t1, req_t2], start_pos=0)
    with op_backend(attention="cuda"):
        with torch.inference_mode():
            out_cuda = model(
                input_ids,
                kv_cache=kv2,
                position_ids=position_ids,
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


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_available("attention")),
    reason="attention CUDA kernel unavailable",
)
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
        req_t = Request("t", prompt_ids)
        assert task_cache.alloc([req_t])[0]
        with op_backend(attention=backend), torch.inference_mode():
            model(
                ids[:3],
                input_mask=key_mask[:, :3],
                position_ids=torch.arange(3, device=device),
                kv_cache=bind_cache(
                    task_cache, workspace, [req_t], start_pos=0, seq_ends=[3]
                ),
            )
            chunk = model(
                ids[3:],
                input_mask=key_mask,
                position_ids=torch.arange(3, 6, device=device),
                kv_cache=bind_cache(
                    task_cache, workspace, [req_t], start_pos=3, seq_ends=[6]
                ),
            )
        return chunk["logits"]

    torch_out = run("torch")
    cuda_out = run("cuda")
    torch.testing.assert_close(cuda_out, torch_out, atol=0.05, rtol=0.05)


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_available("attention")),
    reason="attention CUDA kernel unavailable",
)
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
    req_t1 = Request("t1", prompt_ids[0])
    task_cache.alloc([req_t1])[0]
    req_t2 = Request("t2", prompt_ids[1])
    task_cache.alloc([req_t2])[0]
    kv = bind_cache(task_cache, ws, [req_t1, req_t2], start_pos=0)
    with torch.inference_mode():
        model(input_ids, kv_cache=kv, position_ids=position_ids)

    # Decode step — seq_lens are 9 and 7 (after extending)
    dec_ids = torch.tensor([99, 98], dtype=torch.long, device=device)
    dec_pos = torch.tensor([8, 6], dtype=torch.long, device=device)

    task_cache.alloc([req_t1], [9])[0]
    task_cache.alloc([req_t2], [7])[0]
    kv_t = bind_cache(task_cache, ws, [req_t1, req_t2])
    with torch.inference_mode():
        out_torch = model(dec_ids, kv_cache=kv_t, position_ids=dec_pos)

    kv_c = bind_cache(task_cache, ws, [req_t1, req_t2])
    with op_backend(attention="cuda"):
        with torch.inference_mode():
            out_cuda = model(dec_ids, kv_cache=kv_c, position_ids=dec_pos)

    diff = (out_torch["logits"].float() - out_cuda["logits"].float()).abs().max().item()
    assert diff < 0.05, f"Decode diff (mixed seq_lens): {diff}"


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_available("attention")),
    reason="attention CUDA kernel unavailable",
)
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
    req_t1 = Request("t1", list(range(8)))
    task_cache.alloc([req_t1])[0]
    req_t2 = Request("t2", list(range(6)))
    task_cache.alloc([req_t2])[0]
    task_cache.alloc([req_t1], [9])[0]
    task_cache.alloc([req_t2], [7])[0]
    kv_cache = bind_cache(task_cache, ws, [req_t1, req_t2])

    q = torch.randn(2, 2, D, device="cuda", dtype=torch.bfloat16)
    new_k = torch.randn(2, 1, D, device="cuda", dtype=torch.bfloat16)
    new_v = torch.randn(2, 1, D, device="cuda", dtype=torch.bfloat16)
    out = attention(
        q[:, None],
        new_k[:, None],
        new_v[:, None],
        k_cache=kv_cache.k_buffer[0],
        v_cache=kv_cache.v_buffer[0],
        cache_seqlens=kv_cache.cache_seqlens,
        page_table=kv_cache.page_table,
        is_causal=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        kv_cache.k_buffer[0].flatten(0, 1)[appended_slots(kv_cache)],
        new_k,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        kv_cache.v_buffer[0].flatten(0, 1)[appended_slots(kv_cache)],
        new_v,
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(out).all()


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_available("attention")),
    reason="attention CUDA kernel unavailable",
)
def test_decode_cuda_graph_replay_is_exact(cuda_model):
    """INT32 cache indices must remain graph-capturable and replay exactly."""
    model, _ = cuda_model
    device = "cuda"
    prompt_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    cache = BlockPool(
        n_layers=2,
        n_kv_heads=1,
        head_dim=D,
        max_batch_size=2,
        n_tokens=128,
        page_size=4,
        max_seq_len=64,
        device=device,
        dtype=torch.bfloat16,
    )
    task_cache = _mk_task_cache(cache)
    ws = _ws(cache)
    req_t1 = Request("t1", prompt_ids)
    task_cache.alloc([req_t1])[0]

    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    position_ids = torch.arange(len(prompt_ids), device=device)

    with op_backend(attention="cuda"), torch.inference_mode():
        model(
            input_ids,
            position_ids=position_ids,
            kv_cache=bind_cache(task_cache, ws, [req_t1], start_pos=0),
        )

        task_cache.alloc([req_t1], [len(prompt_ids) + 1])[0]
        kv_cache = bind_cache(task_cache, ws, [req_t1])
        assert kv_cache.page_table.dtype == torch.int32
        assert kv_cache.cache_seqlens.dtype == torch.int32
        assert appended_slots(kv_cache).dtype == torch.int32

        decode_args = {
            "input_ids": torch.tensor([9], dtype=torch.long, device=device),
            "position_ids": torch.tensor([len(prompt_ids)], device=device),
            "kv_cache": kv_cache,
        }
        graph = CUDAGraphRunner(enabled=True)
        graph.forward(model, key=(1,), **decode_args)
        graph.forward(model, key=(1,), **decode_args)
        first = graph.forward(model, key=(1,), **decode_args)["logits"].clone()
        slot = appended_slots(kv_cache)[0]
        first_k = kv_cache.k_buffer.flatten(1, 2)[:, slot].clone()
        first_v = kv_cache.v_buffer.flatten(1, 2)[:, slot].clone()

        second = graph.forward(model, key=(1,), **decode_args)["logits"].clone()
        torch.cuda.synchronize()

    assert graph.has_graph((1,))
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(
        kv_cache.k_buffer.flatten(1, 2)[:, slot], first_k, rtol=0, atol=0
    )
    torch.testing.assert_close(
        kv_cache.v_buffer.flatten(1, 2)[:, slot], first_v, rtol=0, atol=0
    )

    # Replay the same capture with a different request, then cross a page boundary.
    with op_backend(attention="cuda"), torch.inference_mode():
        prompt = [32, 31, 30]
        req_other = Request("other", prompt)
        assert task_cache.alloc([req_other])[0]
        model(
            torch.tensor(prompt, device=device),
            position_ids=torch.arange(3, device=device),
            kv_cache=bind_cache(task_cache, ws, [req_other], start_pos=0),
        )
        for pos in (3, 4):
            assert task_cache.alloc([req_other], [pos + 1])[0]
            bound = bind_cache(task_cache, ws, [req_other])
            assert bound.page_table.data_ptr() == kv_cache.page_table.data_ptr()
            assert bound.cache_seqlens.data_ptr() == kv_cache.cache_seqlens.data_ptr()
            decode_args["input_ids"].fill_(40 + pos)
            decode_args["position_ids"].fill_(pos)
            decode_args["kv_cache"] = bound
            replayed = graph.forward(model, key=(1,), **decode_args)["logits"].clone()
            eager = model(**decode_args)["logits"]
            torch.testing.assert_close(replayed, eager, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_available("attention")),
    reason="attention CUDA kernel unavailable",
)
def test_run_batch_cuda_matches_torch_greedy(cuda_model):
    """Greedy decode (temperature=0) should produce identical tokens."""
    model, _ = cuda_model
    tokenizer = FakeTokenizer()

    prompts = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15, 16]]

    sched = EngineCore(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    out_torch = run_batch(sched, prompts, max_tokens=5, temperature=0.0)
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
    sched2 = EngineCore(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=4,
        max_seq_len=64,
        device="cuda",
        dtype=torch.bfloat16,
        cache=cache_cuda,
    )
    with op_backend(attention="cuda"):
        out_cuda = run_batch(sched2, prompts, max_tokens=5, temperature=0.0)
    sched2.stop()

    assert out_torch == out_cuda, f"Torch={out_torch} != CUDA={out_cuda}"


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available("attention"),
    reason="native CUDA required",
)
def test_mla_chunk_and_decode_match_dense():
    from astrai.model import AutoRegressiveLM
    from tests.support.models import make_tiny_config

    torch.manual_seed(17)
    cfg = make_tiny_config(
        attn_type="mla",
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        kv_lora_rank=16,
    )
    device = torch.device("cuda")
    model = AutoRegressiveLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    pool = BlockPool(2, 1, 64, 1, 16, device, torch.bfloat16, page_size=4, n_tokens=16)
    manager, workspace = KVCacheManager(pool), _ws(pool)
    ids = torch.arange(1, 7, device=device)
    pos = torch.arange(6, device=device)
    req_request = Request("request", ids[:5].tolist())
    assert manager.alloc([req_request])[0]
    with torch.inference_mode(), op_backend(attention="torch"):
        dense = model(ids[None])["logits"][0]
    with torch.inference_mode(), op_backend(attention="cuda"):
        chunks = []
        for start, end in ((0, 1), (1, 5)):
            cache = bind_cache(
                manager, workspace, [req_request], start_pos=start, seq_ends=[end]
            )
            chunks.append(
                model(ids[start:end], position_ids=pos[start:end], kv_cache=cache)[
                    "logits"
                ]
            )
        assert manager.alloc([req_request], [6])[0]
        chunks.append(
            model(
                ids[5:],
                position_ids=pos[5:],
                kv_cache=bind_cache(manager, workspace, [req_request]),
            )["logits"]
        )
    torch.testing.assert_close(torch.cat(chunks), dense, rtol=0.035, atol=0.035)
