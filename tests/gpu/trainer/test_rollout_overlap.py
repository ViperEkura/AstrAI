"""Assigned-GPU fixed-workload parity; this is not a full RL benchmark."""

import json
import time
from copy import deepcopy

import pytest
import torch

from astrai.trainer.backend import ColocatedBackend
from astrai.trainer.rollout import RolloutGenerator, SamplingParams
from tests.support.inference import make_cpu_model, make_cpu_scheduler
from tests.support.tokenizers import FakeTokenizer


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an assigned CUDA test GPU"
)
@pytest.mark.parametrize("concurrency", [32, 64, 128, 256])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("seeded", [False, True])
def test_grouped_overlap_gpu_parity_and_dispatch(tmp_path, concurrency, graphs, seeded):
    torch.manual_seed(3407)
    source = make_cpu_model().to(device="cuda", dtype=torch.bfloat16)
    tokenizer = FakeTokenizer(with_chat_template=True)
    tokenizer.stop_ids = list(range(64)) if seeded else []
    group = 4
    outputs, evidence = [], []
    for overlap in (False, True):
        model = deepcopy(source).train()
        scheduler = make_cpu_scheduler(
            model,
            tokenizer,
            max_batch_size=concurrency,
            enable_overlap=overlap,
            enable_cuda_graph=graphs,
            device="cuda",
            max_seq_len=64,
        )
        generator = RolloutGenerator(
            ColocatedBackend(scheduler),
            tokenizer,
            SamplingParams(
                group_size=group,
                max_tokens=8,
                temperature=1,
                top_p=1,
                top_k=0,
                seed=118 if seeded else None,
            ),
        )
        batch = {
            "instruction": [
                "a" if index % 2 else "abcd" for index in range(concurrency // group)
            ]
        }
        pipeline = []
        original = scheduler.engine_core.submit

        def observe(plan):
            pipeline.append(len(scheduler.engine_core.pending))
            return original(plan)

        scheduler.engine_core.submit = observe
        try:
            generator.generate(batch)  # warm graph/buffer setup before timing
            pipeline.clear()
            torch.manual_seed(118)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            raw = generator.generate(batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            outputs.append(raw)
            assert not scheduler.engine_core.pending and not scheduler._planned
            assert (
                not scheduler._pending_order
                and scheduler._kv_manager.request_count == 0
            )
            assert all(
                not slot["in_use"] for slot in scheduler._executor._result_ring._slots
            )
            assert (1 in pipeline) == overlap
            evidence.append(
                {
                    "overlap": overlap,
                    "graphs": graphs,
                    "concurrency": concurrency,
                    "request_seeded": seeded,
                    "stop_responses": sum(
                        reason == "stop" for row in raw.finish_reasons for reason in row
                    ),
                    "collection_seconds": elapsed,
                    "output_tokens": raw.response_mask.sum().item(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "pending_depth_before_submit": pipeline,
                    "result_ring_enabled": scheduler._executor._result_ring.enabled,
                }
            )
        finally:
            scheduler.stop()
    for name in ("prompts", "prompt_mask", "responses", "response_mask"):
        torch.testing.assert_close(
            getattr(outputs[1], name), getattr(outputs[0], name), rtol=0, atol=0
        )
    torch.testing.assert_close(
        outputs[1].logprobs_old, outputs[0].logprobs_old, rtol=1e-5, atol=1e-5
    )
    assert outputs[0].finish_reasons == outputs[1].finish_reasons
    assert outputs[0].policy_version == outputs[1].policy_version == 0
    if seeded:
        lengths = outputs[0].response_mask.sum(-1)
        assert lengths.min() < lengths.max()
        assert any(
            reason == "stop" for row in outputs[0].finish_reasons for reason in row
        )
    (tmp_path / "collector-evidence.json").write_text(json.dumps(evidence, indent=2))
