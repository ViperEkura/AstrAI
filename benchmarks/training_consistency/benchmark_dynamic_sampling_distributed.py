"""Soak dynamic-sampling rank agreement with intentionally skewed rewards."""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from astrai.inference.scheduler import InferenceScheduler
from astrai.trainer.rollout import (
    BaseRewardModel,
    DynamicSamplingConfig,
    RolloutGenerator,
    RolloutRunner,
)
from tests.helpers import FakeTokenizer, make_model


class RankSkewRewardModel(BaseRewardModel):
    """Disagree on first-round acceptance, then agree on each refill."""

    def __init__(self, rank):
        self.rank = rank
        self.seen = {}
        self.advance_once = False
        self._advanced = False
        self.fail_once = False
        self._failed = False
        self.runner = None

    def reset(self, *, advance_once, fail_once=False):
        self.seen.clear()
        self.advance_once = advance_once
        self._advanced = False
        self.fail_once = fail_once
        self._failed = False

    def score(self, prompts, responses):
        if self.fail_once and self.rank == 0 and not self._failed:
            self._failed = True
            raise RuntimeError("injected rank-local scoring failure")
        rewards = torch.zeros(len(prompts), len(responses[0]))
        for row, prompt in enumerate(prompts):
            seen = self.seen.get(prompt, 0)
            self.seen[prompt] = seen + 1
            ordinal = int(prompt.split("topic ", 1)[1].splitlines()[0])
            if seen > 0 or self.rank == ordinal % dist.get_world_size():
                rewards[row] = torch.arange(rewards.size(1), dtype=torch.float32)
            else:
                rewards[row] = 1.0
        if self.advance_once and not self._advanced:
            assert self.runner is not None
            self.runner.update_weights(self.runner.policy_version + 1)
            self._advanced = True
        return rewards


def percentile(samples, fraction):
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def run(steps, warmup, jitter_interval, prompts, group_size, max_tokens):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(3407)

    model, _ = make_model(device, max_position_embeddings=128)
    tokenizer = FakeTokenizer(with_chat_template=True)
    scheduler = InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=prompts * group_size,
        max_seq_len=128,
        device=device,
        enable_cuda_graph=False,
    )
    generator = RolloutGenerator(
        scheduler=scheduler,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        group_size=group_size,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    reward = RankSkewRewardModel(rank)
    runner = RolloutRunner(
        generator=generator,
        reward_model=reward,
        rollout_interval=1,
        max_policy_lag=1,
        dynamic_sampling=DynamicSamplingConfig(
            enabled=True,
            max_refill_rounds=2,
            max_generated_tokens_per_group=group_size * max_tokens * 4,
            max_total_rollout_tokens_per_step=prompts * group_size * max_tokens * 4,
            max_pending_groups=prompts,
        ),
    )
    reward.runner = runner
    batch = {
        "instruction": [f"Reply about topic {index}" for index in range(prompts)],
        "input": ["briefly"] * prompts,
    }

    generation_schedule = []
    fail_generation = False
    original_generate = generator.generate

    def record_generate(batch, *, generation_seed=None):
        nonlocal fail_generation
        generation_schedule.append(len(batch["instruction"]))
        if fail_generation and rank == 0:
            fail_generation = False
            raise RuntimeError("injected rank-local generation failure")
        return original_generate(batch, generation_seed=generation_seed)

    generator.generate = record_generate
    start_allocated = 0
    start_reserved = 0
    latencies_ms = []
    mixed_versions = 0
    incomplete_batches = 0
    schedule_mismatches = 0
    invalidated_groups = 0

    for iteration in range(warmup + steps):
        measured = iteration >= warmup
        step = iteration - warmup
        jitter = measured and jitter_interval > 0 and step % jitter_interval == 0
        if iteration == warmup:
            torch.cuda.reset_peak_memory_stats(device)
            start_allocated = torch.cuda.memory_allocated(device)
            start_reserved = torch.cuda.memory_reserved(device)
        reward.reset(advance_once=jitter)
        runner.clear_cache()
        generation_schedule.clear()
        dist.barrier()
        started = time.perf_counter_ns()
        result, _ = runner(batch)
        torch.cuda.synchronize(device)
        latency = torch.tensor(
            (time.perf_counter_ns() - started) / 1e6,
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(latency, op=dist.ReduceOp.MAX)
        if rank == 0 and measured:
            latencies_ms.append(float(latency.item()))

        if not measured:
            continue

        versions = {group.behavior_policy_version for group in result.sampling_groups}
        mixed_versions += int(len(versions) != 1)
        incomplete_batches += int(len(result.sampling_groups) != prompts)
        expected_schedule = (
            [prompts, prompts, prompts] if jitter else [prompts, prompts]
        )
        schedule_mismatches += int(generation_schedule != expected_schedule)
        invalidated_groups += int(
            runner.last_sampling_metrics["version_invalidated_groups"]
        )

    failure_propagation = {}
    for failure in ("generation", "scoring"):
        fail_generation = failure == "generation"
        reward.reset(advance_once=False, fail_once=failure == "scoring")
        runner.clear_cache()
        caught = False
        try:
            runner(batch)
        except RuntimeError:
            caught = True
        caught_on_all_ranks = torch.tensor(
            int(caught), dtype=torch.int32, device=device
        )
        dist.all_reduce(caught_on_all_ranks, op=dist.ReduceOp.MIN)
        failure_propagation[failure] = bool(caught_on_all_ranks.item())

    counters = torch.tensor(
        [
            mixed_versions,
            incomplete_batches,
            schedule_mismatches,
            invalidated_groups,
        ],
        dtype=torch.long,
        device=device,
    )
    dist.all_reduce(counters, op=dist.ReduceOp.SUM)
    end_allocated = torch.cuda.memory_allocated(device)
    end_reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    if rank == 0:
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(device),
                    "world_size": world_size,
                    "steps": steps,
                    "warmup": warmup,
                    "jitter_interval": jitter_interval,
                    "prompts": prompts,
                    "group_size": group_size,
                    "max_tokens": max_tokens,
                    "max_rank_latency_median_ms": statistics.median(latencies_ms),
                    "max_rank_latency_p95_ms": percentile(latencies_ms, 0.95),
                    "mixed_version_batches": int(counters[0].item()),
                    "incomplete_batches": int(counters[1].item()),
                    "generation_schedule_mismatches": int(counters[2].item()),
                    "version_invalidated_groups": int(counters[3].item()),
                    "rank_local_failure_propagation": failure_propagation,
                    "rank0_memory": {
                        "allocated_drift_bytes": end_allocated - start_allocated,
                        "reserved_drift_bytes": end_reserved - start_reserved,
                        "max_allocated_bytes": max_allocated,
                        "max_reserved_bytes": max_reserved,
                    },
                },
                indent=2,
            )
        )
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--jitter-interval", type=int, default=10)
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.prompts, args.group_size, args.max_tokens) <= 0:
        parser.error("steps, prompts, group-size, and max-tokens must be positive")
    if min(args.warmup, args.jitter_interval) < 0:
        parser.error("warmup and jitter-interval must be non-negative")
    run(**vars(args))


if __name__ == "__main__":
    main()
