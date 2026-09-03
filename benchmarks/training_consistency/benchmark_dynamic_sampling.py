"""Benchmark versioned low-variance group refill on a real rollout backend."""

import argparse
import json
import statistics
import time

import torch

from astrai.inference.scheduler import InferenceScheduler
from astrai.trainer.rollout import (
    BaseRewardModel,
    DynamicSamplingConfig,
    RolloutGenerator,
    RolloutRunner,
)
from tests.helpers import FakeTokenizer, make_model


class PatternRewardModel(BaseRewardModel):
    """Make odd prompt groups degenerate once, then accept their refill."""

    def __init__(self):
        self.seen = {}
        self.runner = None
        self.advance_once = False
        self._advanced = False

    def reset(self, *, advance_once=False):
        self.seen.clear()
        self.advance_once = advance_once
        self._advanced = False

    def score(self, prompts, responses):
        rewards = torch.zeros(len(prompts), len(responses[0]))
        for row, prompt in enumerate(prompts):
            seen = self.seen.get(prompt, 0)
            self.seen[prompt] = seen + 1
            ordinal = int(prompt.split("topic ", 1)[1].splitlines()[0])
            if ordinal % 2 == 0 or seen > 0:
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


def make_runner(device, prompts, group_size, max_tokens, *, dynamic):
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
    reward = PatternRewardModel()
    runner = RolloutRunner(
        generator=generator,
        reward_model=reward,
        rollout_interval=1,
        max_policy_lag=1,
        dynamic_sampling=DynamicSamplingConfig(
            enabled=dynamic,
            variance_threshold=0.0,
            max_refill_rounds=2,
            max_generated_tokens_per_group=group_size * max_tokens * 4,
            max_total_rollout_tokens_per_step=prompts * group_size * max_tokens * 4,
            max_pending_groups=prompts,
            base_seed=3407,
        ),
    )
    reward.runner = runner
    return runner, reward


def measure(runner, reward, batch, *, trials, warmup):
    latency_ms = []
    generated_tokens = []
    accepted_groups = []
    waste_ratio = []
    for trial in range(warmup + trials):
        reward.reset()
        runner.clear_cache()
        started = time.perf_counter_ns()
        result, _ = runner(batch)
        if result.responses.device.type == "cuda":
            torch.cuda.synchronize(result.responses.device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        variances = result.rewards.float().var(dim=1, unbiased=False)
        metrics = runner.last_sampling_metrics
        if trial >= warmup:
            latency_ms.append(elapsed_ms)
            generated_tokens.append(
                int(metrics.get("total_generated_tokens", result.response_mask.sum()))
            )
            accepted_groups.append(
                int(metrics.get("groups_accepted", (variances > 0).sum()))
            )
            waste_ratio.append(float(metrics.get("rollout_waste_ratio", 0.0)))
    total_tokens = sum(generated_tokens)
    return {
        "median_latency_ms": statistics.median(latency_ms),
        "p95_latency_ms": percentile(latency_ms, 0.95),
        "generated_tokens": total_tokens,
        "accepted_groups": sum(accepted_groups),
        "effective_groups_per_million_tokens": (
            sum(accepted_groups) * 1_000_000 / total_tokens
        ),
        "mean_rollout_waste_ratio": statistics.mean(waste_ratio),
    }


def run(device, prompts, group_size, max_tokens, trials, warmup):
    torch.manual_seed(3407)
    batch = {
        "instruction": [f"Reply about topic {index}" for index in range(prompts)],
        "input": ["briefly"] * prompts,
    }
    baseline, baseline_reward = make_runner(
        device, prompts, group_size, max_tokens, dynamic=False
    )
    dynamic, dynamic_reward = make_runner(
        device, prompts, group_size, max_tokens, dynamic=True
    )
    baseline_result = measure(
        baseline, baseline_reward, batch, trials=trials, warmup=warmup
    )
    dynamic_result = measure(
        dynamic, dynamic_reward, batch, trials=trials, warmup=warmup
    )

    # Fault injection: advance the policy while the first scoring call is in
    # flight. The low-variance refill must invalidate old rows and restart all
    # prompt groups under exactly one new version.
    dynamic_reward.reset(advance_once=True)
    dynamic.clear_cache()
    jittered, _ = dynamic(batch)
    jitter_versions = {
        group.behavior_policy_version for group in jittered.sampling_groups
    }

    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device)
        if str(device).startswith("cuda")
        else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": 3407,
        "prompts": prompts,
        "group_size": group_size,
        "max_tokens": max_tokens,
        "warmup": warmup,
        "trials": trials,
        "baseline": baseline_result,
        "dynamic": dynamic_result,
        "version_jitter": {
            "final_policy_version": jittered.policy_version,
            "behavior_policy_versions": sorted(jitter_versions),
            "mixed_version_groups": len(jitter_versions) != 1,
            "version_invalidated_groups": dynamic.last_sampling_metrics[
                "version_invalidated_groups"
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if min(args.prompts, args.group_size, args.max_tokens, args.trials) <= 0:
        parser.error("prompts, group-size, max-tokens, and trials must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
