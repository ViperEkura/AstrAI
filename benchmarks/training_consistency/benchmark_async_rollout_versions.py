"""Replay a policy update during reward scoring on a real rollout backend."""

import argparse
import inspect
import json
import statistics
import time

import torch

from astrai.inference.scheduler import InferenceScheduler
from astrai.trainer.rollout import BaseRewardModel, RolloutGenerator, RolloutRunner
from tests.helpers import FakeTokenizer, make_model


class AdvancingRewardModel(BaseRewardModel):
    """Advance the visible policy version while the rollout is being scored."""

    def __init__(self):
        self.runner = None

    def score(self, prompts, responses):
        assert self.runner is not None
        self.runner.update_weights(self.runner.policy_version + 1)
        return torch.zeros(len(prompts), len(responses[0]))


def percentile(samples, fraction):
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def run(device, trials, warmup):
    torch.manual_seed(3407)
    model, _ = make_model(device, max_position_embeddings=64)
    tokenizer = FakeTokenizer(with_chat_template=True)
    scheduler = InferenceScheduler(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=1,
        max_seq_len=64,
        device=device,
        enable_cuda_graph=False,
    )
    generator = RolloutGenerator(
        scheduler=scheduler,
        tokenizer=tokenizer,
        max_tokens=1,
        group_size=1,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
    )
    reward = AdvancingRewardModel()
    supports_lag_guard = "max_policy_lag" in inspect.signature(RolloutRunner).parameters
    runner_kwargs = {"max_policy_lag": 0} if supports_lag_guard else {}
    runner = RolloutRunner(
        generator=generator,
        reward_model=reward,
        rollout_interval=1,
        **runner_kwargs,
    )
    reward.runner = runner
    batch = {"instruction": ["Reply briefly"], "input": ["Hi"]}

    accepted_stale = 0
    rejected_stale = 0
    samples_ms = []
    total = warmup + trials
    for index in range(total):
        runner.clear_cache()
        started = time.perf_counter_ns()
        try:
            result, _ = runner(batch)
        except RuntimeError as exc:
            if "policy lag" not in str(exc):
                raise
            rejected_stale += index >= warmup
        else:
            accepted_stale += index >= warmup and (
                result.policy_version < runner.policy_version
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if index >= warmup:
            samples_ms.append(elapsed_ms)

    return {
        "revision_mode": "candidate" if supports_lag_guard else "baseline",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": 3407,
        "warmup": warmup,
        "trials": trials,
        "accepted_stale_rollouts": accepted_stale,
        "rejected_stale_rollouts": rejected_stale,
        "median_trial_ms": statistics.median(samples_ms),
        "p99_trial_ms": percentile(samples_ms, 0.99),
        "final_policy_version": runner.policy_version,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.trials <= 0 or args.warmup < 0:
        parser.error("trials must be positive and warmup must be non-negative")
    print(json.dumps(run(args.device, args.trials, args.warmup), indent=2))


if __name__ == "__main__":
    main()
