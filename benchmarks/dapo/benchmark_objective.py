"""Benchmark GRPO and DAPO objective reductions on one CUDA device."""

import json

import torch


def reduce_loss(loss, mask, aggregation):
    if aggregation == "token":
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)
    lengths = mask.sum(dim=-1)
    valid = lengths > 0
    per_sequence = (loss * mask).sum(dim=-1) / lengths.clamp(min=1.0)
    return (per_sequence * valid).sum() / valid.sum().clamp(min=1)


def objective(
    log_policy,
    log_old,
    log_ref,
    rewards,
    mask,
    *,
    clip_low,
    clip_high,
    aggregation="token",
    overlong_buffer=0,
    penalty_scale=1.0,
):
    if overlong_buffer:
        max_len = mask.shape[-1]
        lengths = mask.sum(dim=-1)
        penalty_start = max_len - overlong_buffer
        penalty = ((penalty_start - lengths) / overlong_buffer).clamp(-1.0, 0.0)
        rewards = rewards + penalty_scale * penalty

    advantages = (rewards - rewards.mean(-1, keepdim=True)) / (
        rewards.std(-1, keepdim=True, unbiased=False) + 1e-8
    )
    advantages = advantages.unsqueeze(-1)
    ratio = torch.exp(log_policy - log_old)
    surrogate = torch.minimum(
        ratio * advantages,
        torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * advantages,
    )
    policy_loss = reduce_loss(-surrogate, mask, aggregation)

    ref_ratio = torch.exp(log_ref - log_policy)
    kl = ref_ratio - torch.log(ref_ratio + 1e-8) - 1.0
    return policy_loss + 0.01 * reduce_loss(kl, mask, aggregation)


def median_ms(fn, warmup=20, repeats=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2], samples[int(len(samples) * 0.99) - 1]


def main():
    torch.manual_seed(3407)
    device = torch.device("cuda")
    output = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": 3407,
        "warmup": 20,
        "repeats": 100,
        "cases": [],
    }
    for batch, group, response_len in (
        (4, 8, 256),
        (4, 8, 1024),
        (8, 8, 2048),
        (4, 8, 4096),
    ):
        shape = (batch, group, response_len)
        log_policy = torch.randn(shape, device=device) * 0.2
        log_old = torch.randn(shape, device=device) * 0.2
        log_ref = torch.randn(shape, device=device) * 0.2
        rewards = torch.randn((batch, group), device=device)
        lengths = torch.randint(
            response_len // 2,
            response_len + 1,
            (batch, group),
            device=device,
        )
        mask = (
            torch.arange(response_len, device=device)[None, None, :]
            < lengths[..., None]
        ).float()

        legacy = lambda: objective(  # noqa: E731
            log_policy,
            log_old,
            log_ref,
            rewards,
            mask,
            clip_low=0.2,
            clip_high=0.2,
        )
        symmetric = lambda: objective(  # noqa: E731
            log_policy,
            log_old,
            log_ref,
            rewards,
            mask,
            clip_low=0.2,
            clip_high=0.2,
        )
        dapo = lambda: objective(  # noqa: E731
            log_policy,
            log_old,
            log_ref,
            rewards,
            mask,
            clip_low=0.2,
            clip_high=0.28,
            aggregation="token",
            overlong_buffer=max(1, response_len // 8),
        )

        legacy_median, legacy_p99 = median_ms(legacy)
        symmetric_median, symmetric_p99 = median_ms(symmetric)
        dapo_median, dapo_p99 = median_ms(dapo)
        output["cases"].append(
            {
                "batch": batch,
                "group": group,
                "response_len": response_len,
                "valid_tokens": int(mask.sum().item()),
                "symmetric_loss_parity_abs": abs(legacy().item() - symmetric().item()),
                "legacy_median_ms": legacy_median,
                "legacy_p99_ms": legacy_p99,
                "candidate_symmetric_median_ms": symmetric_median,
                "candidate_symmetric_p99_ms": symmetric_p99,
                "candidate_symmetric_delta_percent": (
                    symmetric_median / legacy_median - 1.0
                )
                * 100.0,
                "dapo_median_ms": dapo_median,
                "dapo_p99_ms": dapo_p99,
                "dapo_delta_percent": (dapo_median / legacy_median - 1.0) * 100.0,
            }
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
