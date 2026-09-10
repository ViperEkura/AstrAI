"""Compare legacy BF16 and FP32 MoE routing decisions on CUDA."""

import json

import torch


def baseline_route(logits: torch.Tensor, k: int):
    probs = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
    return torch.topk(probs, k, dim=-1, sorted=False)


def candidate_route(logits: torch.Tensor, k: int):
    probs = torch.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(probs, k, dim=-1, sorted=False)
    return weights.to(logits.dtype), indices


def median_ms(fn, logits, k, warmup=20, repeats=100):
    for _ in range(warmup):
        fn(logits, k)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(logits, k)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2], samples[int(len(samples) * 0.99) - 1]


def mismatch_rate(route, logits, k):
    reference = torch.topk(
        torch.softmax(logits.float(), dim=-1), k, dim=-1, sorted=True
    ).indices
    _, selected = route(logits, k)
    reference = torch.sort(reference, dim=-1).values
    selected = torch.sort(selected, dim=-1).values
    return (reference != selected).any(dim=-1).float().mean().item()


def main():
    torch.manual_seed(3407)
    device = torch.device("cuda")
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "warmup": 20,
        "repeats": 100,
        "cases": [],
    }
    for tokens, experts, k in (
        (2048, 8, 2),
        (8192, 8, 2),
        (8192, 64, 8),
        (32768, 64, 8),
    ):
        logits = torch.randn(tokens, experts, device=device, dtype=torch.bfloat16)
        close_logits = logits.mul(0.001)
        old_median, old_p99 = median_ms(baseline_route, logits, k)
        new_median, new_p99 = median_ms(candidate_route, logits, k)
        result["cases"].append(
            {
                "tokens": tokens,
                "experts": experts,
                "top_k": k,
                "baseline_median_ms": old_median,
                "baseline_p99_ms": old_p99,
                "candidate_median_ms": new_median,
                "candidate_p99_ms": new_p99,
                "median_delta_percent": (new_median / old_median - 1.0) * 100.0,
                "standard_logits_baseline_mismatch_percent": mismatch_rate(
                    baseline_route, logits, k
                )
                * 100.0,
                "near_tie_baseline_mismatch_percent": mismatch_rate(
                    baseline_route, close_logits, k
                )
                * 100.0,
                "candidate_mismatch_percent": max(
                    mismatch_rate(candidate_route, logits, k),
                    mismatch_rate(candidate_route, close_logits, k),
                )
                * 100.0,
                "additional_probability_storage_mib": tokens * experts * 2 / 2**20,
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
