"""Benchmark AutoRegressiveLM with KVCache"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict

import torch

from astrai.config import AutoRegressiveLMConfig
from astrai.inference import ContiguousCache, PageCache
from astrai.model.transformer import AutoRegressiveLM


@dataclass
class BenchmarkResult:
    total_tokens: int
    total_time: float
    tokens_per_second: float
    metadata: Dict[str, Any]


class GenerationBenchmark:
    def __init__(
        self,
        config: AutoRegressiveLMConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_type: str = "contiguous",
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.cache_type = cache_type
        self.model = AutoRegressiveLM(config).to(device=device, dtype=dtype)
        self.model.eval()

    @torch.inference_mode()
    def run_prefill_benchmark(
        self,
        batch_size: int = 1,
        prompt_length: int = 512,
        num_trials: int = 10,
    ) -> BenchmarkResult:
        for _ in range(3):
            prompt_ids = torch.randint(
                0,
                self.config.vocab_size,
                (batch_size, prompt_length),
                device=self.device,
                dtype=torch.long,
            )
            _ = self.model(prompt_ids)
        torch.cuda.synchronize()

        total_time = 0.0
        total_tokens = batch_size * prompt_length * num_trials

        for trial in range(num_trials):
            prompt_ids = torch.randint(
                0,
                self.config.vocab_size,
                (batch_size, prompt_length),
                device=self.device,
                dtype=torch.long,
            )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = self.model(prompt_ids)
            end.record()
            torch.cuda.synchronize()

            trial_time = start.elapsed_time(end) / 1000
            total_time += trial_time

            print(
                f"  Trial {trial + 1}/{num_trials}: {prompt_length} tokens in {trial_time:.3f}s "
                f"({prompt_length / trial_time:.1f} tok/s)"
            )

        return BenchmarkResult(
            total_tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
            metadata={
                "benchmark_type": "prefill",
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "dtype": str(self.dtype),
                "device": self.device,
                "cache": "none",
            },
        )

    @torch.inference_mode()
    def run_decoding_benchmark(
        self,
        batch_size: int = 1,
        prompt_length: int = 512,
        gen_length: int = 128,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        total_time = 0.0
        total_tokens = batch_size * gen_length * num_trials

        for trial in range(num_trials):
            prompt_ids = torch.randint(
                0,
                self.config.vocab_size,
                (batch_size, prompt_length),
                device=self.device,
                dtype=torch.long,
            )
            gen_ids = torch.randint(
                0,
                self.config.vocab_size,
                (batch_size, gen_length),
                device=self.device,
                dtype=torch.long,
            )

            head_dim = self.config.dim // self.config.n_heads
            max_seq = prompt_length + gen_length

            if self.cache_type == "contiguous":
                cache = ContiguousCache(
                    self.config.n_layers,
                    batch_size,
                    max_seq,
                    self.config.n_kv_heads,
                    head_dim,
                    self.device,
                    self.dtype,
                )
            else:
                page_size = 128
                n_pages = (max_seq + page_size - 1) // page_size * batch_size
                cache = PageCache(
                    self.config.n_layers,
                    n_pages,
                    page_size,
                    self.config.n_kv_heads,
                    head_dim,
                    self.device,
                    self.dtype,
                )

            task_ids = [f"b{i}" for i in range(batch_size)]
            for tid in task_ids:
                cache.task_alloc(tid, [0] * max_seq)
                for p in range(max_seq):
                    cache.task_extend(tid, p)

            cv = cache.bind_tasks(task_ids, prompt_length, self.device)
            _ = self.model(
                prompt_ids,
                paged_cache=cv,
                position_ids=torch.arange(
                    prompt_length, dtype=torch.long, device=self.device
                )
                .unsqueeze(0)
                .expand(batch_size, -1),
            )
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            for i in range(gen_length):
                pos = prompt_length + i
                cv = cache.bind_tasks(task_ids, pos + 1, self.device)
                _ = self.model(
                    gen_ids[:, i : i + 1],
                    paged_cache=cv,
                    position_ids=torch.full(
                        (batch_size, 1),
                        pos,
                        dtype=torch.long,
                        device=self.device,
                    ),
                )

            end.record()
            torch.cuda.synchronize()

            for tid in task_ids:
                cache.task_free(tid)

            trial_time = start.elapsed_time(end) / 1000
            total_time += trial_time

            print(
                f"  Trial {trial + 1}/{num_trials}: {gen_length} tokens in {trial_time:.3f}s "
                f"({gen_length / trial_time:.1f} tok/s)"
            )

        return BenchmarkResult(
            total_tokens=total_tokens,
            total_time=total_time,
            tokens_per_second=total_tokens / total_time,
            metadata={
                "benchmark_type": "decoding",
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "gen_length": gen_length,
                "dtype": str(self.dtype),
                "device": self.device,
                "cache": self.cache_type,
            },
        )


def print_benchmark_result(result: BenchmarkResult):
    btype = result.metadata["benchmark_type"]
    print(f"\n{' ' + btype.upper() + ' Benchmark ':-^80}")
    print(f"Total Tokens Processed: {result.total_tokens:,}")
    print(f"Time Consumed: {result.total_time:.3f}s")
    print(f"Throughput: {result.tokens_per_second:,.1f} tok/s")
    for k, v in result.metadata.items():
        if k != "benchmark_type":
            print(f"{k.replace('_', ' ').title()}: {v}")
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoRegressiveLM benchmark")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (default: cuda)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="contiguous",
        choices=["contiguous", "paged"],
        help="KV cache type",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--prompt_length", type=int, default=512, help="Prompt length")
    parser.add_argument("--gen_length", type=int, default=128, help="Generation length")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials")
    parser.add_argument(
        "--prefill_only", action="store_true", help="Run prefill benchmark only"
    )
    parser.add_argument(
        "--decode_only", action="store_true", help="Run decoding benchmark only"
    )
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    config = AutoRegressiveLMConfig(
        vocab_size=10000,
        dim=1536,
        n_heads=24,
        n_kv_heads=4,
        dim_ffn=6912,
        max_len=2048,
        n_layers=24,
        norm_eps=1e-5,
    )

    benchmark = GenerationBenchmark(
        config, device=args.device, dtype=dtype_map[args.dtype], cache_type=args.cache
    )

    print("=" * 80)
    print(
        f"Running AutoRegressiveLM Benchmark (device={args.device}, dtype={args.dtype})"
    )
    print("=" * 80)

    if not args.decode_only:
        prefill_result = benchmark.run_prefill_benchmark(
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            num_trials=args.num_trials,
        )
        print_benchmark_result(prefill_result)

    if not args.prefill_only:
        gen_result = benchmark.run_decoding_benchmark(
            batch_size=args.batch_size,
            prompt_length=args.prompt_length,
            gen_length=args.gen_length,
            num_trials=args.num_trials,
        )
        print_benchmark_result(gen_result)
