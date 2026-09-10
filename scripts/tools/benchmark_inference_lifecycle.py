"""Benchmark release/resume latency and memory for colocated inference.

The benchmark keeps model weights resident and measures only the inference
runtime lifecycle: KV storage, request-index buffers, decode workspace, and
CUDA graphs.  Every resume cycle runs the same greedy token batch and verifies
that outputs match the pre-release reference.
"""

import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Optional

import click
import torch

from astrai.config.model_config import AutoRegressiveLMConfig
from astrai.inference.scheduler import InferenceScheduler
from astrai.model.transformer import AutoRegressiveLM

_ASTRAI_1B = {
    "vocab_size": 100000,
    "hidden_size": 1536,
    "num_hidden_layers": 24,
    "intermediate_size": 6912,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
}

_TINY = {
    "vocab_size": 200,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
}


class _TokenIdsOnlyTokenizer:
    stop_ids: list[int] = []


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _latency_summary(samples: list[float]) -> dict:
    return {
        "samples": samples,
        "median": statistics.median(samples),
        "p99": _percentile(samples, 0.99),
    }


def _memory_mib(device: torch.device) -> float:
    return torch.cuda.memory_allocated(device) / (1024**2)


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@click.command()
@click.option("--preset", type=click.Choice(["astrai-1b", "tiny"]), default="astrai-1b")
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option(
    "--max-seq-len", type=click.IntRange(min=2), default=2048, show_default=True
)
@click.option(
    "--prompt-len", type=click.IntRange(min=1), default=128, show_default=True
)
@click.option("--max-tokens", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--trials", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--cuda-graph/--no-cuda-graph", default=False, show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--commit", "source_commit", default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def main(
    preset: str,
    batch_size: int,
    max_seq_len: int,
    prompt_len: int,
    max_tokens: int,
    trials: int,
    cuda_graph: bool,
    device: str,
    source_commit: Optional[str],
    output: Optional[Path],
):
    """Measure memory reclaimed by ``InferenceScheduler.release()``."""
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required")
    if prompt_len + max_tokens > max_seq_len:
        raise click.BadParameter(
            "prompt-len + max-tokens must not exceed max-seq-len",
            param_hint="--max-seq-len",
        )

    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(target)
    config_values = _ASTRAI_1B if preset == "astrai-1b" else _TINY
    config = AutoRegressiveLMConfig(**config_values)
    torch.manual_seed(0)
    model = AutoRegressiveLM(config).to(device=target, dtype=torch.bfloat16).eval()
    torch.cuda.synchronize(target)
    torch.cuda.empty_cache()
    model_only_mib = _memory_mib(target)

    scheduler = InferenceScheduler(
        model=model,
        tokenizer=_TokenIdsOnlyTokenizer(),
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        device=target,
        dtype=torch.bfloat16,
        enable_cuda_graph=cuda_graph,
    )
    prompts = [
        [
            ((batch * prompt_len + index) % (config.vocab_size - 1)) + 1
            for index in range(prompt_len)
        ]
        for batch in range(batch_size)
    ]
    reference = scheduler.run_batch(prompts, max_tokens=max_tokens, temperature=0)
    torch.cuda.synchronize(target)
    initial_resident_mib = _memory_mib(target)

    release_ms = []
    resume_ms = []
    released_mib = []
    resumed_mib = []
    for _ in range(trials):
        started = time.perf_counter_ns()
        assert scheduler.release()
        torch.cuda.synchronize(target)
        release_ms.append((time.perf_counter_ns() - started) / 1e6)
        released_mib.append(_memory_mib(target))

        started = time.perf_counter_ns()
        assert scheduler.resume()
        torch.cuda.synchronize(target)
        resume_ms.append((time.perf_counter_ns() - started) / 1e6)

        actual = scheduler.run_batch(prompts, max_tokens=max_tokens, temperature=0)
        torch.cuda.synchronize(target)
        if actual != reference:
            raise RuntimeError("Greedy output changed after release/resume")
        resumed_mib.append(_memory_mib(target))

    scheduler.release()
    reclaimed_mib = initial_resident_mib - statistics.median(released_mib)
    runtime_mib = initial_resident_mib - model_only_mib
    result = {
        "schema_version": 1,
        "commit": source_commit or _git_commit(),
        "environment": {
            "gpu": torch.cuda.get_device_name(target),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "bfloat16",
        },
        "config": {
            "preset": preset,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "prompt_len": prompt_len,
            "max_tokens": max_tokens,
            "trials": trials,
            "cuda_graph": cuda_graph,
        },
        "memory_mib": {
            "model_only": model_only_mib,
            "initial_runtime_resident": initial_resident_mib,
            "released_samples": released_mib,
            "resumed_samples": resumed_mib,
            "runtime_footprint": runtime_mib,
            "reclaimed": reclaimed_mib,
            "reclaimed_fraction": reclaimed_mib / runtime_mib if runtime_mib else 0.0,
        },
        "latency_ms": {
            "release": _latency_summary(release_ms),
            "resume": _latency_summary(resume_ms),
        },
        "correctness": {
            "greedy_match": True,
            "cycles": trials,
            "reference_tokens": reference,
        },
    }

    rendered = json.dumps(result, indent=2) + "\n"
    if output is None:
        click.echo(rendered, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        click.echo(f"wrote {output}")


if __name__ == "__main__":
    main()
