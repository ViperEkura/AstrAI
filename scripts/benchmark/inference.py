import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import torch

from astrai.config import BaseModelConfig, ConfigFactory
from astrai.extension.runtime.context import op_backend
from astrai.inference.core.cache import BlockPool, KVCacheManager
from astrai.inference.frontend.engine import InferenceEngine
from astrai.inference.worker.graph import CUDAGraphRunner
from astrai.inference.worker.workspace import InferenceWorkspace
from astrai.model import AutoModel, AutoRegressiveLM
from astrai.serialization import adapt_config
from astrai.tokenize import AutoTokenizer

_DTYPES = ["bfloat16", "float16", "float32"]
_CACHES = ["contiguous", "paged"]
_BACKENDS = ("cuda", "flash", "torch")


def _parse_kv_windows(value: Optional[str]) -> List[int]:
    if value is None or not value.strip():
        return []
    try:
        windows = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise click.BadParameter("KV windows must be comma-separated integers") from exc
    if any(window < 4 for window in windows):
        raise click.BadParameter("KV windows must be at least 4 tokens")
    if len(set(windows)) != len(windows):
        raise click.BadParameter("KV windows must not contain duplicates")
    return windows


# Default 1B GQA preset matching the project checkpoint architecture.
_DEFAULT_CONFIG = {
    "vocab_size": 100000,
    "hidden_size": 1536,
    "num_hidden_layers": 24,
    "intermediate_size": 6912,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
}


class BenchmarkResult:
    def __init__(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        tokens_per_second: float,
        latency_ms: float,
        metadata: Optional[Dict] = None,
    ):
        self.name = name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokens_per_second = tokens_per_second
        self.latency_ms = latency_ms
        self.metadata = metadata or {}


class GenerationBenchmark:
    def __init__(
        self,
        model: AutoModel,
        config: BaseModelConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_type: str = "contiguous",
        backend: str = "cuda",
        cuda_graph: bool = False,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.cache_type = cache_type
        self.model = model
        self.config = config
        self.backend = backend
        self.cuda_graph = cuda_graph
        self.tokenizer = tokenizer

    def _make_pool(self, batch_size: int, max_seq_len: int) -> BlockPool:
        if self.cache_type == "contiguous":
            n_tokens = None
        elif self.cache_type == "paged":
            # Keep the total token capacity equal to contiguous mode while
            # routing allocation through the shared paged pool.
            n_tokens = batch_size * max_seq_len
        else:
            raise ValueError(f"unsupported cache type: {self.cache_type}")

        return BlockPool(
            n_layers=self.config.num_hidden_layers,
            n_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.hidden_size // self.config.num_attention_heads,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=self.device,
            dtype=self.dtype,
            page_size=1,
            n_tokens=n_tokens,
        )

    @staticmethod
    def _make_workspace(pool: BlockPool, config: BaseModelConfig) -> InferenceWorkspace:
        return InferenceWorkspace(
            pool.max_batch_size,
            pool.max_seq_len,
            max_q_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            device=pool.device,
            dtype=pool.dtype,
        )

    @staticmethod
    def _make_task_cache(pool: BlockPool) -> KVCacheManager:
        return KVCacheManager(pool)

    def _run_prefill(
        self,
        pool: BlockPool,
        task_cache: KVCacheManager,
        batch_size: int,
        prompt_len: int,
        workspace: InferenceWorkspace,
        skip_lm_head: bool = False,
        prefill_chunk_size: Optional[int] = None,
    ) -> List:
        token_ids = torch.randint(
            0,
            self.config.vocab_size,
            (batch_size, prompt_len),
            device=self.device,
        )
        chunk_size = prefill_chunk_size or prompt_len

        request_ids = [f"bench_{i}" for i in range(batch_size)]
        for tid in request_ids:
            task_cache.alloc_slots(tid, list(range(prompt_len)))

        with torch.inference_mode(), op_backend(attention=self.backend):
            for start_pos in range(0, prompt_len, chunk_size):
                end_pos = min(start_pos + chunk_size, prompt_len)
                kv_cache = task_cache.bind(
                    request_ids,
                    workspace,
                    self.device,
                    start_pos=start_pos,
                    seq_ends=[end_pos] * batch_size,
                )
                self.model(
                    token_ids[:, start_pos:end_pos].reshape(-1),
                    kv_cache=kv_cache,
                    position_ids=torch.arange(
                        start_pos, end_pos, dtype=torch.long, device=self.device
                    ).repeat(batch_size),
                    fwd="prefill",
                    skip_lm_head=skip_lm_head,
                )
        torch.cuda.synchronize()
        return request_ids

    def _run_decode_step(
        self,
        pool: BlockPool,
        task_cache: KVCacheManager,
        request_ids: List,
        seq_len: int,
        workspace: InferenceWorkspace,
    ):
        batch_size = len(request_ids)
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size,), device=self.device
        )
        position_ids = torch.tensor(
            [seq_len] * batch_size, dtype=torch.long, device=self.device
        )
        for tid in request_ids:
            task_cache.extend_slots(tid, seq_len)
        kv_cache = task_cache.bind(request_ids, workspace, self.device)
        with torch.inference_mode(), op_backend(attention=self.backend):
            self.model(
                input_ids,
                kv_cache=kv_cache,
                position_ids=position_ids,
                fwd="decode",
            )

    def run_prefill_benchmark(
        self,
        batch_size: int = 4,
        prompt_length: int = 512,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        pool = self._make_pool(batch_size, prompt_length)
        workspace = self._make_workspace(pool, self.config)
        task_cache = self._make_task_cache(pool)
        request_ids = [f"bench_prefill_{i}" for i in range(batch_size)]
        for tid in request_ids:
            task_cache.alloc_slots(tid, list(range(prompt_length)))

        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (batch_size * prompt_length,),
            device=self.device,
        )
        position_ids = torch.arange(
            prompt_length, dtype=torch.long, device=self.device
        ).repeat(batch_size)
        kv_cache = task_cache.bind(request_ids, workspace, self.device, start_pos=0)

        for _ in range(3):
            with torch.inference_mode(), op_backend(attention=self.backend):
                self.model(
                    input_ids,
                    kv_cache=kv_cache,
                    position_ids=position_ids,
                    fwd="prefill",
                )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_trials):
            with torch.inference_mode(), op_backend(attention=self.backend):
                self.model(
                    input_ids,
                    kv_cache=kv_cache,
                    position_ids=position_ids,
                    fwd="prefill",
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * prompt_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="prefill",
            batch_size=batch_size,
            seq_len=prompt_length,
            tokens_per_second=tps,
            latency_ms=elapsed / num_trials * 1000,
            metadata={"benchmark_type": "prefill", "num_trials": num_trials},
        )

    def run_decoding_benchmark(
        self,
        batch_size: int = 4,
        prompt_length: int = 512,
        gen_length: int = 128,
        num_trials: int = 5,
        overlap: bool = False,
    ) -> BenchmarkResult:
        if self.tokenizer is None:
            raise ValueError("Engine decode benchmark requires a tokenizer")

        # Use the real engine so scheduler, executor, sampling, and graph
        # warmup/replay are included in the measured generation path.
        phrase = "Benchmark the language model with a realistic generation prompt. "
        prompt_ids = self.tokenizer.encode(
            (phrase * (prompt_length // 10 + 2)).strip()
        )[:prompt_length]
        prompt = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        max_seq_len = prompt_tokens + gen_length
        pool = self._make_pool(batch_size, max_seq_len)
        with op_backend(attention=self.backend):
            engine = InferenceEngine(
                model=self.model,
                tokenizer=self.tokenizer,
                max_batch_size=batch_size,
                max_seq_len=max_seq_len,
                cache=pool,
                enable_cuda_graph=self.cuda_graph,
                enable_overlap=overlap,
            )
        prompts = [prompt] * batch_size

        try:
            # Capture graphs and populate the allocator before timing. The
            # first request also includes model/scheduler startup effects.
            engine.generate(
                prompts, max_tokens=gen_length, temperature=0.0, do_sample=False
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()

            # Prefill probe: every timed trial regenerates the batch, so
            # the raw wall time amortizes one B×prompt_len prefill per
            # trial.  Time one max_tokens=1 generation — one prefill plus
            # a single decode step plus retirement, i.e. a slight
            # overcount of the prefill share — and report the per-step
            # decode cost with and without it.
            t_probe = time.perf_counter()
            engine.generate(prompts, max_tokens=1, temperature=0.0, do_sample=False)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            prefill_ms = (time.perf_counter() - t_probe) * 1000

            t0 = time.perf_counter()
            for _ in range(num_trials):
                engine.generate(
                    prompts, max_tokens=gen_length, temperature=0.0, do_sample=False
                )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        finally:
            engine.shutdown()

        steps = gen_length * num_trials
        decode_ms_total = max(elapsed * 1000 - num_trials * prefill_ms, 0.0)
        tokens = batch_size * gen_length * num_trials
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=gen_length,
            tokens_per_second=tokens / elapsed,
            latency_ms=elapsed / steps * 1000,
            metadata={
                "benchmark_type": "engine_decode",
                "num_trials": num_trials,
                "prompt_length": prompt_tokens,
                "kv_len_range": [prompt_tokens, prompt_tokens + gen_length],
                "decode_ms_per_step_excl_prefill": decode_ms_total / steps,
                "prefill_ms_per_trial": prefill_ms,
                "overlap": overlap,
                "op_selection": {
                    op: engine.op_selection.choice(op)[0] or "auto"
                    for op in ("attention", "rotary", "swiglu", "syrk", "symm")
                },
                "cuda_graph": engine.cuda_graph_enabled,
            },
        )

    def run_fixed_kv_decode_benchmark(
        self,
        batch_size: int,
        kv_len: int,
        window_steps: int,
    ) -> BenchmarkResult:
        if kv_len < 4:
            raise ValueError("kv_len must be at least 4 for graph warmup")
        if window_steps < 1:
            raise ValueError("window_steps must be positive")

        max_seq_len = kv_len + window_steps + 1
        pool = self._make_pool(batch_size, max_seq_len)
        workspace = self._make_workspace(pool, self.config)
        task_cache = self._make_task_cache(pool)
        warmup_len = kv_len - 3
        request_ids = self._run_prefill(
            pool,
            task_cache,
            batch_size,
            warmup_len,
            workspace,
            skip_lm_head=True,
            prefill_chunk_size=256,
        )

        b = batch_size
        input_ids_buf = torch.zeros(b, dtype=torch.long, device=self.device)
        position_ids_buf = torch.zeros(b, dtype=torch.long, device=self.device)
        gctx = CUDAGraphRunner(enabled=True)
        graph_key = (b,)

        def decode_graph_step(
            seq_len: int,
            start_event: Optional[torch.cuda.Event] = None,
            end_event: Optional[torch.cuda.Event] = None,
        ) -> None:
            input_ids_buf.copy_(
                torch.randint(0, self.config.vocab_size, (b,), device=self.device)
            )
            position_ids_buf[:] = seq_len
            if not all(task_cache.extend_slots_batch(request_ids, [seq_len] * b)):
                raise RuntimeError("failed to extend KV slots for fixed-KV benchmark")
            kv_cache = task_cache.bind(request_ids, workspace, self.device)
            with torch.inference_mode(), op_backend(attention=self.backend):
                if start_event is not None:
                    start_event.record()
                gctx.forward(
                    self.model,
                    key=graph_key,
                    input_ids=input_ids_buf,
                    kv_cache=kv_cache,
                    position_ids=position_ids_buf,
                    fwd="decode",
                )
                if end_event is not None:
                    end_event.record()

        for seq_len in range(warmup_len, kv_len):
            decode_graph_step(seq_len)
        torch.cuda.synchronize()

        start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(window_steps)
        ]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(window_steps)]
        for index, seq_len in enumerate(range(kv_len, kv_len + window_steps)):
            decode_graph_step(seq_len, start_events[index], end_events[index])
        torch.cuda.synchronize()
        elapsed_ms = sum(
            start.elapsed_time(end) for start, end in zip(start_events, end_events)
        )
        elapsed = elapsed_ms / 1000
        tokens = batch_size * window_steps
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=window_steps,
            tokens_per_second=tokens / elapsed,
            latency_ms=elapsed_ms / window_steps,
            metadata={
                "benchmark_type": "fixed_kv_decode",
                "kv_len_range": [kv_len + 1, kv_len + window_steps],
                "window_steps": window_steps,
                "timing": "cuda_event_model_forward",
                "cache": self.cache_type,
                "backend": str(self.backend),
                "cuda_graph": True,
            },
        )

    def _run_graph_decode_benchmark(
        self,
        batch_size: int,
        prompt_length: int,
        gen_length: int,
        num_trials: int,
    ) -> BenchmarkResult:
        max_seq_len = prompt_length + 5 + gen_length * num_trials
        pool = self._make_pool(batch_size, max_seq_len)
        workspace = self._make_workspace(pool, self.config)
        task_cache = self._make_task_cache(pool)
        request_ids = self._run_prefill(
            pool, task_cache, batch_size, prompt_length, workspace
        )

        b = batch_size
        input_ids_buf = torch.zeros(b, dtype=torch.long, device=self.device)
        position_ids_buf = torch.zeros(b, dtype=torch.long, device=self.device)

        gctx = CUDAGraphRunner(enabled=True)
        graph_key = (b,)

        def _decode_graph_step(seq_len):
            input_ids_buf.copy_(
                torch.randint(0, self.config.vocab_size, (b,), device=self.device)
            )
            position_ids_buf[:] = seq_len
            for tid in request_ids:
                task_cache.extend_slots(tid, seq_len)
            kv_cache = task_cache.bind(request_ids, workspace, self.device)

            with torch.inference_mode(), op_backend(attention=self.backend):
                return gctx.forward(
                    self.model,
                    key=graph_key,
                    input_ids=input_ids_buf,
                    kv_cache=kv_cache,
                    position_ids=position_ids_buf,
                    fwd="decode",
                )

        for i in range(5):
            _decode_graph_step(prompt_length + i)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(gen_length * num_trials):
            _decode_graph_step(prompt_length + 5 + i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * gen_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=gen_length,
            tokens_per_second=tps,
            latency_ms=elapsed / (gen_length * num_trials) * 1000,
            metadata={
                "benchmark_type": "decode",
                "num_trials": num_trials,
                "prompt_length": prompt_length,
                "cuda_graph": True,
            },
        )

    def _run_plain_decode_benchmark(
        self,
        batch_size: int,
        prompt_length: int,
        gen_length: int,
        num_trials: int,
    ) -> BenchmarkResult:
        max_seq_len = prompt_length + 5 + gen_length * num_trials
        pool = self._make_pool(batch_size, max_seq_len)
        workspace = self._make_workspace(pool, self.config)
        task_cache = self._make_task_cache(pool)
        request_ids = self._run_prefill(
            pool, task_cache, batch_size, prompt_length, workspace
        )

        for i in range(5):
            self._run_decode_step(
                pool, task_cache, request_ids, prompt_length + i, workspace
            )
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(gen_length * num_trials):
            self._run_decode_step(
                pool, task_cache, request_ids, prompt_length + 5 + i, workspace
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * gen_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=gen_length,
            tokens_per_second=tps,
            latency_ms=elapsed / (gen_length * num_trials) * 1000,
            metadata={
                "benchmark_type": "decode",
                "num_trials": num_trials,
                "prompt_length": prompt_length,
            },
        )


def print_benchmark_result(result: BenchmarkResult) -> None:
    print("-" * 80)
    print(f"{result.name.upper()} — Batch={result.batch_size}, SeqLen={result.seq_len}")
    print(f"  Throughput : {result.tokens_per_second:.1f} tokens/s")
    print(f"  Latency    : {result.latency_ms:.2f} ms/step")
    for k, v in result.metadata.items():
        if k != "benchmark_type":
            print(f"  {k.replace('_', ' ').title()}: {v}")
    print("-" * 80)


@click.command(name="benchmark", help="Benchmark model throughput and latency.")
@click.option("--device", default="cuda", help="Device.")
@click.option(
    "--dtype", type=click.Choice(_DTYPES), default="bfloat16", help="Data type."
)
@click.option(
    "--cache", type=click.Choice(_CACHES), default="contiguous", help="KV cache type."
)
@click.option(
    "--backend",
    type=click.Choice(_BACKENDS),
    default="cuda",
    help="Attention backend.",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Compare available attention choices side by side.",
)
@click.option("--batch_size", type=int, default=4, help="Batch size.")
@click.option("--prompt_length", type=int, default=512, help="Prompt length.")
@click.option("--gen_length", type=int, default=128, help="Generation length.")
@click.option("--num_trials", type=int, default=5, help="Number of trials.")
@click.option(
    "--kv-windows",
    default=None,
    help="Comma-separated KV lengths for fixed-context decode measurements.",
)
@click.option(
    "--window-steps",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
    help="Decode steps measured at each fixed-KV window.",
)
@click.option("--prefill_only", is_flag=True, help="Prefill benchmark only.")
@click.option("--decode_only", is_flag=True, help="Decode benchmark only.")
@click.option(
    "--cuda-graph/--no-cuda-graph",
    default=True,
    help="Enable or disable CUDA graph capture for engine decode.",
)
@click.option(
    "--overlap/--no-overlap",
    default=False,
    help="Enable the overlapped-commit pipeline for the engine decode benchmark.",
)
@click.option(
    "--ckpt",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Checkpoint directory. If omitted, a randomly-initialized model is "
    "built from --config or the default 1B GQA preset.",
)
@click.option(
    "--config",
    "config_path",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Optional model config JSON (used when --ckpt is omitted to define the "
    "architecture). Defaults to the 1B GQA preset.",
)
def benchmark_command(
    device: str,
    dtype: str,
    cache: str,
    backend: str,
    compare: bool,
    batch_size: int,
    prompt_length: int,
    gen_length: int,
    num_trials: int,
    kv_windows: Optional[str],
    window_steps: int,
    prefill_only: bool,
    decode_only: bool,
    cuda_graph: bool,
    overlap: bool,
    ckpt: Optional[str],
    config_path: Optional[Path],
) -> None:
    """Benchmark model throughput and latency."""
    dtype_map: Dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    kv_window_values = _parse_kv_windows(kv_windows)
    if kv_window_values and (prefill_only or not decode_only):
        raise click.BadParameter(
            "--kv-windows requires --decode-only and cannot be combined "
            "with --prefill_only"
        )
    if kv_window_values and not cuda_graph:
        raise click.BadParameter("--kv-windows requires CUDA graphs")

    if ckpt is not None:
        click.echo(f"Loading model from {ckpt} ...")
        config = ConfigFactory.load(
            adapt_config(
                json.loads((Path(ckpt) / "config.json").read_text(encoding="utf-8-sig"))
            )
        )
        model = AutoModel.from_pretrained(ckpt)
    else:
        raw = dict(_DEFAULT_CONFIG)
        if config_path is not None:
            raw.update(json.loads(config_path.read_text(encoding="utf-8-sig")))
        config = ConfigFactory.load(raw)
        model = AutoRegressiveLM(config)
        click.echo(
            f"Using randomly-initialized model "
            f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)"
        )

    tokenizer = (
        None
        if kv_window_values
        else AutoTokenizer.from_pretrained(ckpt or Path("params"))
    )

    model.to(device=device, dtype=dtype_map[dtype])
    model.eval()

    backends = _BACKENDS if compare else [backend]

    for name in backends:
        bench = GenerationBenchmark(
            model=model,
            config=config,
            device=device,
            dtype=dtype_map[dtype],
            cache_type=cache,
            backend=name,
            cuda_graph=cuda_graph,
            tokenizer=tokenizer,
        )

        click.secho(
            f"Benchmark: device={device} dtype={dtype} backend={name}", bold=True
        )

        if kv_window_values:
            for kv_len in kv_window_values:
                result = bench.run_fixed_kv_decode_benchmark(
                    batch_size=batch_size,
                    kv_len=kv_len,
                    window_steps=window_steps,
                )
                print_benchmark_result(result)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        else:
            if not decode_only:
                result = bench.run_prefill_benchmark(
                    batch_size=batch_size,
                    prompt_length=prompt_length,
                    num_trials=num_trials,
                )
                print_benchmark_result(result)

            if not prefill_only:
                result = bench.run_decoding_benchmark(
                    batch_size=batch_size,
                    prompt_length=prompt_length,
                    gen_length=gen_length,
                    num_trials=num_trials,
                    overlap=overlap,
                )
                print_benchmark_result(result)


if __name__ == "__main__":
    benchmark_command()
