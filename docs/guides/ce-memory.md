# Memory-efficient cross entropy

The default remains Torch. Pretraining (`seq`) and SFT accept these strategy
options through the existing `TrainConfig.strategy_kwargs` field:

```yaml
strategy_kwargs:
  loss_backend: cuda_ce
```

`cuda_ce` retains model-dtype logits and computes CE reductions in FP32 without
materializing full FP32 logits or log-softmax tensors. This is the first option
to measure on RTX 5090. CPU and unavailable-extension runs use Torch.

For a bias-free `AutoRegressiveLM` head, an additional experimental option is:

```yaml
strategy_kwargs:
  loss_backend: cuda_linear_ce
  loss_chunk_size: 512
```

This computes the head and CE inside the model forward, so DDP sees the head
parameter in the returned loss graph. Forward generates logits one row chunk
at a time and saves only per-row normalization statistics. Backward recomputes
vocabulary tiles, replaces each private tile with scaled logits gradients, and
reduces all tokens in one GEMM per weight-gradient tile. This avoids a full
FP32 head-gradient buffer and repeated BF16 accumulation across token chunks.
The smaller hidden-gradient buffer accumulates across vocabulary tiles in FP32.
GEMMs use ATen/cuBLAS; CE and gradient generation use native CUDA kernels. There is
no Liger, Triton or CUTLASS runtime dependency.

## Semantics and limits

- Both kernels return a **sum**; the trainer retains responsibility for valid
  token counts, gradient accumulation and distributed loss normalization.
- SFT masks use `ignore_index=-100`. All-masked sums and gradients are zero.
  Label smoothing in `[0, 1]` is supported.
- FP32 reductions do not imply bitwise Torch parity. Chunked CE changes GEMM
  tiling and summation order. Backward applies the upstream scale before casting
  logits gradients to the model dtype, matching the Torch operation order.
  Extra persistent statistics cost `8 * tokens` bytes; the FP32 hidden-gradient
  accumulator costs `4 * tokens * hidden` bytes (12 MiB at 2048 x 1536).
- Both kernels support first-order gradients only. Keep `loss_backend: torch`
  for higher-order differentiation or debugging exact training trajectories.
- Direct wrappers accept CUDA BF16/FP16/FP32 inputs. Autocast is supported for
  the linear wrapper. Noncontiguous inputs are made contiguous.
- DTensor weights in the chunked path and DTensor logits in the CE path retain
  the Torch computation; these kernels do not operate on local vocabulary
  shards. DDP and sequence context parallelism retain their collective and
  normalization boundaries.
- Inference remains unchanged and returns logits. An explicit training
  `loss_targets` argument to `AutoRegressiveLM.forward` returns `loss_sum` and
  sets `logits=None`. The chunked strategy requires this model interface.
- Chunked mode falls back to full Torch linear+CE on CPU, with a biased head,
  or without the compiled extension. It does not silently choose another
  chunk size or backend based on benchmark results.

## Build, test and measure

Build through the standard CUDA installation (`CSRC_KERNELS=true`) or the
configured CMake `cross_entropy` target. The loader discovers the module; it
is not registered as an attention implementation.

```bash
.venv/bin/python -m pytest tests/extension/test_cross_entropy.py tests/trainer/test_ce_backends.py -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark/cross_entropy.py \
  --mode head --warmup 20 --steps 100 --rounds 3 --out results/ce-head.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark/cross_entropy.py \
  --mode train --variants torch cuda_ce linear512 linear1024 \
  --batch-sizes 1 2 --warmup 20 --steps 100 --rounds 3 \
  --out results/ce-train.json
```

Use an idle GPU, lock clocks for the paired comparison and restore them after
the run. The benchmark restarts from identical checkpoint weights and optimizer
state for each variant, alternates variant order, and reports raw samples,
median/mean/p95, tokens/s, allocated/reserved peaks, and all-parameter differences.
It uses synthetic full-vocabulary tokens already resident on GPU; data loading,
checkpoint IO and multi-GPU communication are not part of its step timing.

Forward includes loss work; chunked backward includes logits recomputation.
Compare complete steps, not the backward column alone. The speed
gate is at most 1% median and 2% p95 regression plus reduced allocated peak.
Numerical tests are a separate gate. Passing these short tests does not establish
long-run convergence equivalence; no backend is enabled by default by this PR.

For a separate numerical audit, pass `--deterministic`. It sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and enables deterministic Torch algorithms.
This can select slower model kernels; do not mix its timings with the default
performance runs. Repeated ordinary Torch runs also report their parameter
differences against the first Torch run to expose baseline nondeterminism.
