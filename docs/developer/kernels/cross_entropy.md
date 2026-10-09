# Cross-entropy kernels

The optional `cuda_linear_ce` path computes a bias-free LM head and
cross-entropy from hidden states in row chunks. User-facing selection and fallback behavior
are documented in the [training guide](../../guides/training.md#cross-entropy-backends).

## Contract

For valid token positions `V`, the CUDA path returns the sum of
cross-entropy terms, `sum(t in V, CE(logits[t], target[t]))`. The trainer
handles valid-token normalization, accumulation, and distributed scaling.
The linear path uses `logits[t] = hidden[t] @ weight.T` for a bias-free
head. Label smoothing and `ignore_index=-100` follow the Torch path.

## Implementation

The strategy computes the head and CE from the model's hidden states and an
LM-head weight view returned by the model. The view keeps the head visible to
DDP's forward-output traversal, including `find_unused_parameters=True`, while
the model remains independent of targets and loss configuration. Forward
generates logits one token chunk at a time, computes the loss, and replaces
private logits scratch with unscaled logits gradients. Each chunk projects
hidden and weight gradients into FP32 buffers during forward. Backward scales
and casts these immutable saved buffers without recomputing logits. Frozen
inputs omit their gradient buffer; inference only computes the loss.
GEMMs use ATen/cuBLAS; CE and gradient generation use native CUDA kernels.
There is no Liger, Triton or CUTLASS runtime dependency.

## Semantics and limits

- The kernel returns a **sum**; the trainer retains responsibility for valid
  token counts, gradient accumulation and distributed loss normalization.
- SFT masks use `ignore_index=-100`. All-masked sums and gradients are zero.
  Label smoothing in `[0, 1]` is supported.
- FP32 reductions do not imply bitwise Torch parity. Chunked CE changes GEMM
  tiling and summation order. Backward scales projected FP32 gradients before
  casting them to the model dtype. Saved gradients cost
  `4 * (tokens * hidden + vocab * hidden)` bytes when both inputs require grad;
  transient logits cost `chunk_size * vocab * sizeof(model_dtype)` bytes.
  Repeated backward with different scalar upstream gradients is supported.
- The kernel supports first-order gradients only. Keep `loss_backend: torch`
  for higher-order differentiation or debugging exact training trajectories.
- The direct wrapper accepts CUDA BF16/FP16/FP32 inputs. Autocast is supported for
  the linear wrapper. Noncontiguous inputs are made contiguous.
- DTensor weights retain the Torch computation; the kernel does not operate
  on local vocabulary shards. DDP and sequence context parallelism retain their collective and
  normalization boundaries.
- Inference remains unchanged and returns logits. The chunked strategy asks
  the model for hidden states and an LM-head parameter view, then returns
  `loss_sum` with `logits=None` from the strategy. The model does not receive
  labels or loss configuration.
- Chunked mode falls back to full Torch linear+CE on CPU, with a biased head,
  or without the compiled extension. It does not silently choose another
  chunk size or backend based on benchmark results.

## Build, test and measure

Build through the standard CUDA installation (`CSRC_KERNELS=true`) or the
configured CMake `cross_entropy` target. The loader discovers the module; it
is not registered as an attention implementation.

```bash
.venv/bin/python -m pytest tests/gpu/extension/test_cross_entropy.py tests/gpu/trainer/test_ce_backends.py -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark/cross_entropy.py \
  --mode head --warmup 20 --steps 100 --rounds 3 --out results/ce-head.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark/cross_entropy.py \
  --mode train --variants torch linear512 linear1024 \
  --batch-sizes 1 2 --warmup 20 --steps 100 --rounds 3 \
  --out results/ce-train.json
```

Use an idle GPU, lock clocks for the paired comparison and restore them after
the run. The benchmark restarts from identical checkpoint weights and optimizer
state for each variant, alternates variant order, and reports raw samples,
median/mean/p95, tokens/s, allocated/reserved peaks, and all-parameter differences.
It uses synthetic full-vocabulary tokens already resident on GPU; data loading,
checkpoint IO and multi-GPU communication are not part of its step timing.

Forward includes loss and gradient projection; backward scales saved gradients.
Compare complete steps, not the backward column alone. The speed
gate is at most 1% median and 2% p95 regression plus reduced allocated peak.
Numerical tests are a separate gate. Passing these short tests does not establish
long-run convergence equivalence; the CUDA backend remains opt-in.

For a separate numerical audit, pass `--deterministic`. It sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and enables deterministic Torch algorithms.
This can select slower model kernels; do not mix its timings with the default
performance runs. Repeated ordinary Torch runs also report their parameter
differences against the first Torch run to expose baseline nondeterminism.

For the maximum micro-batch at a fixed sequence length, use a separate capacity
probe. It completes forward, backward and optimizer steps (including state
allocation), records the allocated/reserved peaks, and stops each backend at its
first CUDA OOM. A passed short probe is not a long-run stability guarantee.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark/cross_entropy.py \
  --mode capacity --variants torch linear512 \
  --batch-sizes 3 4 5 6 7 8 --seq-len 2048 --warmup 2 --steps 3 \
  --memory-fraction 0.95 --out results/ce-capacity.json
```

The default 0.95 memory fraction leaves allocator headroom. Use 1.0 for a
separate limit probe on an idle GPU. The limit is micro-batch per device, not
the effective batch after gradient accumulation; changing sequence length,
optimizer, dtype, checkpointing or model changes the result.
