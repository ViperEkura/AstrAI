# Attention

> Kernel modules `csrc/attention/` (module list in the [family overview](README.md#overview)); 
> python adapters in `astrai/extension/kernel/attention.py`, dispatch policy in `astrai/extension/backend/attention/`.

## Attention Backend

`astrai/extension/backend/attention/` provides the backend abstraction:

- **`AttentionBackend`** (ABC): single abstract `forward`; each subclass branches on `fwd` ("decode" / "prefill" / None) internally, `_check_fwd` guards unknown modes
- **`CudaBackend`**: CUDA kernel dispatch — decode via `attn_paged_decode` (page_size=1), prefill via `attn_paged_prefill` (ragged batch, `qo_indptr` + `kv_indptr`). Default on GPU.
- **`FlashAttnBackend`**: Optional flash-attn dispatch via `flash_attn_varlen_func` over gathered flat K/V.
- **`TorchNativeBackend`**: SDPA with indirect KV cache gather (always-available fallback)

Default priority: cuda > flash > torch. Set ``ASTR_BACKEND=cuda|torch_native|flash``
to override the default.

Select a backend via context manager (mirrors `torch.nn.attention.sdpa_kernel`):

```python
from astrai.extension import attn_backend, ATTN_BACKEND

with attn_backend(ATTN_BACKEND.CUDA):
    engine.generate("hello")
```

The `attention(...)` policy entry point falls back to `FlashAttnBackend` (when
flash-attn is installed and supports the call) or `TorchNativeBackend` when the
automatically selected CUDA backend cannot handle an input. Resolution
precedence is: explicit `attn_backend(...)` context > `ASTR_BACKEND` env >
default. An explicit `attn_backend(...)` selection is strict and raises instead
of silently switching implementations; the env override (and the implicit
default) fall back to the first compatible backend when incapable. Training
calls (`fwd=None`, no KV cache) resolve by capability: the CUDA cache kernels
cannot run without a cache, so they fall back to flash (mask-free/causal calls
only) and finally to torch SDPA.

## Python Wrappers

`astrai/extension/kernel/attention.py` provides Python wrappers for each compiled attention kernel. Each wrapper calls its CUDA kernel directly and raises `RuntimeError` if the `.so` is not available. Fallback to torch SDPA is handled by the attention backend, not the wrapper functions.


Interface (all functions):
```
is_causal: True = causal mask; False = non-causal
mask:      2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool, True=keep)
```

Layout convention: dense q/k/v use `[batch, seq_len, n_heads, head_dim]` (BLHD); packed inference uses `[total_q, n_heads, head_dim]`. `scale=None` selects `1/sqrt(head_dim)`. The public backend accepts any finite scale; native CUDA currently accepts finite positive scales, and automatic selection routes other scales to a capable backend.

## Q Scheduling and KV Addressing

Prefill separates Q work scheduling from KV storage:

- `DenseQSchedule` maps a rectangular grid directly with
  `batch = blockIdx.z` and `q_tile = blockIdx.x`.
- `PackedQSchedule` consumes a compact work map for a packed
  `[total_q, q_heads, head_dim]` tensor.
- `ContigKV` and `PagedKV` only provide KV lengths and translate logical KV
  positions into physical addresses. They do not schedule Q blocks.

For ragged Q lengths `[70, 10, 130]` and 64 rows per Q tile, cache binding
builds:

```text
qo_indptr       = [0, 70, 80, 210]
q_tile_to_batch = [0, 0, 1, 2, 2, 2]
q_tile_to_index = [0, 1, 0, 0, 1, 2]
```

Paged prefill folds query heads and token rows into one packed row space:

```text
G = q_heads / kv_heads
BLOCK_M = 16 * WARPS
blocks_per_host_tile = G * HOST_Q_TILE_ROWS / BLOCK_M
grid = (num_q_tiles * blocks_per_host_tile, kv_heads, 1)
packed_row = token_row * G + head_in_group
```

Each block resolves its request using the host tile map and its packed row
range inside that tile. The query schedule handles dense or packed rows;
`ContigKV` and `PagedKV` resolve storage addresses. Both use the same tiled
online softmax and tensor-core computation.

## Attention tool responsibilities

| Tool | Responsibility | Ownership |
|---|---|---|
| `MmaOp` / `Mma16x8Layout` | Warp instruction, fragment sizes and lane mapping | No tensor or cache state |
| `KernelTraits` | Compile-time tile recipe and typed query/score/output fragments | No execution state |
| `SharedTileLayout` / `KVTileLoader` | Shared-memory placement and asynchronous tile copies | Borrows storage and an address policy |
| `AttentionMma` | Load query fragments, compute QK, accumulate PV | Borrows typed fragments |
| `AttentionMask` | Prepare row bounds and mask addresses once per query tile | Immutable visibility policy |
| `WarpSoftmax` | Online row maxima/sums and output rescaling | Owns two row states |
| split-Q / split-KV kernels | Compose the tools and orchestrate the pipeline | Own register/shared-memory storage |

The producer and consumer share one shared-memory layout. Fragment array sizes
come from the selected MMA atom instead of repeating register counts in every
kernel. Compile-time checks reject incompatible element widths, instruction
shapes and incomplete tiles. No runtime virtual dispatch or heap allocations
are introduced in the device path. These tools remain in existing headers.

## Native launch dispatch

Attention follows the same typed query, plan, and launch structure as GEMM,
in the existing `launcher/attention.cuh`:

1. `with_prefill_kernel` / `with_decode_kernel` select the head dimension
   and mask specialization once. Prefill also selects causality. Paged
   prefill selects a mask-coverage specialization: full masks omit redundant
   extent checks, while partial masks retain them. Cache capacity and total
   query rows provide conservative bounds without device reads. Dense prefill
   retains the checked mask kernel for stable compiler code generation.
2. The selected kernel type builds a plan from host metadata. Decode uses a
   `DecodePlanQuery` containing grid dimensions, KV tile count and cached
   occupancy. `make_decode_plan` is a pure split policy.
3. The torch entry validates or allocates scratch using that plan, then the
   same kernel type launches it. Native harnesses use the same visitor and
   planner without depending on torch.

Plans contain launch geometry and output mode, never tensor addresses or
device sequence lengths. Every call plans from its current shape; residual
split fields in `AttentionParams` do not bypass planning. Paged decode uses
the host cache capacity, so device lengths can change during graph replay.
The existing tile recipes, occupancy reference and split limits are retained.
Scratch still uses the kernel's `MAX_SPLITS` stride, and caller-supplied
buffers are validated even when direct output needs no scratch.

Paged decode selects aligned or unaligned new-token loads from host pointer
and stride metadata. The aligned kernel copies new K/V fragments in 16-byte
units; the other kernel uses scalar loads and stores for new K/V, retaining
arbitrary outer source strides. The last token's page slot is resolved once
per fragment. Only the
first GQA pass writes the cache; every pass reads new K/V directly, so no
cross-block write/read dependency is introduced.

Split merges with at most two parts retain the online recurrence. Larger
merges compute the common maximum, denominator and split weights once per
head with one warp; output dimensions reuse these weights for FP32
accumulation. Empty splits contribute zero, including fully masked rows.

## Parameter boundaries

The native parameter struct groups pointers first, then shapes, strides and
control values, using natural alignment. Mask metadata stays directly in
`AttentionParams`: `mask_b_stride`, `mask_h_stride` and `mask_l_stride`
follow the tensor stride names (`l` is the query sequence axis), while
`mask_k_len` and `mask_q_len` bound the key and query axes. Separate bounds
preserve partial paged masks and singleton query broadcast.

- Callers choose `is_causal`, `mask` and `scale`. Dense alignment comes from
  Q/K lengths; packed alignment comes from each request's device metadata.
- Shapes and strides are extracted from tensors at entry. Native tensor layout
  selection remains available; Python wrappers use BLHD consistently.
- Page-table indices and Q tile maps belong to the cache/workspace binding.
  They remain on device during CUDA Graph replay; no CPU length readback is
  required. Tile maps cannot be inferred from packed Q shape alone.
- Decode scratch tensors are a paired execution resource, retained until both
  launches are submitted. Output buffers allow stable CUDA Graph addresses.
  Split counts and direct-output selection are planned internally.
- With lower-right alignment, a single decode query sees the full KV sequence.
  Decode therefore shares one kernel for causal and noncausal calls.

## Interface contract and execution ownership

- Causality is an explicit boolean. No caller-supplied position offset is needed.
  Dense queries align to the end of K/V: `query_position =
  kv_len - q_len + query_row`. Packed queries use each request's lengths.
  Fully masked rows return zero. The torch adapter constructs the same mask
  for unequal lengths instead of relying on SDPA's upper-left alignment.
- Boolean masks mean `True=keep`. Layouts are `[batch, key]`,
  `[batch, query, key]`, or `[batch, head, query, key]`; singleton batch,
  head and query axes broadcast. In packed inference, query rows are local to
  the current request chunk, while key columns index absolute cache positions.
  Missing rows or columns in a short mask are invisible. A two-dimensional
  model padding mask stays on the key axis and uses `is_causal=True`, so each
  cached chunk gets the correct causal offset. Torch also supports additive
  floating masks. FlashAttention rejects custom masks before updating the
  cache. The mask pointer is the single source of truth for selecting native
  masked kernels.
- `scale` is forwarded unchanged through backend selection and execution.
  Unsupported native scales are rejected explicitly, never replaced by the
  default. Existing external backends receive their original arguments when
  the caller does not supply a scale.
- Cached decode can append the current K/V in the attention kernel. The
  caller reserves the destination and supplies sequence metadata including
  that token. Attention does not advance the scheduler's sequence lengths.
- Decode planning uses shape, the host KV capacity, and cached device occupancy
  before allocating workspace. It does not inspect device sequence lengths.
  One split writes the normalized output directly for head dimensions up to
  128; dimension 256 retains the measured faster graph path. Other calls retain
  the existing partial and combine kernels.
- The native entry owns temporary workspace tensors until all launches are
  enqueued on the current CUDA stream. Explicit scratch buffers must be passed
  together, be contiguous FP32 tensors, and share Q's device. Graph callers
  retain their persistent buffers and output; changing device-side request
  lengths does not change the captured launch plan.

## Implementation plan and performance gates

1. Establish the shared contract and retain the existing fast kernel variants.
   Separate split planning from launches, fix workspace ownership, and add
   direct output for one split. Cover masks, signed causal positions, custom
   scale, irregular layouts, fused append and graph replay. This is the
   current implementation stage.
2. Extend prefill configuration using measured query/key tile sizes, warp
   counts and pipeline stages. Keep query scheduling and KV addressing as
   compile-time policies. Check register and shared-memory limits for each
   architecture; FA SM120 configurations are candidates for measurement.
3. Add native training forward/backward together: forward retains FP32 LSE;
   backward reconstructs probability tiles, computes row `dot(dO, O)`, and
   accumulates dQ/dK/dV without storing the full attention matrix. Aggregate
   GQA gradients over query-head groups. Register autograd and FakeTensor
   behavior through the torch dispatcher before enabling compiled training.
4. Evaluate a separate multi-warp decode variant for wider heads and long KV.
   Its split policy must include combine overhead and preserve current fast
   variants for shapes where the candidate does not improve performance.

Before changing a default, compare baseline and candidate in one process in
baseline/candidate/candidate/baseline order for three rounds, at matching
precision, inputs and GPU clocks. Measure eager calls, graph execution and
model-level training or inference separately. A reproducible regression in
any representative shape blocks the default switch; averages do not hide it.
Keep raw latency measurements per GPU. Eight-GPU training throughput estimates
must be labelled as assuming linear scaling.

Reference scope:

- Implemented: [FA2 split launch](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_fwd_launch_template.h)
  omits the combine launch when there is only one split. AstrAI uses direct
  output for measured favorable head dimensions; its kernels were not copied.
- Architecture check: [FA SM120 adapter](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/flash_fwd_sm120.py)
  confirms SM80-style MMA and the smaller shared-memory capacity. No FA tile
  configuration or pipeline was ported in this stage.
- Planned only: [FA backward](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/csrc/flash_attn/src/flash_bwd_kernel.h)
  informs the LSE-based backward design in step 3. No native attention backward
  was added here.
- Planned only: [PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
  describes dispatcher, autograd and FakeTensor integration for step 3. This
  stage keeps the existing pybind interface.


## Gated DeltaNet reference path

Set `attention.default_type` to `"gdn"` for a model whose layers all use
the differentiable Gated DeltaNet reference module. A minimal autoregressive
attention section is:

```json
{
  "attention": {
    "default_type": "gdn",
    "num_heads": 16,
    "num_kv_heads": 4,
    "gdn": {
      "num_key_heads": 16,
      "num_value_heads": 32,
      "key_head_dim": 128,
      "value_head_dim": 128,
      "conv_kernel_size": 4
    }
  }
}
```

For a hybrid decoder, `attention.layers` lists the type for each layer using
`"gdn"` or `"gqa"`. Its length must equal `num_hidden_layers`.

During training, an unpacked batch without padding passes no mask. GQA uses
the causal backend path and GDN's recurrence is causal by construction.
A boolean right-padding mask is passed as `[batch, seq]`. The model prepares
one mask per attention type: a causal query/key mask for GQA or MLA, and the
original two-dimensional padding mask for GDN. The same mask is reused by
layers of that type. Packed batches use a four-dimensional document-boundary
mask for softmax attention. GDN cannot use that mask: its recurrent and
convolution states would need a reset at every document boundary. Until
boundary resets are implemented, train any model containing GDN on unpacked
sequences; the model rejects packed document masks before entering the blocks.

The GDN head counts and dimensions default to `attention.num_heads` and
`hidden_size / attention.num_heads` when omitted. Key heads are repeated across
value heads and must divide the value-head count. The module applies causal
depthwise local convolution and L2-normalizes query/key vectors without RoPE,
then updates a per-batch, per-value-head state with decay followed by the
delta-rule write. A per-head gated RMSNorm is applied to the recurrent output
before projection:

```text
S_t = decay_t * S_(t-1)
     + beta_t * k_t * (v_t - S_(t-1) * decay_t * k_t)^T
o_t = q_t^T * S_t
```

Parameterization follows the Qwen reference: `decay = exp(-exp(A_log) *
softplus(a + dt_bias))` with `A = exp(A_log) ~ U(0, 16)` and `dt_bias = 1`, the
convolution is a single depthwise pass over the concatenated q/k/v with
`bias=False`, the output gate is `silu`, and `q` is scaled by `1/sqrt(K)` after
normalization.

## GDN operators (training and inference)

`astrai/model/components/gdn_ops.py` holds the three interfaces over the one
rule. Training uses the chunked operator; the other two exist to check it and to
run inference:

| Operator | Sequential steps | Used by |
| --- | --- | --- |
| `chunk_gated_delta_rule` | `O(T/chunk_size)`, matmuls inside a chunk | training, `GDN.prefill` |
| `recurrent_gated_delta_rule` | `O(T)`, one step per token | agreement reference |
| `recurrent_gated_delta_rule_step` | single token | `GDN.decode_step` |

Chunk size defaults to 64 (the kernel convention) and any positive value must
give the same result; the operator is checked against the per-token path at 16,
32, 64 and 128. Causality comes from the lower-triangular decay mask, so the
intra-chunk tensors are `[B, H, T/chunk_size, chunk_size, chunk_size]` — linear
in `T`, never `T x T`.

Interface contract:

- `chunk`/`recurrent` take `[B, T, H, D]` q/k/v and `[B, T, H]` gates, and
  return `[B, T, H, V]`. `_step` takes a single token as `[B, H, D]` / `[B, H]`.
- The state is `[B, H, K, V]` float32 in all three, starting at zeros. Its shape
  does not depend on the number of steps, so decode memory is constant in the
  history length.
- All math is float32 internally; outputs are cast back to the input dtype.
- Training accepts right padding because no later valid token can observe a
  padded update; outputs at padded positions are zeroed. Prefill does not
  accept padding when returning a state for continued decoding.

`GDN.prefill` and `GDN.decode_step` carry a `GatedDeltaNetState` (convolution window plus
recurrent matrix) and are validated to reproduce the training forward token for
token. `GDN.forward` still rejects a KV cache: the paged cache stores
token-indexed K/V, not a recurrent matrix, and wiring a state pool is not done.
Packed-document boundaries are not interpreted either — `prefill` refuses a
padding mask rather than folding pad rows into the state.

Numerical agreement: the chunked and per-token paths agree to ~1e-6 absolute in
float32 on these shapes, and their input gradients agree to ~1e-4.

This PyTorch reference path supports right-padding masks in training, rejects
non-right padding, and does not yet have a paged state pool or ReplaySSM-style
decode bandwidth reduction.

## Gated DeltaNet CUDA kernels

`csrc/gated_deltanet/` holds the kernels that had to be CUDA. The module
is `gated_deltanet` and exposes `gated_deltanet_fwd` and `gated_deltanet_bwd`; the
Python wrappers live in `astrai/extension/kernel/gdn.py`, where `gdn_fwd` mirrors the
entry point it calls.

### `gated_deltanet_fwd`: preparation

The chunked kernels want head-major tensors (`[B, H, T, D]`, head dim
contiguous) with L2-normalized query/key rows and the gate pre-scanned into a
chunk-local cumsum. What the layer's projections plus the local convolution hand
over is `[B, H, D, T]` — head and head-dim outer, *token inner* — so going from
one to the other is a real transpose, not a relabeling. Done with torch, that
work (transposes, dtype casts, the two L2 norms and the cumsum) cost more than
every GDN kernel combined: 53% of the operator's wall clock at T=2048 and 59% at
T=8192, because each piece was its own launch over the whole tensor.

The kernel reads the source tile with the token axis coalesced, stages it in
shared memory and reads it back transposed so the store's head-dim axis is
coalesced too; the L2 norm reduces down each token column in the same pass. The
gate scan is a second launch (a chunk-wide Hillis-Steele scan, one block per
`(b, h, chunk)`). Measured against the torch steps it replaces: **8.9x at T=2048,
9.7x at T=8192**, landing within bf16 rounding of them.

Two traps this kernel produced, both worth keeping:

- **The batch stride is not `H * D * T`.** q/k/v are slices of one concatenated
  convolution buffer (channels = q + k + v), so their batch stride spans all
  three. Assuming `H * D * T` is invisible at batch 1 — `stride(0)` is unused when
  `size(0) == 1` — and silently wrong from batch 2 on. The strides now come from
  the caller.
- **A scan needs a barrier before its first read.** Writing shared memory and
  immediately reading a neighbour without `__syncthreads()` between races; it
  showed up as a 2% relative error in the gate rather than as garbage, because the
  stale values were usually close.

It requires head dim 128, fp32 gates, the projection layout and a sequence length
that is a multiple of the chunk; anything else is rejected rather than guessed at.

### `gated_deltanet_bwd`: the output stage's backward

One of the four stages of the reverse pass, and the first written (FLA's
`chunk_bwd_o`). Given the output gradient it produces `dq`, `dk`, `dv_new`, the
per-chunk state gradient and the gate gradient. `dh` and `dv_new` are written by
more than one stage of a full reverse pass, so both are float32 accumulators this
kernel adds into with atomics; every other output belongs to exactly one block.

All five gradients are checked against autograd of the same stage in torch, with
a *dense* state: a symmetric one hides an axis mix-up in the `d_qe` product, and
that is exactly the bug this test caught. Runtime is 12.9x faster than torch
autograd at T=2048 and 35.9x at T=8192, at 5.8 TFLOP/s — about 19% of this part's
float32 peak, because the kernel is plain FMA with a 96.5 KB shared-memory
footprint that permits one block per SM. Tensor cores and a smaller tile are the
obvious next step.

The other three stages — the inter-chunk state recurrence, the `w`/`u` sandwich
with the UT solve's backward, and the chunk-local cumsum — are not written yet,
and the reference `chunk_gated_delta_rule` remains the only complete trainable
path. Autograd through the reference is the ground truth for checking a complete
backward kernel; a separate backward wrapper is unnecessary.
