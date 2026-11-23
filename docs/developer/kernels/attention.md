# Attention

AstrAI exports one `attention` function from `astrai.extension`. Backend selection uses
`op_backend(attention=...)` for a strict context choice or `set_op` for a process default.

| Call | Tensor layout | Boundaries and behavior |
|---|---|---|
| `attention(q, k, v, *, mask=None, is_causal=False, scale=None)` | BSHD | Dense attention supports gradients. Boolean masks select visible keys and floating masks add to scores. |
| `attention(q, k, v, cu_seqlens_q=offsets, cu_seqlens_k=None, is_causal=False, scale=None)` | THD | Forward-only. Contiguous device int32 prefix sums delimit paired segments. Omitted KV offsets reuse Q offsets. |
| `attention(q, k, v, k_cache=kc, v_cache=vc, cache_seqlens=past, page_table=pages, cu_seqlens_q=None, mask=None, is_causal=False)` | BSHD or THD | Append the input K/V, update both caches in place, and return the attended Q rows. Packed queries require offsets; THD without offsets means one new token per request. |

All three paths support GQA. Rectangular causal attention aligns at the bottom right.
A fully masked row returns zero. The cache operator scales by `1 / sqrt(head_dim)`.
Native FP16 and BF16 kernels cover head dimensions 32, 64, 128 and 256.

## Dispatch and ownership

The public function resolves routing arguments, selects a backend with the
existing `AttentionDispatcher`, and executes it directly. There is no additional
execution wrapper or public mode parameter. Cache arguments take precedence over
rank; otherwise BSHD selects dense and THD selects varlen. Partial cache arguments,
dense offsets, and varlen masks are rejected before execution. Cache calls reject
independent KV offsets and custom scale. Gradient requirements are derived from
grad mode and Q/K/V, independently of the operation's layout. Backend capability
checks and strict/soft selection remain in this layer. CUDA, Torch and Flash
modules implement the selected calls.
The three stateless `kernel/attention.py` wrappers alone translate arguments
to the compiled ABI. They do not select backends or retain cache state.
The three native C++ entries own tensor validation before launching work:
dense checks and packing live in attention.cu, varlen in varlen.cu, and cache
in cache.cu. Shared native checks cover dtype, masks, integer bounds, scale and split settings.
Dense Q/stride extraction stays in attention.cu; cache.cu owns split workspace
allocation and its tensor lifetimes. entry.h contains only helpers used by
multiple native entries.
Python only rejects ambiguous routing options; it does not repeat native tensor
metadata checks. Torch reference entries retain the checks needed to run without
a compiled extension. Varlen prefix sums are a caller contract: start at zero, remain nondecreasing,
and end at the corresponding token total. Native varlen checks metadata without
launching a separate value-validation kernel. Torch reference execution checks
values while materializing segment indices; paged kernels retain their addressing guards.

GQA and MLA share the model-side scaled_dot_product_attention function, which binds the
current layer's K/V views, query offsets and workspace before calling attention.
They are adjacent in the model module; GDN and its state follow them. Extension
code does not import model cache classes.

## Query scheduling and KV addressing

Dense queries use the rectangular grid. The packed schedule finds each tile's
request from the device prefix sums. Varlen K/V uses contiguous token ranges.
Cache K/V uses physical pages, each shaped `[page_size, kv_heads, head_dim]`.
For the i-th appended token of request b:

```python
position = cache_seqlens[b] + i
block, offset = divmod(position, page_size)
physical_page = page_table[b, block]
```

The runner binds the batch's page table and past lengths once per step.
The workspace derives `q_len` from the scheduled token counts. Regular multi-token
batches use the common count. Packed batches use the total count and share
`cu_seqlens_q` across layers. Single-token prefill uses packed boundaries.
Decode reuses stable metadata, output and split buffers across graph replays.
The storage prepares each layer's K/V views once. Model inference selects
those retained views; gradient-enabled reference calls obtain fresh views. The cache type preserves one-token
prefill while all model calls use the same public attention entry.

## Attention tool responsibilities

| Tool | Responsibility | Ownership |
|---|---|---|
| `MmaOp` / `Mma16x8Layout` | Warp instruction, fragment sizes and lane mapping | No tensor or cache state |
| `KernelTraits` | Compile-time tile recipe and typed query/score/output fragments | No execution state |
| `SharedTileLayout` / `KVTileLoader` | Shared-memory placement and asynchronous tile copies | Borrows storage and an address policy |
| `AttentionMma` | Load query fragments, compute QK, accumulate PV | Borrows typed fragments |
| `AttentionMask` | Prepare row bounds and mask addresses once per query tile | Immutable visibility policy |
| `WarpSoftmax` | Online row maxima/sums and output rescaling | Owns two row states |
| `PrefillMainloop` / `DecodeMainloop` | Schedule loads, apply masks, compute QK/softmax/PV | Borrow shared memory and fill a result tile |
| `OutputEpilogue` / `SplitEpilogue` | Normalize and store final output or FP32 split partials | Borrow the completed tile |
| `AttentionStorage` / `AttentionTile` | Shared K/V buffers and register results | Owned by the kernel invocation |

The producer and consumer share one shared-memory layout. Fragment array sizes
come from the selected MMA atom instead of repeating register counts in every
kernel. Compile-time checks reject incompatible element widths, instruction
shapes and incomplete tiles. No runtime virtual dispatch or heap allocations
are introduced in the device path. The device entry `attention_kernel<Mainloop, Epilogue>` owns storage, runs the
selected mainloop, and stores valid tiles through its epilogue. Mainloops do
not include or call epilogues. Decode partials are merged by `split_kv_combine`.
This follows FA3's compile-time composition while retaining AstrAI's warp MMA,
copy pipeline, tile schedules and numerical recurrence.

`kernel/attention/mainloop.cuh` contains both prefill and decode mainloops;
`epilogue.cuh` contains output handling and split combination.
`kernel.cuh` defines their shared storage/result contract, the attention entry
and the paged KV append kernel. Prefill retains a separate append launch on the
same stream before attention, so all query blocks observe the new K/V.

For power-of-two page sizes, the KV address context computes the shift once
per request/head and reuses it in the tile loop. Other page sizes retain
integer division. This changes neither the public parameters nor the launch
specializations and does not depend on a GPU architecture.

## Native launch dispatch

The three operation launchers in `launcher/attention.cuh` select template
specializations and launch directly. There are no kernel wrapper classes,
query objects, launch plans, typed visitors or nested `detail` namespace.
This follows the direct templated launch structure of
[FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h);
AstrAI retains its own tile recipes and memory policies.

Each native entry checks BF16/FP16 support before dispatch, then directly calls
its launcher with `bf16` first and `fp16` otherwise. Attention does not use type lists, dtype
tags, recursive visitors or error-formatting callbacks.

Dense and varlen use `launch_attention_split_q` with different addressing.
Regular multi-token cache calls tile query positions and their grouped query heads
with grid axes (Q tile, KV head, request). Unequal request lengths use packed query
offsets. Both paths append new K/V before computing attention. Single-token decode
queries use `launch_attention_split_kv` with fused append. The split-KV
launcher selects fixed/adaptive splitting and direct/combined output itself.
Head dimension, causality, mask coverage and aligned loads retain their existing
specializations.

In ordinary decode, split `s` processes 16-token KV tiles numbered
`s, s + 64, s + 128, ...` within each request. Page-table capacity bounds the
launch grid. On replay, the live sequence length determines which splits have
work. The tile assignment and reduction order are independent of batch size,
batch slot and graph capacity. Rollout's fixed-size splits remain contiguous.

The standard workspace reserves 64 partial results per query head and batch
row. The native caller validates reusable buffers and allocates scratch when
a legacy 32-slot buffer cannot hold all the partials for the launch.

Paged decode selects aligned or unaligned new-token loads from host pointer
and stride metadata. The aligned kernel copies new K/V fragments in 16-byte
units; the other kernel uses scalar loads and stores for new K/V, retaining
arbitrary outer source strides. The last token's page slot is resolved once
per fragment. Only the
first GQA pass writes the cache; every pass reads new K/V directly, so no
cross-block write/read dependency is introduced.

Split merges with at most two parts retain the online recurrence. Larger
merges compute the common maximum, denominator and split weights independently
in each warp. Output dimensions read the weights through warp shuffles for FP32
accumulation, avoiding shared memory and block synchronization. Empty splits
contribute zero, including fully masked rows.

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
