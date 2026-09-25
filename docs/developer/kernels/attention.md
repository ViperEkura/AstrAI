# Attention

> Kernel modules `csrc/kernels/attention/` (module list in the [family overview](README.md#overview)); 
> python adapters in `astrai/extension/ops/attention.py`, dispatch policy in `astrai/extension/backend/attention.py`.

## Attention Backend

`astrai/extension/backend/attention.py` provides the backend abstraction:

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

`astrai/extension/ops/attention.py` provides Python wrappers for each compiled attention kernel. Each wrapper calls its CUDA kernel directly and raises `RuntimeError` if the `.so` is not available. Fallback to torch SDPA is handled by the attention backend, not the wrapper functions.


Interface (all functions):
```
is_causal: True = causal mask; False = non-causal
mask:      2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool, True=keep)
```

Layout convention: all q/k/v are `[batch, seq_len, n_heads, head_dim]` (blhd). Scale is always `1/sqrt(head_dim)`.

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

Paged prefill launches (MMA path, GQA head packing):

```text
grid.x = num_q_tiles * HB   # HB = min(G, WARPS): q heads packed per block
grid.y = kv_heads * ceil(G / HB)
grid.z = 1
```

The tensor-core prefill kernel packs `HB = min(G, WARPS)` query heads of one
kv-head group into a block, so K/V tiles stream once per block instead of once
per q head (~HB× less global K/V traffic). Warp `w` handles head slot `w / WPH`
and 16-row chunk `w % WPH`, where `WPH = WARPS / HB`; `G = q_heads / kv_heads`
and `G = 1` (MHA) degenerates to the historical one-head-per-block layout.
Each host Q tile (64 rows, `Q_TILE_ROWS`) splits into `HB` packed blocks along
`grid.x`. Each block resolves its request and request-local row range in O(1):

```cpp
host_tile = blockIdx.x / HB;
batch = q_tile_to_batch[host_tile];
row_base = q_tile_to_index[host_tile] * 64 + (blockIdx.x % HB) * (64 / HB);
```

The kernel then uses `qo_indptr[batch]` for the packed Q base and adjacent
`qo_indptr` / `kv_indptr` entries for that request's Q and KV lengths. This
avoids the previous per-block linear scan over the batch, shared-memory
broadcast, mapping barrier, and upper-bound grid with potentially invalid
blocks.

## Gated DeltaNet reference path

`attn_type: "gdn"` selects the differentiable recurrent Gated DeltaNet
reference module, not an SDPA backend. A minimal autoregressive config is:

```json
{
  "attn_type": "gated_deltanet",
  "gated_deltanet_num_key_heads": 16,
  "gated_deltanet_num_value_heads": 32,
  "gated_deltanet_key_head_dim": 128,
  "gated_deltanet_value_head_dim": 128,
  "gated_deltanet_conv_kernel_size": 4
}
```

The head counts and dimensions default to `num_attention_heads` and
`hidden_size / num_attention_heads` when omitted. Key heads are repeated across
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
- Padded rows carry `beta = 0` and `g = 0`, so right padding neither writes to
  nor decays the state and the returned state matches an unpadded run.

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

`csrc/kernels/gated_deltanet/` holds the kernels that had to be CUDA. The module
is `gated_deltanet` and exposes `gated_deltanet_fwd` and `gated_deltanet_bwd`; the
Python wrappers live in `astrai/extension/ops/gdn.py`, where `gdn_fwd` mirrors the
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
path. `gdn_ops.chunk_gated_delta_rule_backward` is autograd through that
reference: it is the ground truth a backward kernel is checked against, not a
second implementation to maintain.
