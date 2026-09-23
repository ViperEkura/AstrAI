# Attention

> Kernel modules `csrc/kernels/attention/` (module list in the
> [family overview](README.md#overview)); python adapters in
> `astrai/extension/ops/attention.py`, dispatch policy in
> `astrai/extension/backend/attention.py`.

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
