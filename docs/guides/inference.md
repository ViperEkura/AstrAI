# Inference

## Contents

- [KV Cache](#kv-cache)
- [KVCache System](#kvcache-system)
- [Attention Backend](#attention-backend)
- [Continuous Batching](#continuous-batching)
- [Sampling](#sampling-strategy-pattern)
- [Protocol Handlers](#protocol-handlers-strategy-pattern)
- [Engine & GenerateResult](#engine--generateresult)
- [HTTP API](#http-api) — endpoints, SSE, errors, stats
- [Engine API](#engine-api)

## KV Cache

At decode time, only the last query token matters. All previous K/V are cached to avoid recomputation:

$$
o_n = \sum_j \text{softmax}\left(\frac{q_n k_j}{\sqrt{d_k}}\right) v_j
$$

RoPE is applied **before** KV cache write, not after — otherwise position encoding drift occurs.

## KVCache System

Three-layer separation (SGLang-inspired): storage, index table, allocator.

```
PagePool (top-level manager, orchestrates all layers)
  ├── KVStorage              k_buffer / v_buffer [n_layers, size, n_kv_heads, head_dim]
  ├── ReqToTokenPool         req_to_token [num_reqs, max_ctx_len] → physical token slot
  ├── Allocator              bitmask-based page allocator + ref-count + LRU (paged mode only)
  └── RadixCache             exact, page-aligned prefix matching (paged mode, page_size > 1)
```

`PagePool` supports two modes:

- **Contiguous (default)**: pre-allocates `max_batch_size * max_seq_len` token slots. `req_to_token` is a trivial linear mapping (`slot = req_idx * max_seq_len + pos`). No dynamic allocation.
- **Paged** (`page_size=1` or `>1` with `n_tokens` set): shared token pool with on-demand allocation. `Allocator` provides ref-counted allocation and LRU eviction. When `page_size > 1`, `RadixCache` also enables prefix sharing.

`RadixCache` indexes complete token pages as parent-linked radix edges. Lookup walks from the root and compares each page's exact token tuple, so an identical page can only be reused under the same parent prefix. Hash values are retained for introspection, but never determine a match.

Only fully materialized KV pages enter the radix. A partial final page remains private to its request and is released when the request ends. On completion, the scheduler records the prompt plus generated tokens already decoded into KV; it excludes the final sampled token because that token has not yet passed through the model. A later request resumes prefill immediately after the longest complete-page hit.

`bind_tasks()` returns a `KVCache` dataclass — pure data, no methods:

```
KVCache
  ├── k_buffer, v_buffer     [n_layers, size, n_kv_heads, head_dim]
  ├── req_to_token           [num_reqs, max_ctx_len]
  ├── req_pool_indices       [batch_size]
  ├── seq_lens               [batch_size]
  ├── out_cache_loc          [batch, seq_len] — write indices for this forward
  ├── max_len                int — max(seq_lens), avoids GPU sync in decode
  ├── kv_indptr              [batch + 1] int32 — prefix sum of seq_lens, precomputed once per step
  ├── qo_indptr              [batch + 1] int32 — prefix sum of per-request q_lens (prefill), precomputed once per step
  ├── q_tile_to_batch        [num_q_tiles] int32 — prefill: Q tile → request (precomputed once per step)
  ├── q_tile_to_index        [num_q_tiles] int32 — prefill: Q tile → request-local tile index
  ├── decode_o_part          [batch, n_heads, head_dim] — decode split-K partial output buffer
  ├── decode_ml_part         [batch, n_heads] — decode split-K partial max/logsum buffer
  └── decode_out             [batch, n_heads, head_dim] — decode output accumulator
```

Attention layers do raw buffer indexing: `k_buffer[layer_id, out_cache_loc] = k` to write, `k_buffer[layer_id, indices]` to gather.

## Attention Backend

Inference code calls the policy API exported by `astrai.extension`. The
extension implementation is split into two layers:

- `astrai.extension.backend` owns capability checks, backend selection,
  fallback, and KV cache I/O.
- `astrai.extension.ops` contains direct wrappers around compiled CUDA kernels;
  these wrappers raise if a kernel is unavailable and do not fall back.

Attention computation (cache I/O + SDPA/kernel dispatch) is decoupled from the model via `AttentionBackend` ABC:

```
AttentionBackend (ABC)
  ├── CudaBackend          CUDA kernel dispatch (default on GPU)
  ├── FlashAttnBackend     Optional flash-attn dispatch (fallback)
  └── TorchNativeBackend   SDPA + indirect KV cache gather (always-available fallback)
```

Default priority is cuda > flash > torch. Automatic selection may choose a
compatible fallback for a particular call. Set
`ASTR_BACKEND=cuda|torch_native|flash` to override the default process-wide;
an explicit `attn_backend(...)` context still takes precedence over the env
override.

Select via context manager (mirrors `torch.nn.attention.sdpa_kernel`):

```python
from astrai.extension import attn_backend, ATTN_BACKEND

with attn_backend(ATTN_BACKEND.CUDA):
    engine.generate("hello")
```

Environment and context selections are strict: if the selected backend cannot
handle the call, inference raises an error rather than silently switching.

`CudaBackend` decode path: writes K/V via `new_k`/`new_v` while calling `attn_paged_decode` — the `req_to_token` table serves directly as the page table (conceptually a single-token "page" per slot, i.e. `page_size=1`; the op itself takes no `page_size` argument). No explicit K/V gather needed.

`CudaBackend` prefill path: writes K/V, then calls `attn_paged_prefill` — a ragged-batch (paged) prefill kernel that reads K/V directly from the flat pool via `req_to_token`, addressing each request's `q_len`/`kv_len` through `qo_indptr` and `kv_indptr`. No explicit K/V gather needed.

The scheduler packs requests with the same prefix-cache start position and
attention backend into one prefill forward even when their prompt lengths differ.
Requests with different prefix hit lengths remain separate batches.

Fallback: when `CudaBackend` cannot handle an input (wrong dtype or head_dim), `FlashAttnBackend` is tried next (if installed), then `TorchNativeBackend`.

This fallback is performed by the public `attention(...)` policy entry point
only when no backend was explicitly selected. Import from
`astrai.extension.ops` only for direct kernel tests or when failure on a missing
kernel is the intended behavior.

### Rotary Embedding Backend

Rotary embedding is applied via `apply_rotary_emb` in `astrai/extension/backend/rotary.py`, which auto-dispatches:

- **CUDA kernel** (`rotary_emb.cu`): fused cos/sin lookup + rotation in a single kernel, used when the kernel is available, the input is bf16 on CUDA, and `torch.is_grad_enabled()` is `False` (inference mode)
- **Torch fallback**: complex multiply path (`torch.view_as_complex` → `torch.complex` multiply → `torch.view_as_real`), used during training (supports autograd backward) or when the CUDA kernel is not available

`RotaryEmbedding` stores a cos/sin table `freqs_cis` of shape
`[max_len, dim/2, 2]` (f32 — `[cos, sin]` pairs) and `forward()` returns
a `[batch, seq_len, dim/2, 2]` slice indexed by `position_ids`. Both
attention backends share the same rotary dispatch — it is backend-agnostic.

## Continuous Batching

`InferenceScheduler` runs a daemon thread with a 4-phase loop:

```
1. Cleanup → Record complete materialized pages, then release task-owned KV resources
2. Refill  → Pop from waiting_queue, task_alloc resources, activate
3. Prefill → Group by (prompt_len, start_pos), run full forward
4. Decode  → Run single-token forward for each same-position group
```

For in-process training rollout, `InferenceScheduler.update_weights(version)`
acknowledges that the shared model was updated in place. Versions are monotonic;
the scheduler rejects updates while requests are queued and invalidates reusable
prefix KV pages before exposing the new version. Synchronous `run_batch()` and
weight updates are serialized so a generation cannot straddle two versions.

### Releasing Inference Runtime Memory

Colocated training can keep the shared model weights resident while returning
inference-only allocations to CUDA between rollout phases:

```python
engine.release()  # cancels requests; drops KV, workspace, and CUDA graphs
# run the memory-heavy training phase
engine.resume()   # rebuilds the original runtime and restarts the scheduler
```

`InferenceScheduler.release()` is idempotent and preserves both model weights
and `policy_version`. It first stops the scheduling loop, cancels queued work,
and invalidates reusable prefix pages. `resume()` reconstructs the cache and
executor with the original batch/sequence bounds; a scheduler that was running
before release is restarted automatically. New generation requests fail with a
clear error while the runtime is released.

The lifecycle is available only when the scheduler owns its cache (the default).
An externally injected `PagePool` has ambiguous ownership, so release fails
before changing state rather than invalidating a cache that another component
might share.

## Sampling (Strategy Pattern)

```
BaseSamplingStrategy (ABC)
  ├── TemperatureStrategy
  ├── TopKStrategy
  ├── TopPStrategy
  └── SamplingPipeline
```

`SamplingPipeline` composes them: Temperature → Top-K → Top-P → softmax → multinomial.  
`sample()` is a convenience shortcut for one-shot usage.

## Protocol Handlers (Strategy Pattern)

```python
class ProtocolHandler:  # concrete orchestrator
    def __init__(self, request, engine, builder): ...
    async def handle(self):
        prompt, ctx, stops = builder.prepare(request, engine)
        agen = engine.generate_async(prompt, ...)
        if stream: self._handle_stream(agen, ctx, stops)
        else:      return await self._handle_non_stream(agen, ctx, stops)
```

`ResponseBuilder` (ABC): `prepare()`, `format_stream_start()`, `format_chunk()`, `format_stream_end()`, `format_response()`.

`OpenAIResponseBuilder` → `/v1/chat/completions`, `AnthropicResponseBuilder` → `/v1/messages`.

Adding a protocol = one builder file, no handler subclassing needed.

## Engine & GenerateResult

```
InferenceEngine
  ├── generate(prompt, stream, ...) → str | List[str] | Generator
  ├── generate_async(prompt, ...)   → AsyncGenerator
  ├── get_stats()                   → Dict
  ├── release() / resume()          → bool
  └── shutdown()
```

Use `scripts/tools/benchmark_inference_lifecycle.py` to measure reclaimed memory,
release/resume latency, and greedy-output parity for the AstrAI 1B preset. The
[L20 results](../benchmarks/inference_release_resume_l20.md) include raw JSON for
2K, 8K, and 32K context bounds.

`GenerateResult` uses `Condition` for non-streaming (`wait_completion()`) and `Event` for streaming (`wait()`). Stream callback is `cb(token)`.

## Launching the Server

`scripts/tools/server.py` accepts every option as a CLI flag or from a YAML
config file (`--config serve.yaml`); explicit CLI flags override YAML values.
The YAML `server:` section mirrors the flags:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  device: cuda
  dtype: bfloat16
  max_batch_size: 16
  max_seq_len: null
```

```bash
python scripts/tools/server.py --config serve.yaml
python scripts/tools/server.py --config serve.yaml --port 9000  # CLI wins
```

In Docker, `scripts/serve.sh` drives the same YAML (a `runtime:` section
controls ports/GPU/mounts); see
[Docker Serving](../developer/docker-serving.md).

## HTTP API

```
POST /v1/chat/completions   OpenAI
POST /v1/messages            Anthropic
GET  /health                 {"status":"ok","model_loaded":true}
GET  /stats                  scheduler statistics
```

### OpenAI

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

Response:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1717000000,
  "model": "astrai",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
}
```

Streaming SSE: `object: "chat.completion.chunk"` — starts with role delta, then token chunks, ends with finish chunk + usage stats, then `data: [DONE]`.

### Anthropic

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"astrai","system":"You are helpful.","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

Supports `stop_sequences` and streaming via `event: content_block_delta`. Anthropic streams also end with the shared `data: [DONE]` sentinel after `event: message_stop`.

### Request Parameters

The HTTP protocols and direct engine API have distinct request models and defaults.

**OpenAI** (`ChatCompletionRequest`):

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | `"astrai"` | Model name returned in responses |
| `messages` | List[dict] | required | Chat messages (role, content) |
| `temperature` | Optional[float] | 1.0 | Sampling temperature (0.0-2.0) |
| `top_p` | Optional[float] | 1.0 | Nucleus threshold (0.0-1.0) |
| `top_k` | Optional[int] | 50 | Top-k count |
| `max_tokens` | Optional[int] | 2048 | Max generation length |
| `stream` | Optional[bool] | False | Stream output |
| `stop` | Optional[Union[str, List[str]]] | None | Stop sequences |
| `n` | Optional[int] | 1 | Accepted for API compatibility, **ignored** (always returns a single choice) |
| `presence_penalty` | Optional[float] | 0.0 | Accepted for API compatibility, **ignored** |
| `frequency_penalty` | Optional[float] | 0.0 | Frequency penalty (-2.0 to 2.0) |
| `logit_bias` | Optional[Dict[int, float]] | None | Accepted for API compatibility, **ignored** |
| `user` | Optional[str] | None | Accepted for API compatibility, **ignored** |
| `tools` | Optional[List[ToolDef]] | None | Tool definitions for function calling |
| `tool_choice` | Optional[Union[str, Dict[str, Any]]] | `"auto"` | Tool selection mode or explicit tool choice |

> `n`, `presence_penalty`, `logit_bias`, and `user` are validated by the request
> model but ignored by the server (a warning is logged when a non-default value
> is supplied).

**Anthropic** (`MessagesRequest`):

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | `"astrai"` | Model name returned in responses |
| `messages` | List[AnthropicMessage] | required | User/assistant messages |
| `system` | Optional[str] | None | System prompt |
| `max_tokens` | int | 1024 | Max generation length |
| `temperature` | Optional[float] | 1.0 | Sampling temperature (0.0-2.0) |
| `top_p` | Optional[float] | 1.0 | Nucleus threshold (0.0-1.0) |
| `top_k` | Optional[int] | 50 | Top-k count |
| `stream` | Optional[bool] | False | Stream output |
| `stop_sequences` | Optional[List[str]] | None | Stop sequences |

### SSE Streaming Format

**OpenAI** (`/v1/chat/completions`, `stream=true`):

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"astrai",
       "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":0,"model":"astrai",
       "choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"astrai",
       "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}

data: [DONE]
```

**Anthropic** (`/v1/messages`, `stream=true`):

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","model":"astrai","role":"assistant",
       "content":[],"usage":{"input_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{...}}

event: message_stop
data: {"type":"message_stop"}

data: [DONE]
```

### Error Responses

The server returns standard HTTP status codes. Pydantic validation errors (e.g. missing required fields)
are handled automatically by FastAPI with 422 status. The only application-level error is engine initialization:

| Status | Meaning |
|--------|---------|
| 200 | Success |
| 422 | Unprocessable entity (Pydantic validation) |
| 503 | Service unavailable (model not loaded, engine not ready) |

Error response body (503):

```json
{
    "detail": "Engine not initialized"
}
```

### Stats Endpoint

```
GET /stats
```

Response:

```json
{
    "total_tasks": 128,
    "total_tokens": 10240,
    "active_tasks": 3,
    "waiting_queue": 2
}
```

## Engine API

```python
# Non-streaming
engine.generate("Hello", stream=False)          # -> str
engine.generate(["A", "B"], stream=False)       # -> List[str]

# Streaming
engine.generate("Hello", stream=True)           # -> Generator[str]
engine.generate(["A", "B"], stream=True)        # -> Generator[Tuple[int, str]]

# Async
async for token in engine.generate_async("Hello", ...):    # -> AsyncGenerator[str]
    print(token)
```

> Document Update Time: 2026-08-22
