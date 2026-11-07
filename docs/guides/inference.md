# Inference

## Engine API

```python
from astrai import build_engine

engine = build_engine("./params")
try:
    print(engine.generate("Hello", max_new_tokens=128, do_sample=True, temperature=0.8))
    for text in engine.generate("Hello", stream=True, max_new_tokens=128):
        print(text, end="", flush=True)
finally:
    engine.shutdown()
```

A string prompt returns a string, or text deltas with `stream=True`.
A list of prompts returns a list, or `(prompt_index, text_delta)` pairs.
`generate_async()` yields text asynchronously; `generate_events()` yields
structured chunks with incremental `delta_token_ids`, text and terminal usage.
Consumers needing token history must accumulate those deltas.

Direct engine calls inherit the model's generation settings. Without a sampling
setting they use greedy decoding; `temperature` alone does not enable sampling.
Set `do_sample=True` explicitly. `max_tokens` aliases `max_new_tokens`;
passing both is an error. See [generation configuration](../developer/huggingface.md#tokenizer-and-generation-files)
for EOS and length precedence.

The HTTP adapter and demo scripts select sampling when temperature is positive
and greedy decoding when it is zero. Their defaults differ from direct engine calls.

## Cache and backend selection

The default contiguous cache reserves `max_batch_size × max_seq_len` slots.
Pass a finite `kv_tokens` to select paged allocation; `page_size > 1` also
enables prefix reuse. Sequence limits apply to prompt plus output.
Chunked prefill is disabled by default.

Automatic attention dispatch tries compatible CUDA, FlashAttention and Torch
implementations. Use `set_op("attention", "cuda")` for a soft default or
`op_backend(attention="cuda")` for a strict scope around engine construction.
The worker snapshots that selection when created.

For cache layout, scheduling, CUDA graphs and ownership, see
[inference internals](../developer/inference/README.md). Cache views are
`PrefillKVCache` and `DecodeKVCache`, defined on the model side in
[model/kv_cache.py](../../astrai/model/kv_cache.py).
For strict backend contracts, see [attention kernels](../developer/kernels/attention.md).

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
[Docker Serving](../developer/docker/serving.md).

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
are handled automatically by FastAPI with 422 status. An uninitialized engine
returns 503. Other request-processing failures can still propagate as server errors;
these statuses are not an exhaustive error taxonomy:

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

Example fields (additional timing statistics are also returned):

```json
{
    "total_tasks": 128,
    "total_tokens": 10240,
    "running": 3,
    "waiting_tasks": 2
}
```
