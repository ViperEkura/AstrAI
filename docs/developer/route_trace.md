# Versioned MoE route trace contract

`astrai.moe` defines `RouteTraceV0`, a compact behavior artifact for one
contiguous token span. The contract is intentionally separate from model
capture, rollout transport, expert dispatch, replay, and GRPO admission. Merely
importing or constructing these objects does not change a forward pass or the
training objective.

## Identity and semantics

A trace is interpretable only when all four components agree:

- `RouteIdentityV0` binds `policy_version`, model/checkpoint revisions, router
  and load-controller versions, and the router schema digest.
- `RouterSchemaV0` binds layer/expert/top-k geometry, EP world size, score and
  selected-weight semantics, expert and token ordering, padding, backend/kernel
  semantics, and a content-addressed parallel layout.
- `RouteTokenLayoutV0` identifies one sample, its full sequence and prompt
  boundaries, and the captured contiguous prompt/response span.
- `RouteTraceLevel` states whether the payload carries IDs only or IDs plus
  selected weights and top-k margin.

`RouterSchemaV0.fingerprint` is canonical JSON SHA-256. A trace whose identity
does not contain that exact digest is rejected at construction and decode time.
Consumers call `require_compatible_route_trace(...)` with their expected
identity, schema, token span, and minimum payload level. Every field is compared
exactly; there is no silent fallback for an unknown version or semantic field.

## Tensor layout

One trace represents one sample span:

```text
topk_ids          [tokens, moe_layers, top_k]
selected_weights  [tokens, moe_layers, top_k]  optional
topk_margin       [tokens, moe_layers]         optional
valid_mask        [tokens]                     optional
```

IDs use the narrowest unsigned storage capable of representing all experts:

```text
num_experts <= 256    -> uint8
num_experts <= 65536  -> uint16
otherwise             -> uint32
```

Use `pack_route_ids(...)` to range-check and compact an integer tensor. Valid
top-k rows must contain unique in-range experts. Invalid/padded rows must be
canonical zeros, making their serialized bytes deterministic and preventing
uninitialized padding from becoming part of behavior identity.

`IDS_WEIGHTS_MARGIN` requires both optional floating tensors. Selected weights
must be detached, finite, non-negative, and conform to the declared
`full-softmax-selected` or `selected-renormalized` semantics. Margins are the
non-negative top-k versus top-(k+1) score gap. The helper
`compute_topk_margin(...)` computes this diagnostic after an explicit FP32
conversion, including for BF16 near ties.

## Safe serialization

`RouteTraceCodecV0.dumps()` produces bytes suitable for an async rollout
buffer. It does not use pickle. The outer container is safetensors; a strict
canonical JSON manifest describes each logical tensor's dtype, shape, byte
count, and SHA-256. Unsigned types unsupported by older safetensors releases
are encoded as checked raw `uint8` buffers. The v0 wire format records and
requires little-endian tensor bytes rather than silently interpreting a payload
with the host's native byte order.

`RouteTraceCodecV0.loads()` returns detached CPU tensors and rejects:

- unknown/missing manifest or tensor fields;
- unknown codec or schema versions;
- malformed identity, schema, or token boundaries;
- tensor-name, dtype, shape, byte-count, or checksum mismatch;
- unsupported byte order, sparse/meta tensors, or non-canonical negative zero;
- payloads above the caller's byte budget;
- any trace invariant that construction would reject.

The codec digest is available as `trace.artifact_digest`. Artifact storage or
transport should retain that digest together with the policy/checkpoint
identity, but those integrations belong in follow-up changes.

## Codec microbenchmark

The repository includes a deterministic serialization-only benchmark:

```bash
python scripts/benchmark_route_trace_codec.py \
  --tokens 8192 --layers 40 --top-k 22 --num-experts 512 \
  --level ids --device cpu --warmups 5 --repeats 50
```

The JSON output separates logical tensor bytes, wire bytes, and an int64-ID
baseline, and reports serialize/deserialize latency. Construction happens
outside the timed region. The benchmark does not measure model capture, D2H
overlap, rollout transport, replay, training throughput, or end-to-end memory.

## Diagnostics only

`compare_route_traces(...)` requires exact router semantics, token layout,
validity mask, and device while allowing behavior/current versions to differ.
It reports:

- ordered top-k equality;
- unordered expert-set overlap and flip fraction;
- sparse selected-weight L1 shift when both traces carry weights;
- behavior margin mean and fragile-position fraction when margin is available.

Missing optional data produces `None`, never a fabricated zero. Empty valid
populations, mismatched layouts, and requested margin diagnostics without
margins fail closed. These scalars are observations only; this module never
rejects, downweights, or replays a rollout.

## Example

```python
import torch

from astrai.moe import (
    PaddingLayout,
    RouteIdentityV0,
    RouterSchemaV0,
    RouteTokenLayoutV0,
    RouteTraceCodecV0,
    RouteTraceLevel,
    RouteTraceV0,
    SelectedWeightSemantics,
    TokenSpanKind,
    canonical_json_digest,
    pack_route_ids,
)

schema = RouterSchemaV0(
    num_moe_layers=2,
    num_experts=64,
    top_k=2,
    expert_parallel_world_size=1,
    score_function="softmax-fp32-before-topk",
    score_dtype="float32",
    expert_id_ordering="score-descending-stable-index",
    selected_weight_semantics=SelectedWeightSemantics.SELECTED_RENORMALIZED,
    token_ordering="single-sequence-token-major",
    padding_layout=PaddingLayout.NONE,
    backend="astrai-torch",
    kernel_semantics_version="torch-topk-v1",
    parallel_layout_hash=canonical_json_digest({"placement": "all-local"}),
)
identity = RouteIdentityV0(
    policy_version=12,
    model_revision="model-12",
    router_state_version=12,
    checkpoint_revision="checkpoint-12",
    router_schema_hash=schema.fingerprint,
)
layout = RouteTokenLayoutV0(
    sample_id="opaque-sample-id",
    sequence_token_count=2,
    prompt_token_count=0,
    token_offset=0,
    token_count=2,
    span_kind=TokenSpanKind.FULL_SEQUENCE,
)
trace = RouteTraceV0(
    identity=identity,
    router_schema=schema,
    token_layout=layout,
    level=RouteTraceLevel.IDS,
    topk_ids=pack_route_ids(torch.tensor([[[0, 1]], [[2, 3]]]), 64),
)
wire_bytes = RouteTraceCodecV0.dumps(trace)
restored = RouteTraceCodecV0.loads(wire_bytes)
```

Before adding capture or replay, a follow-up must define how model revision,
checkpoint revision, router-state version, kernel semantics, token boundaries,
and parallel-layout digest are sourced atomically. That integration must remain
opt-in and preserve the current model path when disabled.

## Checkpoint-recompute route validation

`astrai.moe.compare_recompute_routes(...)` compares ordered top-k expert IDs
from the original forward and activation-checkpoint recomputation. It
normalizes integer storage to a canonical CPU representation and reports
content hashes plus mismatched layer, token-row, and top-k-slot counts. Shape,
negative-ID, duplicate-ID, and missing-layer ambiguities fail closed.

`GradientCheckpointingCallback` can attach one observer to each checkpoint
invocation. Its `route_validation` modes are:

- `off` (default): the existing non-reentrant checkpoint path is unchanged;
- `record`: compare routes and publish bounded counters into training metrics;
- `error`: publish the same counters and stop before the optimizer step if a
  mismatch, malformed observation, missing recomputation, or cross-rank
  observation-count inconsistency is present.

The observer uses the checkpoint API's separate forward/recompute contexts,
so concurrent or reverse-order backward graphs are paired by invocation rather
than by a global call counter. It stores only detached CPU top-k IDs until the
matching recomputation and then releases them. In distributed training, small
integer summaries are gathered on the initialized process group and every rank
receives the same decision. This is generic in world size; tests exercise two
and four Gloo ranks.

The opt-in path disables checkpoint early-stop for the wrapped invocation so
the post-forward observer always receives the recomputed route. It is therefore
incompatible with `torch.compile` in this version and has measurable overhead.
It does not select or replay experts and does not change gradients when routes
match.

Measure a local MoE forward/backward against the existing checkpoint path with:

```bash
python scripts/benchmark_moe_recompute_validation.py \
  --tokens 128 --dim 128 --dim-ffn 256 \
  --num-experts 8 --top-k 2 --device cpu \
  --warmups 3 --repeats 10
```

The report includes route hashes, mismatch counters, output/gradient parity,
and paired-path latency. It excludes route replay, optimizer work, distributed
training throughput, and end-to-end model performance.
