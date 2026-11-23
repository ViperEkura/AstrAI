# Inference Architecture

Inference separates user-facing text processing, request scheduling and device
execution. The layers run **in one process**: colocated rollout shares the
trainer's model, while a serving instance uses a background engine thread.
Separate rollout replicas are placed by the training backend.

## Three-layer class relationships

The inference engine has three layers: **Frontend → Core → Runner**.
Scheduling and KV management belong to Core; workspace and graph execution
belong to Runner. Runner is implemented in the `worker/` package.

Network adapters are entrypoints around Frontend. Model/cache tensor layouts
are dependencies used by Runner, and `contracts.py` carries shared data.
These are not additional inference-engine layers.

```mermaid
classDiagram
    direction TB
    namespace Frontend {
        class InferenceEngine
        class InputProcessor
        class OutputProcessor
        class RequestTracker
    }
    namespace Core {
        class EngineCore
        class Scheduler
        class StepPlanner
        class KVCacheManager
        class PolicyVersionGuard
    }
    namespace Runner {
        class GPUModelRunner
        class InferenceWorkspace
        class CUDAGraphRunner
        class PendingExecution
    }

    InferenceEngine --> InputProcessor : prepare inputs
    InferenceEngine --> OutputProcessor : decode events
    InferenceEngine *-- RequestTracker : track consumers
    InferenceEngine --> EngineCore : submit and cancel
    EngineCore *-- Scheduler : schedule and apply results
    EngineCore *-- PolicyVersionGuard : serialize operations
    Scheduler *-- StepPlanner : plan windows
    Scheduler *-- KVCacheManager : allocate and release KV
    EngineCore *-- GPUModelRunner : execute plans
    EngineCore --> PendingExecution : materialize results
    GPUModelRunner --> PendingExecution : return handles
    GPUModelRunner *-- InferenceWorkspace : device buffers
    GPUModelRunner *-- CUDAGraphRunner : capture and replay
    KVCacheManager --> BlockPool : reserve physical pages
    InferenceWorkspace --> BlockPool : gather execution rows
```

| Layer | Owns | Why it is separate |
| --- | --- | --- |
| Frontend | Input processing, consumer tracking, output decoding and text stops | Text APIs can change without altering scheduling; tokenized rollout can enter core directly |
| Core | Execution loop, request queues, planning, KV allocation and policy versions | Scheduling, accepted output and resource lifetime need coordinated state ownership |
| Runner | Input tensors, model execution, sampling, workspace, graphs and deferred results | Device optimizations do not decide request completion or logical KV release |

The source import direction is **frontend → core → worker → model/cache ABI**;
all layers can use `contracts.py`. `tests/inference/test_layering.py` rejects
reverse runtime imports. Execution snapshots carry cache slots. The workspace
binds them to model tensors using the shared pool. See [cache binding](core.md#cache-binding-boundary).

## Design patterns in use

These names describe existing responsibilities; they do not require additional
base classes, registries or forwarding layers.

| Pattern | Current implementation | Benefit and constraint |
| --- | --- | --- |
| Facade | `InferenceEngine.generate`, `generate_events` and `score` | One text API hides registration and cleanup; scheduling stays in core |
| Adapter | OpenAI/Anthropic network modules | Translate protocols around the same frontend; no per-protocol scheduler |
| Event sink | `OutputEventSink` implemented by `RequestTracker` | Core emits token/terminal events without importing consumers; this is one sink, not a general event bus |
| Immutable execution messages | `SchedulerOutput` and `ModelRunnerOutput` | Identity and version travel with delayed work; the worker cannot mutate a live `Request` through a plan |
| Deferred result | `PendingExecution` | Submission and host materialization are separate; `commit()` resolves data, while the scheduler applies request state |
| Resource reuse | `InferenceWorkspace`, `ResultRing`, `CUDAGraphRunner` | Stable buffers reduce allocation and copying; their reuse must wait for the relevant execution to retire |

`EngineCore` coordinates these collaborators directly. There is no need for a
second executor facade while execution remains local. `StepPlanner` is a small
planning collaborator, not a scheduler plugin hierarchy. Sampling is a function,
and is not represented as a class in the diagrams.

## Cooperation across one execution

```mermaid
sequenceDiagram
    participant F as Frontend
    participant T as RequestTracker
    participant C as EngineCore
    participant S as Scheduler
    participant W as GPUModelRunner
    participant P as PendingExecution
    F->>F: Tokenize and assign request IDs
    F->>T: Register consumers
    F->>C: Submit prepared inputs
    C->>S: Admit and enqueue
    loop Until completion or cancellation
        C->>S: schedule()
        S-->>C: SchedulerOutput (identity, version, windows)
        C->>W: execute_model(plan)
        W-->>C: Handle for each submitted group
        C->>P: commit() during drain
        P-->>C: ModelRunnerOutput
        C->>S: update_from_output(result)
        S-->>T: Accepted tokens, then terminal event
    end
    T-->>F: Events for decoding and usage
```

This shows logical order. Supported decode overlap may submit the next forward
before materializing the previous result. The distinction between **planned**,
**materialized** and **accepted** progress is therefore necessary:

- A planned window reserves resources but does not prove its KV has been written.
- Materialization waits for execution/copy completion. Even a prefill chunk
  without a sampled token needs this boundary.
- The scheduler validates identity, order and policy version before accepting
  output. Cancellation can discard a result without making its resources safe
  to reuse early.

## State ownership and lifetime

| State | Owner | Release or reuse boundary |
| --- | --- | --- |
| Consumer queues and completion notifications | RequestTracker | Unregister after collection or early exit |
| Live requests and accepted tokens | Scheduler | Terminal state plus retirement of in-flight executions |
| Request-to-KV allocation and prefix index | KVCacheManager / BlockPool | Free after retirement; publish only completed prefix pages |
| Outstanding execution queue | EngineCore | Drain or safely retire every submitted handle |
| Latest decode relay and sampling state | GPUModelRunner | Reuse only for the same ordered request identities and valid KV binding |
| Host copy slots and device buffers | ResultRing / InferenceWorkspace | Completion and consumption before overwrite |
| Model version and mutation lock | PolicyVersionGuard | Stopped/drained engine, successful mutation and KV invalidation |

The core's pending queue accounts for all submitted groups. The runner's latest
decode handle supports relay and overlap checks; it is not a second request
queue. Likewise, consumer completion and GPU resource retirement are different
milestones. Merging these states solely because their names look similar would
erase required lifetime boundaries.

## Current scope and tradeoffs

The useful vLLM-style separation is scheduling versus execution, with explicit
plan/result messages. AstrAI keeps direct in-process calls for shared-model
training; it has no EngineCore/worker RPC or process executor.

All prefill start-offset groups share one token budget per scheduler step.
Decode tokens are scheduled separately. The runner submits separate forwards
by phase and prefill start offset. Chunking remains off by default.

Both allocation policies preallocate GPU storage. `kv_tokens=None` selects
contiguous allocation; a finite value selects paged allocation, with radix
prefix reuse when `page_size > 1`. Paged admission currently reserves the
prompt's pages. Progressive admission and recompute preemption are not present.

## Layer details

| Document | Scope |
| --- | --- |
| [Frontend](frontend.md) | Facade, input/output processing and consumer tracking |
| [Core](core.md) | Driver, scheduling, KV ownership and policy versioning |
| [Runner](worker.md) | Execution messages, deferred results and device resources |
| [Events](events.md) | Cancellation, terminal delivery and shutdown |
