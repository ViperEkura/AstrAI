# Rollout path

Rollout turns prompts into scored, versioned results for online objectives.
The backend chooses execution placement; generator formats and decodes;
runner scores and validates; Trainer passes the result to the strategy.

```mermaid
flowchart LR
    Setup[configure_rollout] --> Context[TrainContext.rollout]
    Context --> Runner[RolloutRunner]
    Trainer --> Runner
    Runner --> Generator[RolloutGenerator]
    Generator --> Backend[RolloutBackend]
    Runner --> Reward[BaseRewardModel]
    Runner --> Result[RolloutResult]
    Result --> Trainer
    Trainer --> Strategy[BaseStrategy]
    Runner --> Publishers[Replica publishers]
```

## Source modules

| Module | Responsibility |
| --- | --- |
| backend.py | Colocated or frozen replica execution and policy-version updates |
| rollout/types.py | RawRollout, RolloutResult, SamplingParams and reward-model contract |
| rollout/generator.py | Prompt preparation, grouped generation and decoding |
| rollout/runner.py | Reward scoring, versions, replay cache, replica publishers and validation evaluator |
| rollout/setup.py | Backend assembly and resolved async device/sampling settings |
| rollout/batching.py | Prompt slicing and ordered worker-result merge |
| rollout/async_round.py | Round scheduling, reply validation and prefetch |
| rollout/worker.py | Spawned frozen inference replica and request execution |
| rollout/protocol.py | Versioned Pipe messages and result envelopes |
| rollout/nccl_transport.py | Separate learner-to-worker weight broadcasts and ACKs |

## Placement and version boundary

ColocatedBackend shares the learner model; ReplicaBackend owns a frozen copy.
Both implement RolloutBackend so RolloutGenerator does not know where inference
runs. The policy snapshot boundary prevents generation from observing a
partially published update. RolloutRunner rejects stale versions before a
scored result reaches the learner. Trainer performs the actor update through
`apply_weight_update()` and invokes `publish_weights(version, model)` inside
that boundary. RolloutRunner holds the replica publishers; Strategy only sees
the prepared objective inputs. Online validation uses a separate RolloutEvaluator.

AsyncRoundCoordinator distributes prompts, merges results in input order and
checks request, round and policy identities. Worker processes keep their own
KV cache and CUDA Graph. The Pipe carries control and CPU results; a separate
NCCL group transports CUDA model weights. For startup, overlap, failure and
resume details, see [async_round.md](async_round.md).
