# Callbacks and checkpoint state

Callbacks observe a TrainContext at lifecycle boundaries. Trainer dispatches
epoch, batch and optimizer hooks; TrainSession owns startup and teardown.

## Source modules

| Module | Responsibility |
| --- | --- |
| callbacks/base.py | Callback protocol and registry |
| callbacks/checkpoint.py | Atomic checkpoint publication |
| callbacks/metrics.py | Committed event history, validation and metric state |
| callbacks/gradient_noise.py | Passive gradient probes and scalar noise EMA |
| callbacks/metric_util.py | Stateless metric accessors and gradient norm |
| callbacks/optimization.py | Gradient clipping and activation checkpointing |
| callbacks/progress.py | Epoch progress display |
| optional_extras.py | Optimizer, scheduler, FP8 and strategy-owned extras |

## Event order

`on_train_begin` runs after context construction. A synchronous global update
calls batch hooks around each forward/backward, then `after_backward` once
the global gradients are normalized, before clipping. This hook receives
local and global loss normalizers. The noise probe observes local accumulated
gradient norms using read-only tensor post-accumulate hooks, which execute
before the native DDP reducer's node hooks. It does not replace the reducer
or its communication hook. Per-parameter FP32 squaring temporaries are reduced to scalars;
no model-shaped estimator state persists. Hooks exist only during
the final microbatch of a sampled update and are cleaned up on failure.

The default `before_optimizer_step` order is gradient clipping, metrics,
checkpoint callback, then progress. The clipping callback records the current
update's pre-clipping total norm, so metrics and progress see that value.
`after_optimizer_step` commits pending metric records and noise observations
before the checkpoint callback can publish them. A failed update does not
advance the persisted estimator. Async commits also notify metrics if
publication raises after the policy version has already advanced. `on_shutdown` discards pending observations;
session teardown closes workers before checkpoint publication.

## Persistence

`CheckpointCallback` writes weights and registered optimizer/scheduler/FP8 or
strategy extras. In the same staging directory, `MetricCallback` writes
`metric.jsonl` and `metric_state.json`; the manifest and atomic directory
rename cover both files. The JSON state identifies its schema version and
committed optimizer step. Its optional gradient-noise state contains EMA
scalars, sample count, last probe step, interval, decay and normalization.
It contains no model-shaped tensors or parameter object identities.

Only a full resume inherits metric history and estimator state. Warm starts
begin a new history. Older checkpoints without estimator state start a new
warmup and log that fact when noise sampling is requested. Incompatible
sampling settings, invalid scalar state, and mismatched checkpoint steps
are refused. Normal loss/LR logging requires no noise estimator.

RNG state is currently not included in checkpoint extras. Metric-history
continuity alone does not imply stochastic training replay.
