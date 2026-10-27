# Configuration normalization

`ComponentSpec` is the runtime source of optimizer type and constructor
parameters. The CLI converts optimizer-specific flags directly to that spec.
`optimizer_name` and `optimizer_hyperparameters` remain accepted as legacy
inputs only when an explicit `optimizer` spec is absent. An explicit spec wins
over conflicting legacy metadata, preserving the optimizer that training
actually constructs.

Legacy Mano parameter names are normalized at this input boundary. Historical
fixed AdamW/NAdamW metadata constants are descriptive fields, not constructor
arguments, and are omitted from the normalized parameters.

The legacy attributes and serialized metadata are derived from the spec. They
are no longer independently writable configuration fields. To change an
optimizer, replace `config.optimizer` with a new `ComponentSpec`; stale legacy
metadata cannot override it. Existing checkpoint type-name fallback remains
in the checkpoint reader.

Async rollout device format, uniqueness, worker counts and learner separation
are checked by one pure configuration function. Runtime setup additionally
checks GPU availability and the actual learner rank/device, then produces
`ResolvedAsyncRolloutConfig` for the coordinator. Direct coordinator callers
now pass `config=ResolvedAsyncRolloutConfig(...)`, along with the model,
tokenizer path, reward model and test/runtime dependencies. Worker-specific
rank, rendezvous and tensor-layout data remain in `RolloutWorkerSpec`.
