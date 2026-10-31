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

## Optimizer precision

BF16 models built through the trainer use persistent FP32 master parameters.
The optimizer updates these masters and keeps its floating-point state in FP32,
then copies the rounded weights back into the original model. FP32-only models
keep their existing optimizer unless CPU offload is requested.

Set `optimizer.params.cpu_offload: true` to keep masters, optimizer state and
step computation on CPU. CUDA transfers share one pinned buffer sized for the
largest FP32 parameter and run synchronously. This saves GPU memory at the cost of
host computation and transfers; it does not overlap with forward/backward.
CPU offload is disabled by default. It is particularly expensive for Muon's
Newton-Schulz matrix operations. Each DDP rank owns its own host optimizer.

Optimizer checkpoints contain both the inner optimizer state and FP32 masters;
restoring copies the masters back to the compute weights. CPU and GPU placement
can change on restore. Legacy optimizer checkpoints without masters are rejected:
start a new run from their model weights with `load_state_dict.optimizer=False`.
Reconstructing masters from rounded BF16 weights cannot recover previous updates.

This is an update-precision fix, not a complete mixed-precision policy: gradient
accumulation, reduction and clipping still use the model-side gradient dtype.
DTensor/FSDP/TP master parameters are explicitly unsupported until their layouts
and checkpoint restore are validated. Do not treat this as resolving all of #109.
