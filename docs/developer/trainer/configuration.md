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

Use one precision policy independently of optimizer placement:

~~~yaml
precision:
  compute_dtype: bfloat16
  gradient_dtype: float32
  master_dtype: float32
  optimizer_state_dtype: float32
  export_dtype: bfloat16
~~~

Compute, main gradients, persistent master weights and optimizer moments support
float32 or bfloat16. Gradient dtype model uses native parameter gradients.
Optimizer update arithmetic remains FP32; selecting BF16 master or moment
storage rounds those values after every step and can lose small updates.
Optimizer step counters remain FP32. FP32-only models with default FP32 storage
keep their native optimizer.

Library defaults preserve the model factory's compute dtype and native export
dtype (None); the CLI defaults compute to bfloat16. Other defaults are float32.
CLI flags such as --master_dtype and --export_dtype override individual YAML
precision fields. Old FP32 class aliases and model_dtype/executor gradient
configuration bridges have been removed.

An explicit export_dtype controls model.safetensors only. Float32 exports use
actual master weights when present. Checkpoints also save original compute
weights in training_model.pt, alongside masters and moments in their actual
storage formats. Full resume loads the original compute weights and requires
matching compute, gradient, master and state precision; export precision may
change. A warm start reads the exported model weights.

Set `optimizer.params.cpu_offload: true` to store masters and optimizer state
on CPU between steps. For a CUDA model, each parameter and its state are moved
to GPU, updated there using the GPU gradient, and copied back to CPU; the BF16
compute weights are refreshed on GPU. There is no CPU optimizer computation.
Persistent masters and state are pinned CPU tensors. Upload, GPU updates and
return copies run in separate CUDA streams, with at most two parameters staged
at once. A staging budget limits their combined master/state storage to a
quarter of device memory or half the available allocator headroom; one complete
parameter is always the minimum working set. Muon still updates complete
matrices. Device buffers remain alive until their consumer streams finish.

The pipeline drains completion events before step returns, so CPU checkpoints
can be saved immediately. It removes the repeated CPU-to-pinned staging copies
and the per-transfer stream synchronization. Loading state restores pinned
storage. The full CPU optimizer state is page-locked, rather than keeping only
one pinned scratch buffer. Place offload workers and their host memory on the
GPU-local NUMA node when measuring or launching a CPU-offloaded run.

CPU state storage is disabled by default and supports AdamW, SGD and
MuonAdamW/ShardedMuon with unsharded parameters. Other optimizers are rejected
because repeated per-parameter steps may change their semantics. Each DDP rank
retains its own CPU state; GPU state sharding is separate work. Staging limits
cross-parameter batching and does not overlap optimizer mutation with
forward/backward. GPU space must still fit the largest parameter, gradient,
state and compute scratch.

Optimizer checkpoints contain both the inner optimizer state and selected-format masters;
restoring copies the masters back to the compute weights. CPU and GPU placement
can change on restore. Optimizer checkpoints without masters or explicit precision metadata are rejected:
start a new run from their model weights with `load_state_dict.optimizer=False`.
Reconstructing masters from rounded BF16 weights cannot recover previous updates.

### Gradient accumulation

BF16 training with the none/DDP executors defaults to persistent FP32 main
gradients. Each backward contribution is copied or added into a parameter view
of a contiguous FP32 bucket, and the ordinary BF16 gradient is released.
Optimizer updates, global-objective normalization, clipping and gradient
metrics all consume the main gradient. FP32-only models keep native autograd
accumulation. This policy improves accumulation across backward calls; it does
not turn individual BF16 backward operators into FP32 computations.

Distributed BF16 training uses the same FP32 storage for accumulation and
asynchronous all-reduce, without additional BF16 DDP reducer buckets or per-step
packing. Buckets launch in deterministic reverse-parameter order as gradients
become ready during the synchronized backward. Earlier no_sync contributions
remain local. Parameters absent from the final microbatch still contribute if
they were used earlier in the accumulation window. The reduced gradient is
averaged over the configured process group before trainer normalization.

Set precision.gradient_dtype to model for native autograd/DDP accumulation,
or bfloat16 for explicit BF16 gradient storage. The default is float32. Set
find_unused_parameters=True for conditional distributed models: gradient
presence is then agreed across ranks, and globally unused parameters retain
None gradients so weight decay and momentum do not update them. The main-gradient path
supports dense replicated BF16/FP32 parameters, not DTensor/FSDP/TP, sparse
gradients, delayed DDP reduction or native DDP mixed_precision/device_mesh.
DDP gradient_as_bucket_view and static_graph hints need no separate buffers or
hook-learning phase with this path.

Custom runtime optimizers must declare supports_main_grad=True and consume
parameter_gradient/main_grad. An unaware optimizer is rejected before training,
rather than silently seeing empty BF16 gradients.

Optimizer zero_grad clears main-gradient presence (or zeros existing gradients
for set_to_none=False), while retaining reusable bucket storage. Checkpoints
remain update-boundary snapshots: save masters and optimizer state, not partial
gradient accumulation. FP32 master checkpoint and legacy-reset rules above
still apply. Complete optimizer-state GPU sharding remains separate work.

The design follows Megatron-LM's separation of compute parameters and main_grad
buffers: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel.py
