# Training

Use [Quick Start](../quick-start.md) for the first run and
[CLI reference](params.md) for options. This guide covers objective selection,
precision, rollout and checkpoint behavior.

## Training Loop

Trainer groups the epoch permutation into global optimizer windows. The default
`epoch_tail_policy: drop` records and skips an incomplete tail; `flush` updates
using the real sample and target counts. CP and TP ranks do not multiply the
data count. Online replay publishes only after the complete data window.

See [lifecycle](../developer/trainer/lifecycle.md) for startup and cleanup,
[callbacks](../developer/trainer/callbacks.md) for metric commit boundaries,
and [accumulation](../developer/internals.md#gradient-accumulation-mechanics)
for normalization and step counts.

## Strategies

| Training type | Objective and data |
| --- | --- |
| `seq` | Next-token cross-entropy over token sequences |
| `sft` | Cross-entropy over targets selected by `loss_mask` |
| `dpo` | Preferred/rejected responses with aligned masks and a frozen reference |
| `grpo` | Grouped responses, masks, rewards and behavior log-probabilities |
| `online_grpo`, `online_dpo` | Generate and score responses through rollout |
| `online_ppo` | Online actor-critic updates with GAE and a separate critic optimizer |

Use [preprocessing](preprocessing.md) for on-disk schemas,
[loss formulas](../developer/internals.md#training-loss-formulas) for the
objectives, and [strategy contracts](../developer/trainer/strategy.md) for
programmatic batches and capability declarations.

### Cross-entropy backends

SEQ and SFT use Torch cross-entropy by default. For a bias-free
`AutoRegressiveLM` head, configure the experimental fused linear CE backend
with the CLI options below to avoid materializing full head logits.
Python callers put these keys in `TrainConfig.strategy_kwargs`:

```yaml
training:
  loss_backend: cuda_linear_ce
  loss_chunk_size: 512
```

The chunked path falls back to Torch on CPU, with a biased head, or without
the extension. The CUDA path returns a loss sum; the trainer normalizes by
valid tokens across accumulation and distributed workers. SFT masking with
`ignore_index=-100` and label smoothing are supported. Use the Torch backend
for higher-order gradients or exact trajectory debugging. DTensor weights continue to use the Torch computation.

See [cross-entropy kernels](../developer/kernels/cross_entropy.md) for the
implementation contract, numerical limits, and benchmark procedure.

### GRPO options

Online rollout carries aligned `logprobs_old`, avoiding a second old-policy
forward. Offline callers without these values must synchronize `old_model`
between generation rounds. Only masked response targets contribute to the loss.

`loss_aggregation="token"` weights valid tokens equally; `"sequence"` weights
responses equally. `clip_eps_low` and `clip_eps_high` override the symmetric
clipping interval. Optional `overlong_max_len`, `overlong_buffer_len` and
`overlong_penalty_scale` apply a linear length penalty before group normalization.
See [CLI options](params.md#strategy-specific) for defaults.

### Online Rollout

`online_grpo` and `online_dpo` use the respective GRPO and DPO strategies with
a `RolloutRunner`. The runner renders prompts through the tokenizer chat
template, generates grouped responses through the inference backend, then scores
them with a `BaseRewardModel`. It refreshes cached rollouts every
`rollout_interval` optimizer steps. `online_grpo` carries the sampler's aligned
behaviour log-probabilities into the loss, so it does not allocate or synchronize
a separate old-policy model.

`online_ppo` is actor-critic PPO on the same rollout pipeline. A `ValueModel`
critic (backbone warm-started from the policy, zero-initialized value head)
scores the rollout states; advantages come from GAE(`--ppo_gamma`,
`--ppo_gae_lambda`) with the terminal reward on each response's last token and
the reference-KL penalty (k3 estimator, `--grpo_kl_coef`) folded into per-token
rewards. Advantages and returns are computed once per rollout and pinned on the
`RolloutResult`, so replayed steps optimize fixed targets. The critic has its
own optimizer, stepped outside the policy-version lock, and persists as
`value_model.pt`/`value_optimizer.pt` checkpoint extras — resume without them
fails loudly, and `scripts/docker/lib/train-common.sh` (via
`CHECKPOINT_EXTRA_FILES` in `scripts/train.sh`) treats a PPO checkpoint as
incomplete when they are missing.

Every successful optimizer step mutates the shared model and advances its
monotonic `policy_version` under the same generation lock. The scheduler
invalidates reusable KV prefixes before accepting the new version, so an async
rollout cannot observe partially updated weights under the previous version.
`RawRollout` and `RolloutResult` retain the version that actually generated
their behavior log-probabilities, so cached rollout samples remain attributable
even while later optimizer steps advance the live policy. Results from a future
version or beyond `rollout_max_policy_lag` are rejected before training. The
final version check and rollout-cache publication share that policy lock, so a
concurrent update cannot land between validation and cache insertion.

Online strategies require `TrainConfig.reward_model_fn`. The CLI accepts
`--reward_model module:qualified_name`, or `training.reward_model` in YAML,
pointing to a callable reward factory.

### Rollout backends and validation sampling

For one-learner online GRPO, set `TrainConfig(rollout_mode="async_round",
rollout_devices=worker_devices, rollout_interval=1,
rollout_max_policy_lag=1)`, where `worker_devices` contains distinct CUDA
devices separate from the learner device. The learner holds the trainable
policy, optimizer and frozen KL reference; each listed device hosts one frozen
generation process. A round distributes its prompts across the listed workers;
each worker handles up to `ceil(batch_per_device / len(rollout_devices))`
prompts. The learner accumulates prompt microbatches
(`async_train_microbatch_prompts=1`) and steps the optimizer once per round.
The next round generates while the current round trains, with at most one
optimizer-version lag; a stale or failed round is never partially trained.

Async mode supports one CUDA learner with dense models and `online_grpo`;
validation is not supported. It is available through both Python configuration
and the CLI example below. See [async rounds](../developer/trainer/async_round.md)
for worker startup, NCCL weight transfer, version checks and failure deadlines.
Synchronous rollout remains the default.

Where generation physically runs is a *backend* choice
(`astrai/trainer/backend.py`): by default the scheduler wraps the training
model object in-process (`ColocatedBackend`, weight updates are free).
Setting `--rollout_device` builds a frozen replica on the selected device
with its own scheduler and KV pool; a `P2PCopyPublisher` copies the
training weights into the replica inside the policy-version lock on every
optimizer step, so the replica's generations stay version-attributable. The
copy transfers the full model state at every optimizer step.

The scheduler's KV pool is sized from the model's full
`max_position_embeddings` by default; `--rollout_pool_seq_len` right-sizes it
to the true rollout horizon (it must cover the longest prompt plus
`rollout_max_tokens`). Requests beyond the budget are
rejected (`prompt_too_long`) rather than silently truncated.

`--rollout_val_device` gives *validation* its own replica, so evaluation
generation never disturbs the training scheduler's KV pool. Validation
sampling is decoupled from training via the `rollout_val_*` overrides
(`--rollout_val_temperature 0` decodes greedily; group size, top-p, top-k,
and max tokens inherit the training rollout when unset). Online validation
runs through a `RolloutEvaluator` that reports reward statistics
(`reward_mean`/`reward_std`/`reward_max`/`response_len_mean`) in the
`validation` metric events instead of the training RL loss, which is
degenerate under greedy decoding or `group_size == 1`.

## Precision

Use the top-level YAML `precision` section or the CLI dtype flags, rather than
`model.dtype`. CLI defaults use BF16 computation and FP32 gradients, master
weights and optimizer state; export follows the resolved compute dtype unless
overridden. Supported parallel layouts and CPU state offload are documented in
[training configuration](../developer/trainer/configuration.md).

## LR Schedulers

| Type | Class | Description |
|------|-------|-------------|
| Cosine | `CosineScheduler` | Linear warmup → cosine decay to `min_rate` |
| SGDR | `SGDRScheduler` | Cosine annealing with warm restarts (`t_mult=2`) |
| WSD | `WSDScheduler` | Warmup-Stable-Decay with quadratic decay |

Created by `SchedulerFactory.create(schedule_type, optimizer, **kwargs)`. Valid types: `"cosine"`, `"sgdr"`, `"wsd"`. The training CLI always creates a scheduler and defaults `--schedule_type` to `"cosine"`.

## Gradient Checkpointing

Trades compute for memory by recomputing activations during backward pass. Specify module types via `gradient_checkpointing_modules`:

```python
from astrai.model.components.mlp import MLP

config = TrainConfig(..., gradient_checkpointing_modules=[MLP])
```

The training CLI's `--gradient_checkpointing` selects MLP modules, leaving
attention forwards unwrapped. The callback wraps each selected module's
`forward` with `torch.utils.checkpoint.checkpoint(use_reentrant=False)`,
compatible with `torch.compile`. It uses `nn.Module.apply()` for traversal and
works through DDP wrappers without manual unwrap. An empty list (default) is a
no-op.

## Checkpoint

```
Checkpoint(state_dict, epoch, consumed_samples, extra, meta, config)
  ├── save(save_dir)    atomically publishes manifest.json + metadata + weights + optional {key}.pt
  └── load(save_dir, broadcast=False, verify_checksums=False)    loads locally or broadcasts from rank-0
```

`Checkpoint.save()` writes to a hidden sibling staging directory, records the file
sizes in `manifest.json`, flushes the files, and atomically renames the completed
directory into place. That rename is the completeness guarantee: a directory is
never visible half-written, which is why v2 no longer hashes its files (hashing an
8.5 GB checkpoint cost ~45 s per save). Publication is write-once per step — saving
to an already-published step is a no-op and returns `False`, so the interval save
and the shutdown save can both run without re-writing anything. Legacy v1
directories still load, and `Checkpoint.verify(dir)` re-checks their checksums on
demand.

During training, `CheckpointCallback` uses the executor checkpoint context so
only rank 0 receives a state dict and calls `save()`.

Optimizer/scheduler state persisted by default via `Checkpoint.extra`.

Use `TrainConfig.load_state_dict` to select which component states are loaded.
The CLI YAML uses the same mapping under `training`:

```yaml
training:
  load_state_dict:
    model: true
    optimizer: true
    scheduler: false
```

These are the effective defaults for a new stage. Omitted keys use their
defaults; unknown component names are rejected. `model: false` skips reading
model weights and initializes a new model from its configuration.
`optimizer: false` skips the optimizer extra. `scheduler: false` starts the
configured schedule from its beginning. Weight-only checkpoints initialize
fresh optimizer/scheduler state.

When loading optimizer history for a new stage, current parameter-group
settings such as LR and weight decay are retained. Loading the scheduler
also restores its saved learning rate. Changing optimizer type or parameter
layout requires `optimizer: false`.

`--resume` restores the data cursor and other training extras. With no explicit
component flags, it loads all three components, including saved optimizer
settings. Explicit `load_state_dict` entries override these defaults even on
resume. To start a new stage on new data, omit `--resume`.

### Resume metadata

Training resumes require `resume_contract_version: 1` in `meta.json`. There is
no fallback to the old per-rank training loop. A weight-only warm start remains
available without `--resume`.

The contract records `optimizer_step`, `dataset_len`, `epoch_sample_offset`,
`random_seed`, the strategy, DP/CP/TP sizes, DP mode, batch and accumulation
sizes, tail policy, rollout mode, and replay settings. Synchronous offsets count
samples in the unpadded epoch permutation and identify a completed global
window (or the exact epoch end). Async offsets count committed samples across
equal padded rank shards; async resume requires the same learner layout.

To migrate an older checkpoint, stop its readers and writers, back up
`meta.json` and `manifest.json`, and supply the missing contract fields from
the original run configuration. Derive the epoch offset from committed sample
accounting, not a prefetched sampler position. For an unpadded first epoch,
the offset equals `consumed_samples`; it must be divisible by the original
global effective batch, unless it is the epoch end. Update the manifest's
metadata size (and its checksum if present). Do not change weights, optimizer
moments, scheduler counters, or metric history. If the committed position
cannot be established, retain the original checkpoint rather than rounding
its cursor. The checkpoint file format version and resume contract version
are independent.

### A checkpoint is one coherent state

Everything a checkpoint records describes the same point in training:

- the data cursor advances on optimizer commits and when an incomplete
  epoch-tail window is discarded; `--resume` starts after that cursor, while
  a window interrupted before a commit is read again;
- `meta.json` reports the optimizer step those weights belong to; the cursor
  also accounts for any discarded epoch-tail samples;
- the epoch data order and scheduler position are described by the saved state.

`Trainer` seeds every generator from `random_seed` before building the context, so
a fresh run starts reproducibly. The current checkpoint extras do not restore
RNG state, so stochastic training is not guaranteed to replay bit for bit. The DataLoader gets its own epoch-seeded generator, because PyTorch draws the
worker base seed from the global generator on every iterator creation — including
the single-process one — and a resumed run creates one extra iterator.

Signals are taken after an optimizer commit: a request that arrives inside an
accumulation window finishes that window first (at most `grad_accum_steps - 1` more
batches). At epoch end, a shorter window follows `epoch_tail_policy`.
The checkpoint records the committed or discarded data cursor. If an optimizer
or scheduler update fails after mutating state, `checkpoint_safe` remains false:
shutdown leaves the previous published checkpoint intact rather than labeling
the partial update with the previous step number.

Stopping is cooperative across ranks: a rank that receives a signal forwards it to
the launcher, the launcher signals every rank, and each step ends with a one-element
all-reduce of the stop flag, so all ranks leave on the same step. That matters because
the rank that saves still needs the others alive for the collectives that materialise
the weights. A rank that dies outright (SIGKILL, OOM kill) cannot be checkpointed at
all: the survivors fail their next collective — gloo reports the closed connection at
once, NCCL waits for the process-group timeout (10 minutes by default) — and the run
resumes from the last interval checkpoint. Rank processes are tied to the launcher
with `PR_SET_PDEATHSIG`, so killing the launcher leaves no GPU workers running
unsupervised.

`process_group_timeout` bounds how long a stalled collective may block; leave it
unset to keep the backend default (10 minutes for NCCL). After a normal stop signal,
the local launcher waits for every rank to finish its current update and checkpoint,
even if that takes longer than the 30-second failure-cleanup grace period. A rank
blocked outside a collective may still require an external forced stop.

The DataLoader is dropped before anything is written. Its workers usually take the
same signal that stopped training, and PyTorch turns a dead worker into a
`RuntimeError` from its SIGCHLD handler — one that can land in the middle of the
save and cost the checkpoint.

The shutdown is an ordered plan owned by `TrainSession`: run the registered
cleanup hooks (the DataLoader workers), call `on_shutdown` once on every observer,
close the rollout, run `on_train_end`, then the final barrier. Every step is
attempted even when an earlier one raised, so a failed save cannot also skip the
rollout shutdown.

Each published step directory contains `metric.jsonl` with the committed
metric history through that checkpoint. The metric callback buffers records until
the optimizer update succeeds; the checkpoint writer copies the previous history
and new records into its staging directory before publishing the manifest. Resume
continues from the selected checkpoint's log, so a newer, abandoned branch of
metrics is never rewritten or mistaken for the resumed run.

The equivalent CLI override takes JSON:
`--load_state_dict '{"optimizer": false}'`.

Model config (`context.model_config`) saved into `config.json` during training via `CheckpointCallback`.

## Model loading

Construction and restore order are described in
[trainer lifecycle](../developer/trainer/lifecycle.md).

Pretrained policy loading uses the same model-owned `hf_mapping.json` weight
conversion as `AutoModel.from_pretrained`, including norm offsets and declared
vision/MTP exclusions. Missing required tensors or unsupported HF weight keys
fail before the optimizer is created. An untied output head must be present in
the checkpoint; a tied embedding/head uses the model's canonical embedding key.
New LoRA adapter parameters may be initialized when loading a base policy, while
resume requires the saved adapters as well as the backbone. Critic head
initialization remains separate from policy loading.

The generic model-loading API defaults to strict loading:
`AutoModel.from_pretrained(path, strict=True)`. Deliberate partial loading uses
`strict=False`; missing parameters keep their initialized values, unexpected
keys are ignored, and shape mismatches still fail. To train that deliberately
initialized model, save it as a complete model directory first. Training and
full resume always require complete policy weights, except newly created LoRA
adapter parameters during a warm start.

`load_model_weights(directory)` and `save_model_weights(state_dict, directory)`
read and write weight dictionaries only. They do not construct a model, convert
HF layouts, or decide resume semantics. Checkpoint metadata records the available
source revision, mapping SHA-256, explicit exclusions and adapter initialization.
An unavailable source revision is recorded as `null`.

Tokenizer and generation file ownership, export behavior and stop settings are
defined in [HF file conventions](../developer/huggingface.md#tokenizer-and-generation-files).
Native AstrAI weights and optimizer state do not become a Transformers model
merely by using those file names.

For factory-initialized native training, `param_path` may supply only the native
config/tokenizer directory. A directory declaring an HF mapping must contain
policy weights, and resume always requires checkpoint weights.

## Training CLI

```bash
python scripts/tools/train.py --config train.yaml --dry-run
python scripts/tools/train.py --config train.yaml
```

The dry run builds and validates the same TrainConfig as execution, including
component parameters, precision, strategy options, and distributed settings. It
prints the resolved dataset window, optimizer/scheduler parameters, effective
batch size, and component loading policy. It reads model configuration and may
import a configured reward factory, but does not construct a trainer, load model
weights or datasets, or start distributed workers. Checkpoint/data compatibility
is still validated when training loads those resources.


Explicit CLI flags override YAML. The dry run validates configuration without
starting training. See [Quick Start](../quick-start.md#5-train) for a direct
single-GPU command and [distributed training](distributed.md) for launch settings.

### Switching training components

A YAML config can select an importable model, optimizer and scheduler without
changing Trainer. The model config and optional starting weights still come
from `training.param_path`; construction happens separately in every rank.
Registered AstrAI model, optimizer and scheduler names can be used instead
of import paths.

```yaml
model:
  type: my_package.models:MyModel
  params: {width: 2048}
precision:
  compute_dtype: bfloat16
dataset:
  type: packed
  params:
    path: /path/to/dataset
    train_type: seq
    window_size: 2048
    stride: 2048
optimizer:
  type: my_package.optim:make_optimizer
  params: {lr: 0.0002, weight_decay: 0.1}
scheduler:
  type: my_package.schedules:MySchedule
  params: {warmup_steps: "$warmup_steps", total_steps: "$total_steps"}
training:
  train_type: seq
  param_path: /path/to/model
```

All four entries use `ComponentSpec(type, params)`. The dataset entry opens
inside each training rank and stores the resulting dataset and DataLoader in
`TrainContext`. The optimizer target receives the prepared model (or its
parameters when its first argument is named `params` or `parameters`). The scheduler target
receives the optimizer. In scheduler parameters, `warmup_ratio` is a
reserved setting used to compute `$warmup_steps`; the other available
placeholders are `$decay_steps` and `$dataset_len`. The usual flat CLI
options remain supported and normalize to these component specs.


### Configuring asynchronous rollout from the CLI

Use a reward factory import path (`module:qualified_name`) returning your reward
model. A YAML example for one learner and two rollout workers:

```yaml
training:
  train_type: online_grpo
  param_path: /path/to/policy
  data_root_path: /path/to/dataset
  dp_mode: none
  rollout_mode: async_round
  rollout_devices: [cuda:1, cuda:2]
  rollout_interval: 1
  rollout_max_policy_lag: 1
  group_size: 4
  reward_model: my_rewards:make_reward
  rollout_startup_timeout_s: 300
  rollout_worker_timeout_s: 600
  rollout_weight_timeout_s: 600
  async_train_microbatch_prompts: 1
```

```bash
CUDA_VISIBLE_DEVICES=2,4,5 .venv/bin/python scripts/tools/train.py --config train.yaml
```

Here the learner uses logical `cuda:0` (physical GPU 2), and workers use logical
`cuda:1`/`cuda:2` (physical GPUs 4/5). Repeat `--rollout_devices cuda:N` for each
worker when overriding YAML; explicit flags replace the YAML list. Unknown keys
in supported YAML sections raise an error. The CLI maps its `group_size` option
to the algorithm's `strategy_kwargs`; Python callers use
`strategy_kwargs={"group_size": 4}`. Python model factories must be pickleable and
should construct models on CPU; worker placement is resolved afterward. Guard
`Trainer.train()` with `if __name__ == "__main__":` when using a custom script.


## Component construction

Use `Factory.create(name, ...)` for registered components. It validates
constructor arguments strictly instead of discarding unknown arguments.
`create_from_config()` remains the explicit adapter for selecting constructor
parameters from a broad model configuration; explicit overrides are validated.

Built-in top-level models accept a validated `config` object. Extra
`ComponentSpec.params` must match the chosen constructor. Decoder blocks use
`create_from_config()` to select the attention and feed-forward parameters
from the full model configuration. Custom constructors declaring `**kwargs`
remain responsible for validating those extra options.

Built-in callbacks are constructed directly.
Pass custom instances with `Trainer(config, callbacks=[MyCallback()])`.
See [callback contracts](../developer/trainer/callbacks.md) for lifecycle hooks.
