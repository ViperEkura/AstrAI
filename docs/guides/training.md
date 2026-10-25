# Training

## Contents

- [Autoregression](#autoregression)
- [Causal Mask](#causal-mask)
- [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
- [Training Loop](#training-loop)
- [Strategies](#strategies) — SEQ, SFT, DPO, GRPO, online rollout
- [Cross-entropy backends](#cross-entropy-backends) — SEQ/SFT loss selection
- [LR Schedulers](#lr-schedulers)
- [Gradient Checkpointing](#gradient-checkpointing)
- [Checkpoint](#checkpoint)
- [TrainContextBuilder](#traincontextbuilder-builder-pattern)
- [Training CLI](#training-cli)

### Autoregression

Given a token sequence, the model predicts the probability of the next token. Each generated token is appended to the input and fed back, repeating until an end-of-sequence token or max length.

### Causal Mask

```
sequence : [[1, 2, 3, 4, 5, 6]]
input_ids: [[1, 2, 3, 4, 5]]
target_ids: [[2, 3, 4, 5, 6]]
```

Lower-triangular mask prevents attending to future positions:

```
[[0, -inf, -inf, -inf, -inf],
 [0,    0, -inf, -inf, -inf],
 [0,    0,    0, -inf, -inf],
 [0,    0,    0,    0, -inf],
 [0,    0,    0,    0,    0]]
```

### Rotary Position Embedding (RoPE)

RoPE embeds position into Q/K vectors via complex rotation:

$$ q_i = R_i W_q x_i, \quad k_j = R_j W_k x_j, \quad q_i^T k_j = x_i^T W_q^T R_{i-j} W_k x_j $$

`RotaryEmbedding` pre-computes a cos/sin table `freqs_cis` of shape
`[max_len, dim/2, 2]` (f32 — `[cos, sin]` pairs). `forward()` returns
a `[batch, seq_len, dim/2, 2]` slice indexed by `position_ids`.
`apply_rotary_emb` applies the rotation: during training it uses torch
complex multiply (autograd-compatible); during inference it auto-dispatches
to a fused CUDA kernel when available.

## Training Loop

Two-level loop: **epoch** → **batch**. The outline below shows offline training:
an optimizer step fires every `grad_accum_steps` batches, or on the last batch
of an epoch if fewer remain. Online strategies may publish several updates per batch.

```
on_train_begin
  model.train()
  on_epoch_begin
    for batch in dataloader:
      with executor.accumulate(model, force_sync=is_epoch_tail):
        context.pending_samples += samples_in_batch_across_dp_ranks
        on_batch_begin
        loss_output = strategy(batch)
        context.loss = loss_output["loss"].item()
        context.metrics = loss_output["metrics"]
        stand_loss = loss_output["loss"] / executor.grad_accum_steps
        executor.backward(stand_loss)
        if executor.sync_gradients:
          if executor.window_steps < executor.grad_accum_steps:
            scale_gradients(grad_accum_steps / window_steps)
          context.optimizer_steps_completed += 1
          context.consumed_samples += context.pending_samples
          before_optimizer_step
          strategy.optimizer_step(optimizer)
          optimizer.zero_grad()
          if scheduler:
            scheduler.step()
          context.pending_samples = 0
          after_optimizer_step
        on_batch_end
    on_epoch_end
on_train_end
```

### Callback Lifecycle

| Hook | Fires | Default callback |
|------|-------|-----------------|
| `on_train_begin` | Before training starts | `GradientCheckpointingCallback`, `CheckpointCallback`, `MetricCallback` |
| `on_epoch_begin` | Start of each epoch | `ProgressBarCallback` |
| `on_batch_begin` | Every batch | — |
| `before_optimizer_step` | Every accumulation window, before `optimizer.step()` | `MetricCallback`, `ProgressBarCallback`, `GradientClippingCallback` |
| `on_batch_end` | Every batch | — |
| `after_optimizer_step` | Every accumulation window, after `optimizer.step()` and `scheduler.step()` | `CheckpointCallback` |
| `on_epoch_end` | End of each epoch | `MetricCallback`, `ProgressBarCallback` |
| `on_shutdown` | The run is ending: a graceful stop or a failure | `CheckpointCallback`, `MetricCallback` |
| `on_train_end` | Training exits after `on_train_begin` completes (via `finally`) | `GradientCheckpointingCallback`, `CheckpointCallback`, `MetricCallback` |

Default callbacks (in order): `gradient_checkpointing` (activation checkpointing, optional), `checkpoint` (safetensors, rank-0), `metric` (JSONL + validation, rank-0), `progress_bar` (tqdm, rank-0), `gradient_clipping`. The gradient-clipping callback is always registered and always calls `executor.clip_grad_norm()` with the numeric `max_grad_norm` value.

Strategies return `{"loss": Tensor, "metrics": Dict[str, float]}` when called by the trainer. Built-in metrics include the task-specific loss and, for MoE models, `moe_aux_loss` plus `moe_aux_loss_weighted`. Direct `compute_loss(batch)` calls continue to return a single loss tensor.

## Strategies

### SEQ (Pre-training)

Next-token cross-entropy with optional label smoothing:

$$
L_{\text{PT}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{\lt t}; \theta)
$$

Keys: `input_ids`, `target_ids`. Optional: `label_smoothing`.

### SFT (Supervised Fine-Tuning)

Masked cross-entropy (`ignore_index=-100`) over response tokens:

$$
L_{\text{SFT}} = -\frac{1}{L}\sum_{t=P+1}^{P+L} \log P(s_t \mid s_{\lt t}; \theta)
$$

Keys: `input_ids`, `target_ids`, `loss_mask`, `position_ids`. Optional: `label_smoothing`.

### Cross-entropy backends

SEQ and SFT use Torch cross-entropy by default. For a bias-free
`AutoRegressiveLM` head, configure the experimental fused linear CE backend
through `TrainConfig.strategy_kwargs` to avoid materializing full head logits:

```yaml
strategy_kwargs:
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

### DPO (Direct Preference Optimization)

Frozen reference model, preference margin via log-ratio:

$$
L_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

Parameters: `beta=0.1`, `reduction="sum"`. Keys: `chosen`, `rejected`, `chosen_mask`, `rejected_mask`.

### GRPO (Group Relative Policy Optimization)

Token-level PPO with group-normalized advantages. Advantages are derived from
scalar per-response rewards, group-normalized, and broadcast across all response
tokens. Only response tokens contribute to the loss (prompt tokens are masked
out):

$$
\text{Advantage}_i = \frac{r_i - \mu}{\sigma + \epsilon}
$$

$$
L_{\text{GRPO}} = -\mathbb{E}_t\left[\min\left(\rho_t A,\; \text{clip}\left(\rho_t, 1-\epsilon, 1+\epsilon\right)A\right)\right] + \lambda \cdot \mathbb{E}_t\left[\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1\right]
$$

where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$ is the
per-token importance sampling ratio against the behaviour policy
and the expectations are over valid response tokens. GRPO reduces over the
actual data-parallel group: each rank backpropagates its local response-loss
sum multiplied by the DP degree and divided by the global valid-token count.
Sequence aggregation uses the global count of nonempty responses instead.
Policy and KL terms share this denominator. A rank with no valid response
tokens contributes zero while joining the same collective schedule.

Use `rl_microbatch_prompts=1` to backpropagate one complete prompt group at
a time within an optimizer update. The denominator is frozen over the whole
update before its first forward. The trainer performs one optimizer step,
scheduler step and online version publication after the last microbatch.
`rl_minibatch_prompts` still creates independent optimizer updates, and
`rl_update_epochs` repeats those updates over the collected rollout; neither
is a substitute for microbatch accumulation. GRPO requires
`grad_accum_steps=1` for legacy cursors. The global-window trainer also
supports `grad_accum_steps>1`: all input chunks of the update share the same
frozen response denominator. Async rounds use `async_train_microbatch_prompts`
instead of `rl_microbatch_prompts`.

Unequal local group counts use zero-token scheduling padding, which never
enters the objective or consumed-prompt cursor. A globally empty synchronous update
performs no backward, weight decay, scheduler step or version publication.
GRPO checkpoints record actual optimizer updates separately from consumed
prompts, preserve short final batches, and save after a complete rollout
round. An error during a partial round retains the last complete checkpoint.
When empty data advances the cursor at an already published step, a new
`_samples_<cursor>` checkpoint preserves the earlier immutable snapshot.
Legacy resume requires the same balanced per-rank cursor; global-window
checkpoints retain the global cursor and upstream DP-resize resume contract.

The MoE auxiliary objective keeps its independent coefficient and existing
rank-local router-statistics convention. A nonzero auxiliary coefficient
with multiple microbatches or an empty-group rank is rejected: summing local
auxiliary means cannot reproduce the nonlinear whole-batch router objective.
Dense DDP gradient parity is covered by CPU/Gloo oracles; GPU, FSDP and
MoE auxiliary microbatch support need separate qualification. Reported loss,
policy and KL metrics sum across the update; other diagnostics describe the
last local microbatch.

Online GRPO reuses the
per-token `logprobs_old` captured by the rollout sampler, avoiding an
`old_model` copy and a repeated forward pass. Offline GRPO keeps `old_model` as
a compatibility fallback. The KL term regularises $\pi_\theta$ towards a frozen
reference model (`ref_model`, typically the SFT checkpoint).

Parameters: `group_size=4`, `clip_eps=0.2`, `kl_coef=0.01`. The optional
`clip_eps_low` and `clip_eps_high` parameters replace the symmetric interval
with $[1-\epsilon_{low}, 1+\epsilon_{high}]$. Leaving both unset preserves the
existing symmetric objective. DAPO Clip-Higher can be selected explicitly, for
example with `clip_eps_low=0.2` and `clip_eps_high=0.28`. Offline callers that
do not provide `logprobs_old` must sync `old_model` weights via
`sync_old_model()` between data-generation rounds.

`loss_aggregation="token"` (the default) divides by the total number of valid
response tokens, matching DAPO's token-level policy-gradient loss. Set it to
`"sequence"` to first average each response and then weight responses equally,
matching the original GRPO reduction for controlled ablations.

DAPO soft overlong shaping is enabled by setting `overlong_max_len` and a
positive `overlong_buffer_len`. If $L$ is the valid response length, the added
reward is zero through $L_{max}-L_{buffer}$, falls linearly to -1 at $L_{max}$,
and is multiplied by `overlong_penalty_scale`. It is disabled by default and
does not alter the reward-model output in place.

Keys: `prompts`, `responses`, `masks`, `rewards`, and optional
`logprobs_old` (required when `old_model` is not configured).

### Online Rollout

`online_grpo` and `online_dpo` use the respective GRPO and DPO strategies with
a `RolloutRunner`. The runner renders prompts through the tokenizer chat
template, generates grouped responses through `Scheduler`, then scores
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

Online strategies require `TrainConfig.reward_model_fn`. `train.py` exposes the
rollout sampling parameters but does not yet offer a CLI argument for the reward
model factory.

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

The process control plane uses one ordered duplex `multiprocessing.Pipe` per
worker. Messages have an explicit kind, request ID, round ID and policy version:
`MODEL_READY`, `READY`, `GENERATE`/`RESULT`, `WEIGHT_SYNC`/`WEIGHT_SYNC_ACK`,
`ERROR`, and `STOP`.
The coordinator accepts a reply only when its request ID, round and version
match the pending command. `RESULT` carries CPU rollout tensors, while
`WEIGHT_SYNC` carries only a version notification; policy tensors are broadcast
from the learner through an NCCL group containing the learner and all workers.
Worker process sentinels and finite deadlines are polled alongside the pipes,
so a crash or hang aborts the entire round. This is a local one-to-one
protocol; no network broker or cross-worker traffic is required.

Each worker first allocates its replica, joins the NCCL group, and receives the
initial learner weights before
constructing its inference backend and capturing CUDA Graphs. After each
optimizer step, the learner advances its policy version. At the next idle
round boundary it broadcasts the live weights once through NCCL; every worker
synchronizes its CUDA device before returning `WEIGHT_SYNC_ACK`. All workers
participate in each broadcast, even when a short final round has fewer prompts
than workers. The application does not allocate a shared-memory weight buffer;
NCCL selects its own transport. Token IDs, masks, aligned
behaviour log-probabilities and CPU rewards are assembled on the host before
the learner moves them to its device; KV cache and full logits remain on the
generation devices. The async mode is programmatic-only, supports
`online_grpo` with one CUDA learner and dense models, and does not yet support
validation. Each worker owns its Scheduler and CUDA Graph state in its own
process. The worker factory must be pickleable, and a Python script that calls
`Trainer.train()` must use an `if __name__ == "__main__":` guard. Worker startup
has a 300-second deadline; `rollout_worker_timeout_s` (default 600) bounds
generation and weight transfer. A failed worker aborts the whole round and
the coordinator terminates the remaining workers.
The existing synchronous backend remains the default.

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

### A checkpoint is one coherent state

Everything a checkpoint records describes the same point in training:

- the data cursor advances on optimizer commits and when an incomplete
  epoch-tail window is discarded; `--resume` starts after that cursor, while
  a window interrupted before a commit is read again;
- `meta.json` reports the optimizer step those weights belong to; the cursor
  also accounts for any discarded epoch-tail samples;
- the RNG states, the data order and the scheduler position are saved with them.

`Trainer` seeds every generator from `random_seed` before building the context, so
a fresh run is reproducible; on `--resume` the checkpoint's RNG states replace that
seed. The DataLoader gets its own epoch-seeded generator, because PyTorch draws the
worker base seed from the global generator on every iterator creation — including
the single-process one — and a resumed run creates one extra iterator.

Signals are taken after an optimizer commit: a request that arrives inside an
accumulation window finishes that window first (at most `grad_accum_steps - 1` more
batches). At epoch end, a shorter window is discarded without an optimizer
update. The checkpoint records the advanced data cursor and matching RNG state.

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

## TrainContextBuilder (Builder Pattern)

```python
context = TrainContextBuilder(config).with_param_path(param_path, resume=True).build()
# Returns TrainContext with model, strategy, optimizer, scheduler, dataloader, checkpoint
```

- Loads checkpoint weights before the model is wrapped
- Creates executor via `ExecutorFactory.create(cfg.dp_mode, grad_accum_steps=cfg.grad_accum_steps, **cfg.executor_kwargs)`
- Calls `executor.prepare(model_fn, optimizer_fn, scheduler_fn, before_wrap=...)`; the executor creates, wraps, then builds the optimizer and scheduler for the wrapped model
- Creates `RDSampler` for shuffle+resume
- Builds strategy via `StrategyFactory.create(train_type, model, device, **kwargs)`

## Training CLI

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python scripts/tools/train.py \
    --dp_size=4 \
    --dp_mode=ddp \
    --train_type=seq \
    --data_root_path=/path/to/dataset \
    --param_path=/path/to/model \
    --batch_per_device=4 \
    --grad_accum_steps=8 \
    --warmup_ratio=0.05 \
    --max_lr=1e-4 \
    --max_grad_norm=1.0 \
    --weight_decay=0.1 \
    --window_size=2048 \
    --ckpt_interval=10000 \
    --ckpt_dir=./checkpoint \
    --random_seed=3407 \
    --label_smoothing=0.05 \
    > out.log 2> err.log &
```

Full parameter reference at [params.md](params.md).

### Switching training components

A YAML config can select an importable model, optimizer and scheduler without
changing Trainer. The model config and optional starting weights still come
from `training.param_path`; construction happens separately in every rank.
Registered AstrAI model, optimizer and scheduler names can be used instead
of import paths.

```yaml
model:
  type: my_package.models:MyModel
  dtype: bfloat16
  params: {width: 2048}
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

> Document Update Time: 2026-09-20

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
