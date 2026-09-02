# Training

## Contents

- [Autoregression](#autoregression)
- [Causal Mask](#causal-mask)
- [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
- [Training Loop](#training-loop)
- [Strategies](#strategies) — SEQ, SFT, DPO, GRPO, online rollout
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

Two-level loop: **epoch** → **batch**. Optimizer step fires every `grad_accum_steps` batches.

```
on_train_begin
  model.train()
  on_epoch_begin
    for batch in dataloader:
      with executor.accumulate(model):
        on_batch_begin
        loss_output = strategy(batch)
        context.loss = loss_output["loss"].item()
        context.metrics = loss_output["metrics"]
        stand_loss = loss_output["loss"] / executor.grad_accum_steps
        executor.backward(stand_loss)
        context.consumed_samples += (
            context.config.batch_per_device * context.world_size
        )
        on_batch_end

        if executor.sync_gradients:
          before_optimizer_step
          optimizer.step()
          strategy.on_optimizer_step()
          optimizer.zero_grad()
          if scheduler:
            scheduler.step()
          after_optimizer_step
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
| `on_error` | On exception during training | `CheckpointCallback`, `MetricCallback` |
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
(`old_model`, synced externally between data-generation rounds) and the
expectations are over valid response tokens. The KL term regularises
$\pi_\theta$ towards a frozen reference model (`ref_model`, typically
the SFT checkpoint).

Parameters: `group_size=4`, `clip_eps=0.2`, `kl_coef=0.01`. External sync of `old_model` weights via `sync_old_model()` between data-generation rounds.

Keys: `prompts`, `responses`, `masks`, `rewards`.

Set `loss_variant="dr_grpo"` to use the two bias-removal changes from
[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783):

$$
A_i^{\text{Dr.GRPO}} = r_i - \mu, \qquad
L_{\text{policy}} = \frac{1}{BG L_{\max}}
\sum_{i=1}^{B}\sum_{j=1}^{G}\sum_{t=1}^{L_{\max}}
m_{ijt}\,\ell_{ijt}.
$$

Unlike standard GRPO, this variant does not divide centered rewards by their
group standard deviation and does not divide policy loss by the number of valid
tokens. `max_completion_length` is therefore required as a fixed generation
budget; online training uses `rollout_max_tokens` when it is not set explicitly.
The KL regularizer remains independently configurable. Set `kl_coef=0` to match
the paper's no-KL recipe.

### Online Rollout

`online_grpo` and `online_dpo` use the respective GRPO and DPO strategies with
a `RolloutRunner`. The runner renders prompts through the tokenizer chat
template, generates grouped responses through `InferenceScheduler`, then scores
them with a `BaseRewardModel`. It refreshes cached rollouts every
`rollout_interval` optimizer steps. `online_grpo` synchronizes `old_model` when
a fresh rollout is produced.

Online strategies require `TrainConfig.reward_model_fn`. `train.py` exposes the
rollout sampling parameters but does not yet offer a CLI argument for the reward
model factory.

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
from astrai.model.components.decoder_block import DecoderBlock

config = TrainConfig(..., gradient_checkpointing_modules=[DecoderBlock])
```

Callback wraps each `DecoderBlock.forward` with `torch.utils.checkpoint.checkpoint(use_reentrant=False)`, compatible with `torch.compile`. Uses `nn.Module.apply()` for traversal — works through DDP wrappers without manual unwrap. Empty list (default) means no-op.

## Checkpoint

```
Checkpoint(state_dict, epoch, consumed_samples, extra, meta, config)
  ├── save(save_dir)    meta.json (epoch/consumed_samples/timestamp) + config.json (model config) + model.safetensors + optional {key}.pt (optimizer.pt, scheduler.pt)
  └── load(save_dir, broadcast=False)    loads from local disk; set broadcast=True to broadcast metadata from rank-0
```

`Checkpoint.save()` writes whenever it is called. During training, `CheckpointCallback` uses the executor checkpoint context so only rank 0 receives a state dict and calls `save()`.

Optimizer/scheduler state persisted by default via `Checkpoint.extra`.  
Model config (`context.model_config`) saved into `config.json` during training via `CheckpointCallback`.

## TrainContextBuilder (Builder Pattern)

```python
context = TrainContextBuilder(config).with_param_path(param_path, resume=True).build()
# Returns TrainContext with model, strategy, optimizer, scheduler, dataloader, checkpoint
```

- Loads checkpoint weights before the model is wrapped
- Creates executor via `ExecutorFactory.create(cfg.parallel_mode, grad_accum_steps=cfg.grad_accum_steps, **cfg.executor_kwargs)`
- Calls `executor.prepare(model_fn, optimizer_fn, scheduler_fn, before_wrap=...)`; the executor creates, wraps, then builds the optimizer and scheduler for the wrapped model
- Creates `RDSampler` for shuffle+resume
- Builds strategy via `StrategyFactory.create(train_type, model, device, **kwargs)`

## Training CLI

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python scripts/tools/train.py \
    --nprocs=4 \
    --parallel_mode=ddp \
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

> Document Update Time: 2026-08-02
