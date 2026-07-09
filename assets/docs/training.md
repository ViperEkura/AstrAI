# Training

## Contents

- [Autoregression](#autoregression)
- [Causal Mask](#causal-mask)
- [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
- [Training Loop](#training-loop)
- [Strategies](#strategies) — SEQ, SFT, DPO, GRPO
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

The complex rotation `freqs_cis` is pre-computed once (`cos, sin` pairs per position). `apply_rotary_emb` multiplies Q/K as complex numbers.

## Training Loop

Two-level loop: **epoch** → **batch**. Optimizer step fires every `grad_accum_steps` batches.

```
on_train_begin
  model.train()
  on_epoch_begin
    for batch in dataloader:
      on_batch_begin
      with executor.accumulate(model):
        loss = strategy.compute_loss(batch)
        context.loss = loss.item()
        stand_loss = loss / executor.grad_accum_steps
        executor.backward(stand_loss)
        context.consumed_samples += (
            context.config.batch_per_device * context.world_size
        )
        on_batch_end

        if executor.sync_gradients:
          on_optimizer_step
          optimizer.step()
          optimizer.zero_grad()
          if scheduler:
            scheduler.step()
    on_epoch_end
on_train_end
```

### Callback Lifecycle

| Hook | Fires | Default callback |
|------|-------|-----------------|
| `on_train_begin` | Before training starts | `GradientCheckpointingCallback` |
| `on_epoch_begin` | Start of each epoch | `ProgressBarCallback` |
| `on_batch_begin` | Every batch | — |
| `on_optimizer_step` | Every accumulation window | `GradientClippingCallback`, `MetricCallback`, `ProgressBarCallback` |
| `on_batch_end` | Every batch | `CheckpointCallback` |
| `on_epoch_end` | End of each epoch | `MetricCallback`, `ProgressBarCallback` |
| `on_error` | On exception during training | `CheckpointCallback`, `MetricCallback` |
| `on_train_end` | Training ends (always via finally) | `CheckpointCallback`, `MetricCallback`, `GradientCheckpointingCallback` |

Default callbacks (in order): `gradient_checkpointing` (activation checkpointing, optional), `checkpoint` (safetensors, rank-0), `metric` (JSONL + validation, rank-0), `progress_bar` (tqdm), `gradient_clipping`.

## Strategies

### SEQ (Pre-training)

Next-token cross-entropy with optional label smoothing:

$$
L_{\text{PT}} = -\sum_{t=1}^{T} \log P(x_t \mid x_{\lt t}; \theta)
$$

Keys: `input_ids`, `target_ids`. Optional: `label_smoothing`.

### SFT (Supervised Fine-Tuning)

Masked cross-entropy (`ignore_index=-100`) over response tokens:

$$
L_{\text{SFT}} = -\sum_{t=P+1}^{P+L} \log P(s_t \mid s_{\lt t}; \theta)
$$

Keys: `input_ids`, `target_ids`, `loss_mask`, `position_ids`. Optional: `label_smoothing`.

### DPO (Direct Preference Optimization)

Frozen reference model, preference margin via log-ratio:

$$
L_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

Parameters: `beta=0.1`, `reduction="mean"`. Keys: `chosen`, `rejected`, `chosen_mask`, `rejected_mask`.

### GRPO (Group Relative Policy Optimization)

On-policy PPO with group-normalized advantages:

$$
\text{Advantage}_i = \frac{r_i - \mu}{\sigma + \epsilon}
$$

$$
L_{\text{GRPO}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{ref}}}A,\; \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{ref}}}, 1-\epsilon, 1+\epsilon\right)A\right)\right] + \lambda \cdot \mathbb{E}\left[(\log\pi_\theta - \log\pi_{\text{ref}})^2\right]
$$

Parameters: `group_size=4`, `clip_eps=0.2`, `kl_coef=0.01`, `sync_interval=200`, `reduction="mean"`.

Keys: `prompts`, `responses`, `masks`, `rewards`.

## LR Schedulers

| Type | Class | Description |
|------|-------|-------------|
| Cosine | `CosineScheduler` | Linear warmup → cosine decay to `min_rate` |
| SGDR | `SGDRScheduler` | Cosine annealing with warm restarts (`t_mult=2`) |
| WSD | `WSDScheduler` | Warmup-Stable-Decay with sqrt cooldown |

Created by `SchedulerFactory.create(schedule_type, optimizer, **kwargs)`. Valid types: `"cosine"`, `"sgdr"`, `"wsd"`. Omit to use no scheduler.

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
  ├── save(save_dir)    rank-0 only: meta.json (epoch/consumed_samples/timestamp) + config.json (model config) + model.safetensors + optional {key}.pt (optimizer.pt, scheduler.pt)
  └── load(save_dir, broadcast=False)    loads from local disk; set broadcast=True to broadcast metadata from rank-0
```

Optimizer/scheduler state persisted by default via `Checkpoint.extra`.  
Model config (`context.model_config`) saved into `config.json` during training via `CheckpointCallback`.

## TrainContextBuilder (Builder Pattern)

```python
context = (
    TrainContextBuilder(config)
        .with_resume_dir(resume_dir)
        .build()
)
# Returns TrainContext with model, strategy, optimizer, scheduler, dataloader, checkpoint
```

- Loads checkpoint weights if provided
- Creates executor via `ExecutorFactory.create(cfg.parallel_mode, grad_accum_steps=cfg.grad_accum_steps, **cfg.executor_kwargs)`
- Calls `executor.prepare(model, optimizer, dataloader, scheduler)` for model distribution (e.g. DDP) + gradient accumulation wrappers
- Creates `ResumableDistributedSampler` for shuffle+resume
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

> Document Update Time: 2026-07-09
