# Internals

Mathematical foundations and internal algorithms for AstrAI's training, inference, and preprocessing pipelines. For practical usage guides, see [Training](../guides/training.md), [Inference](../guides/inference.md), and [Preprocessing](../guides/preprocessing.md).

For training component ownership, startup and failure cleanup, see [Trainer internals](trainer/README.md).

## Autoregression & Causal Masking

Given a token sequence, the model predicts the probability of the next token. Each generated token is appended to the input and fed back, repeating until an end-of-sequence token or max length.

```
sequence : [[1, 2, 3, 4, 5, 6]]
input_ids: [[1, 2, 3, 4, 5]]
target_ids: [[2, 3, 4, 5, 6]]
```

A lower-triangular causal mask prevents attending to future positions:

```
[[0, -inf, -inf, -inf, -inf],
 [0,    0, -inf, -inf, -inf],
 [0,    0,    0, -inf, -inf],
 [0,    0,    0,    0, -inf],
 [0,    0,    0,    0,    0]]
```

This ensures position $i$ can only attend to positions $\leq i$, which is essential for autoregressive generation.

## Rotary Position Embedding (RoPE)

RoPE embeds position into Q/K vectors via complex rotation:

$$ q_i = R_i W_q x_i, \quad k_j = R_j W_k x_j, \quad q_i^T k_j = x_i^T W_q^T R_{j-i} W_k x_j $$

`RotaryEmbedding` pre-computes a cos/sin table `freqs_cis` of shape
`[max_len, dim/2, 2]` (f32 — `[cos, sin]` pairs). `forward()` returns
a `[batch, seq_len, dim/2, 2]` slice indexed by `position_ids`.
`apply_rotary_emb` applies the rotation: during training it uses torch
complex multiply (autograd-compatible); during inference it auto-dispatches
to a fused CUDA kernel when available. The key property is that the dot
product $q_i^T k_j$ depends only on the relative position $i - j$, not the
absolute positions.

**Critical for inference**: RoPE is applied **before** KV cache write, not after. If applied after caching, position encoding drift occurs because cached K/V would have stale rotation factors.

## Training Loss Formulas

### SEQ (Pre-training)

Next-token cross-entropy with optional label smoothing:

$$ L_{\text{PT}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{\lt t}; \theta) $$

### SFT (Supervised Fine-Tuning)

Masked cross-entropy (`ignore_index=-100`) over selected target tokens:

$$ L_{\text{SFT}} = -\frac{1}{L}\sum_{t=P+1}^{P+L} \log P(s_t \mid s_{\lt t}; \theta) $$

The sum is over positions selected by `loss_mask`. Preprocessing decides which
roles and template tokens receive supervision; SFT does not infer them from text.

### DPO (Direct Preference Optimization)

Frozen reference model, preference margin via log-ratio:

$$ L_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right] $$

Parameters: `beta=0.1`, `reduction="sum"`.

### GRPO (Group Relative Policy Optimization)

Token-level PPO with group-normalized advantages:

$$ \text{Advantage}_i = \frac{r_i - \mu}{\sigma + \epsilon} $$

$$ L_{\text{GRPO}} = -\mathbb{E}_t\left[\min\left(\rho_t A,\; \text{clip}\left(\rho_t, 1-\epsilon, 1+\epsilon\right)A\right)\right] + \lambda \cdot \mathbb{E}_t\left[\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1\right] $$

Where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$ is the per-token importance sampling ratio. Online rollout records $\log \pi_{\text{old}}$ when each token is sampled and reuses those values directly during training; offline batches may fall back to a synchronized `old_model`. Advantages are derived from scalar per-response rewards, group-normalized, and broadcast across all response tokens. Only response tokens contribute to the loss.

Parameters: `group_size=4`, `clip_eps=0.2`, `kl_coef=0.01`. Optional
`clip_eps_low`/`clip_eps_high` values enable DAPO-style asymmetric clipping;
unset values inherit `clip_eps` for backward-compatible symmetric clipping.
The `loss_aggregation` switch selects token-level DAPO weighting or equal
sequence weighting. Optional `overlong_max_len`/`overlong_buffer_len` settings
add the DAPO linear soft-overlong penalty before group advantage normalization.

### MoE Load Balancing

MoE layers add a differentiable load-balancing term based on mean router probabilities and top-k expert assignment frequency. The training objective is:

$$ L = L_{\text{task}} + \lambda_{\text{MoE}} L_{\text{aux}} $$

`TrainConfig.moe_aux_loss_coef` controls $\lambda_{\text{MoE}}$ (default `0.01`). The unweighted and weighted auxiliary losses are logged separately.

## Lifecycle, cache and preprocessing contracts

- [Trainer lifecycle](trainer/lifecycle.md) owns startup, updates and cleanup.
- [Callbacks](trainer/callbacks.md) owns committed metric history and publication.
- [Inference core](inference/core.md) owns KV allocation and prefix reuse;
  the [worker](inference/worker.md) executes immutable plans.
- [Preprocessing](../guides/preprocessing.md) defines masks and position IDs.
  Resetting position IDs at a document boundary does not isolate attention
  between packed documents.

## Gradient Accumulation Mechanics

Trainer plans global windows and explicitly selects the final synchronized
microbatch of each learner update with `accumulate_explicit(model, sync=...)`.
The executor's read-only `sync_gradients` property reports that selection;
`grad_accum_steps` is configuration, not a hidden microstep counter.

Non-final microbatches use the executor's `_no_sync` implementation:

- `NoneExecutor`: a no-op context.
- `DDPExecutor`: `model.no_sync()` suppresses gradient all-reduce.
- `FSDPExecutor`: disables gradient synchronization on each `FSDPModule` and
  optionally retains unsharded parameters until the synchronized backward.

The optimizer and scheduler returned by `prepare()` are real objects. Trainer
calls their update/clear operations only at the chosen boundary; direct callers
must do the same; calls outside that boundary are not silently suppressed.

After summing target-weighted losses, Trainer reduces the actual target counts
and rescales gradients to the global mean. `consumed_samples` advances only when
the full window is committed. CP and TP ranks do not multiply the sample count.

### Effective batch size

```
effective_batch = dp_size × batch_per_device × grad_accum_steps
```

### Total optimizer steps

```
global_batch = dp_size * batch_per_device * grad_accum_steps
full_windows, tail = divmod(remaining_samples_in_epoch, global_batch)
epoch_steps = full_windows + int(epoch_tail_policy == "flush" and tail > 0)
```

This is the offline count. Synchronous online windows contribute
`rl_update_epochs * ceil(window_samples / rl_minibatch_prompts)` updates when
minibatching is configured, or `rl_update_epochs` otherwise. `step_plan.py`
supplies the shared counts and update ranges; a resumed epoch counts only its
remaining samples. `drop` records and skips the incomplete window without an
optimizer update.
