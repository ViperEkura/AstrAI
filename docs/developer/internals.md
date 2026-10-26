# Internals

Mathematical foundations and internal algorithms for AstrAI's training, inference, and preprocessing pipelines. For practical usage guides, see [Training](../guides/training.md), [Inference](../guides/inference.md), and [Preprocessing](../guides/preprocessing.md).

For training component ownership, startup and failure cleanup, see [Trainer internals](trainer/README.md).

## Contents

- [Autoregression & Causal Masking](#autoregression--causal-masking)
- [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
- [Training Loss Formulas](#training-loss-formulas)
- [Training Loop Internals](#training-loop-internals)
- [Callback Lifecycle](#callback-lifecycle)
- [KV Cache Mathematics](#kv-cache-mathematics)
- [Mask Algorithm Internals](#mask-algorithm-internals)
- [Gradient Accumulation Mechanics](#gradient-accumulation-mechanics)

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

$$ q_i = R_i W_q x_i, \quad k_j = R_j W_k x_j, \quad q_i^T k_j = x_i^T W_q^T R_{i-j} W_k x_j $$

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

Masked cross-entropy (`ignore_index=-100`) over response tokens only:

$$ L_{\text{SFT}} = -\frac{1}{L}\sum_{t=P+1}^{P+L} \log P(s_t \mid s_{\lt t}; \theta) $$

Prompt tokens are masked out via `loss_mask`; only response tokens contribute to the loss.

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

## Training Loop Internals

Synchronous training uses **epoch → global data window → learner update →
microbatch**. Trainer chooses the update boundary; executors perform the
requested synchronization and return ordinary optimizers and schedulers.

```
on_train_begin
for epoch:
  on_epoch_begin
  for global window:
    prepare a rollout once when using an online strategy
    for learner update in learner_update_slices(...):
      for microbatch in this update:
        with executor.accumulate_explicit(model, sync=is_last_microbatch):
          on_batch_begin
          backward(loss * actual_target_count)
          on_batch_end
      reduce objective counts and normalize accumulated gradients
      before_optimizer_step
      strategy.optimizer_step(optimizer)
      optimizer.zero_grad()
      scheduler.step() if configured
      commit the window cursor only after its final learner update
      after_optimizer_step
  on_epoch_end
on_shutdown
on_train_end
```

The global-window objective uses actual valid token/sample counts, including
uneven `flush` tails and empty ranks. It does not divide every microbatch by a
fixed accumulation factor. Empty ranks participate in synchronization with
zero objective weight. Data progress advances at complete window boundaries;
each learner update has its own optimizer step.

Async rounds require `grad_accum_steps=1`. A strategy emits one update containing
its already-weighted microbatch losses; Trainer backpropagates them, then commits
once and publishes one policy version. It does not add another accumulation
counter or loss scaling factor.

Strategy metrics are detached and converted to Python `float` values before the
`LossOutput` is returned; only `LossOutput.loss` remains a differentiable tensor.

With `cp_size > 1`, `TrainContextBuilder` wraps the strategy in `CPStrategy`
(`astrai/parallel/cp.py`): the loss path shards the batch
across the cp group, runs the strategy's forward/reduction on the local slice,
and rescales the local token loss to the global mean. Metric and validation
reductions go through `ParallelTopology.reduce_mean`/`reduce_sum` over the dp
group — cp peers hold values derived from the same batch, so summing them
would double-count.

## Callback Lifecycle

| Hook | Fires | Default callback |
|------|-------|-----------------|
| `on_train_begin` | Before training starts | `GradientCheckpointingCallback`, `CheckpointCallback`, `MetricCallback` |
| `on_epoch_begin` | Start of each epoch | `ProgressBarCallback` |
| `on_batch_begin` | Every batch | — |
| `before_optimizer_step` | Every learner update, before `optimizer.step()` | `MetricCallback`, `ProgressBarCallback`, `GradientClippingCallback` |
| `on_batch_end` | Every batch | — |
| `after_optimizer_step` | Every learner update, after `optimizer.step()` and `scheduler.step()` | `MetricCallback`, `CheckpointCallback` |
| `on_epoch_end` | End of each epoch | `ProgressBarCallback` |
| `on_shutdown` | Session completion, requested stop or failure after context construction | `MetricCallback`, `CheckpointCallback` |
| `on_train_end` | Session teardown after context construction, including a failed start callback | `GradientCheckpointingCallback`, `CheckpointCallback`, `MetricCallback` |

Default callbacks (in order): `gradient_checkpointing` (activation checkpointing, optional), `metric` (JSONL + validation), `checkpoint` (safetensors, rank-0 writer), `progress_bar` (tqdm, rank-0), `gradient_clipping`. The gradient-clipping callback is always registered and always calls `executor.clip_grad_norm()` with the numeric `max_grad_norm` value.

## KV Cache Mathematics

At decode time, only the last query token matters. All previous K/V are cached to avoid recomputation:

$$ o_n = \sum_j \text{softmax}\left(\frac{q_n k_j}{\sqrt{d_k}}\right) v_j $$

The cache stores $k_j$ and $v_j$ for all previous positions. At each decode step, only $q_n$ (the current query) is computed fresh, and attention is computed against the cached K/V.

**RoPE ordering**: RoPE is applied to Q/K **before** writing to the KV cache. This is essential because:
1. The cached K values already contain the rotation for their original positions.
2. The new Q is rotated for its current position.
3. The dot product $q_n^T k_j$ then correctly depends on $n - j$ (relative position).

If RoPE were applied after caching, the rotation factors would be inconsistent between cached and new tokens.

### Cache Architecture

Three-layer separation (SGLang-inspired):

- **KVStorage**: Flat token-level buffers `[n_layers, size, n_kv_heads, head_dim]`.
- **ReqToTokenPool**: Index table `[req_idx, pos] → physical token slot`, shared across all layers.
- **Allocator + RadixCache**: Paged-mode allocation with ref-counting, LRU eviction, and exact page-aligned prefix sharing when `page_size > 1`.

`BlockPool` orchestrates all three (the per-request accounting facade is `KVCacheManager`); In contiguous mode (default), `req_to_token` is a trivial linear mapping. In paged mode, slots are allocated on demand. `RadixCache` walks exact token-page edges from the root, preserving parent-prefix context instead of treating a page hash as a globally unique key. Only complete pages whose KV entries have been materialized are shared; partial pages remain request-private and are released at completion. The final sampled token is excluded because it has not yet been decoded into KV.

`bind_tasks()` returns a `KVCache` dataclass with `kv_indptr`, a prefix-sum index over sequence lengths computed once per step and shared across layers. Attention layers access buffers directly — no methods, no abstraction.

### Attention dispatch

The extension separates runtime choice, per-operator policy, and exact kernels:

- `runtime/context.py` owns nested `op_backend(...)` scopes and snapshots.
- `runtime/operation.py` defines `OpSpec` and `OpDispatcher`.
- `backend/attention/dispatcher.py` selects CUDA, FlashAttention, or torch
  functions according to the call's `AttentionSpec`.
- `kernel/` adapters invoke one compiled kernel and raise if it is unavailable.

CUDA attention writes and reads the paged KV cache for supported inference
prefill and decode calls. FlashAttention supports compatible inference and
training calls; torch SDPA is the reference fallback. Automatic selection
tries CUDA, FlashAttention, then torch. Explicit arguments and contexts are
strict; `ASTR_OPS` and `set_op` are soft process defaults.

```python
from astrai.extension import attention, op_backend

with op_backend(attention="torch"):
    output = attention(q, k, v)
```

Create an inference scheduler inside the desired scope to capture that choice
for its worker. Rotary embedding dispatches independently and uses torch
autograd whenever backward is needed. CUDA Graph eligibility is checked in
the inference worker against its captured attention choice.

## Mask Algorithm Internals

### Template mode (`template: true`)

1. Prepend BOS token (masked)
2. For each message in the field's array:
   1. Render through `chat_template` for that single message
   2. Encode rendered text
   3. Apply mask rule for the message's role

### Non-template mode

Encode the field value as text. Mask value is 1 (train) or 0 (mask) per the section's `action`.

### Text config detection

When no section uses `template` and all sections have `action: "train"`, the builder omits `loss_mask` from the output — all tokens are trained.

### Position ID strategies

| Mode | Behavior |
|------|----------|
| `none` | No position IDs generated |
| `doc_reset` | Reset position to 0 at each document boundary in packed sequences |
| `continuous` | Continuous position IDs across packed documents |

Default is `doc_reset`, which ensures each document in a packed bin starts from position 0, preventing position encoding drift between unrelated documents.

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
must do the same. `GradientState`, `AccumOptimizer`, `AccumScheduler`,
`executor.accumulate()` and `window_steps` are removed. Calling `optimizer.step()`
outside a commit boundary is no longer silently suppressed.

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

> Document Update Time: 2026-09-05
