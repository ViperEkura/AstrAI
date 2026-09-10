# Distributed Training

The `--dp_mode` setting selects the data-parallel gradient-sync strategy: **single GPU** (`none`), **replicated with gradient all-reduce** (`ddp`), or **sharded** (`fsdp`), plus **Context Parallel** (`--cp_size`) as an orthogonal sequence-sharding dimension for long-context pretraining. This guide covers when to use each, how to launch multi-GPU training, and how gradient accumulation works.

## Contents

- [Quick Start](#quick-start)
- [DP Modes](#dp-modes)
- [Context Parallelism](#context-parallelism)
- [Gradient Accumulation](#gradient-accumulation)
- [Process Launching](#process-launching)
- [NCCL Troubleshooting](#nccl-troubleshooting)
- [Checkpoint Saving](#checkpoint-saving)
- [Total Steps Calculation](#total-steps-calculation)
- [Real Examples](#real-examples)
- [CLI Parameters](#cli-parameters)

## Quick Start

### Single GPU

```bash
python scripts/tools/train.py \
    --train_type=sft \
    --param_path ./params \
    --data_root_path ./dataset \
    --dp_mode=none \
    --dp_size=1 \
    --batch_per_device=4 \
    --grad_accum_steps=8
```

### Multi-GPU DDP (4 GPUs)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/tools/train.py \
    --train_type=sft \
    --param_path ./params \
    --data_root_path ./dataset \
    --dp_mode=ddp \
    --dp_size=4 \
    --batch_per_device=4 \
    --grad_accum_steps=8
```

### Multi-GPU FSDP (4 GPUs)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/tools/train.py \
    --train_type=sft \
    --param_path ./params \
    --data_root_path ./dataset \
    --dp_mode=fsdp \
    --dp_size=4 \
    --batch_per_device=4 \
    --grad_accum_steps=8
```

> `--dp_mode` defaults to `fsdp`. You can omit it for FSDP.

## DP Modes

| Mode | `--dp_mode` | Param Layout | Memory | When to Use |
|------|-------------------|--------------|--------|-------------|
| Single GPU | `none` | Full, replicated | Highest | Small models, DPO/GRPO, debugging |
| DDP | `ddp` | Full, replicated | High | Most multi-GPU training |
| FSDP | `fsdp` | Sharded (DTensor) | Lowest | Large models that don't fit in single GPU |

### NoneExecutor

No wrapping. The model runs as-is on a single device. Gradient accumulation still works via `AccumOptimizer`/`AccumScheduler` (they gate `step()` on the sync counter). Checkpoint saving is a plain `state_dict()` call.

### DDPExecutor

Wraps the model with `torch.nn.parallel.DistributedDataParallel`. Each rank has a full copy of the model; gradients are all-reduced across ranks. Uses `gradient_as_bucket_view=True` and `broadcast_buffers=False` by default (hardcoded in `train.py`).

During gradient accumulation, non-sync micro-steps use `model.no_sync()` to skip gradient all-reduce. Only the final micro-step triggers the all-reduce.

### FSDPExecutor (FSDP2 / `fully_shard`)

Uses PyTorch's FSDP2 per-module API (`torch.distributed.fsdp.fully_shard`). Each model child (e.g., each `DecoderBlock`) is individually sharded — parameters become `DTensor`s distributed across ranks. No `FlatParameter`, original parameter names are preserved.

Key differences from DDP:
- **Lower memory**: parameters are sharded, not replicated.
- **Custom grad norm**: FSDP gradients are `DTensor`s, so `clip_grad_norm` computes the local norm, then all-reduces to get the global norm.
- **Collective checkpoint ops**: `unshard()` and `full_tensor()` are collective — all ranks must call them even though only rank-0 saves. The executor handles this via `dist.barrier()` in `checkpoint_context`.
- **Root skipped**: `fully_shard` is applied to direct children only (not the root model) due to an `ABC + Generic[T]` MRO incompatibility.

## Context Parallelism

Context parallelism (`--cp_size`) shards the **sequence** dimension across a group of contiguous ranks — orthogonal to DDP/FSDP, which shard the batch and the parameters. It targets long-context pretraining where activation memory scales with `batch × seq_len` and a single rank cannot hold the window.

```bash
export CUDA_VISIBLE_DEVICES=0,1
ASTR_BACKEND=torch python scripts/tools/train.py \
    --train_type=seq \
    --param_path ./params \
    --data_root_path ./dataset/cached \
    --dp_mode=fsdp \
    --dp_size=1 \
    --cp_size=2 \
    --window_size=32768 \
    --batch_per_device=1 \
    --grad_accum_steps=8
```

How it works:

- The world decomposes as `dp × cp × tp` (`astrai/parallel/topology.py`); cp requires `tp_size == 1` for now. Ranks in one cp group consume the **same** batches — the data sampler shards over dp groups only — and each rank holds a contiguous slice of every sequence.
- All per-token computation (norms, projections, MLP, RoPE) runs locally on plain tensors. Only attention crosses ranks: the training forward routes through torch SDPA, which `torch.distributed.tensor.experimental.context_parallel` replaces with ring attention inside the loss computation (`astrai/parallel/cp.py`). The custom CUDA attention kernels and FlashAttention do not participate and CP fails loudly if they are active — set `ASTR_BACKEND=torch`.
- Strategies stay CP-oblivious: seq/sft implement the token-mean two-phase protocol (`forward_tokens` + `reduce_loss` in `BaseStrategy`), and `CPStrategy` decorates them — it shards the batch buffers between the phases and rescales the local reduction to the global mean (`CPState.mean_loss`: the true mean for logging, the cp-scaled loss for backward so DDP/FSDP's averaged gradient reduce reproduces the single-device gradient).
- Weights, optimizer state, and checkpoints are untouched — CP shards activations only.

Constraints (enforced with explicit errors at launch):

- `--train_type` of `seq` or `sft` — i.e. strategies declaring a token-mean loss reduction (`LossReduction.TOKEN_MEAN`); sequence-level strategies are refused at launch. SFT requires one document per row (no packing): the doc-boundary attention mask cannot ride the ring attention kernels, so `SFTStrategy.shard_spec` rejects packed batches. RL strategies (dpo/grpo/online_*) additionally need cross-shard logprob handling plus a context-parallel rollout path and stay unsupported.
- `--dp_mode` must be `ddp` or `fsdp` (unsynchronized cp peers would hold gradients of different slices).
- Requires a flash/cuDNN SDPA kernel — train in bf16. The mem-efficient kernel's logsumexp layout breaks the ring's partial-attention merge, so `CPState.shard` restricts SDPA to flash/cuDNN and rejects others.
- MoE is not supported yet (the load-balance aux loss would be computed per sequence slice).
- Head-tail causal load balancing is disabled (`_cp_options.enable_load_balance = False`): with contiguous chunks, later ranks do more cross-chunk attention work, but the math is exact. The balanced path produced incorrect losses on torch 2.11 and is revisited when upstream fixes land.

## Tensor Parallelism

Tensor parallelism (`--tp_size`) shards the **feature** dimension across a group of contiguous ranks — model parallelism inside a layer, orthogonal to DDP/FSDP (batch and parameters) and CP (sequence). It targets the case where a single layer's projections dominate activation or weight memory.

```bash
export CUDA_VISIBLE_DEVICES=0,1
python scripts/tools/train.py \
    --train_type=seq \
    --param_path ./params \
    --data_root_path ./dataset/cached \
    --dp_mode=fsdp \
    --dp_size=1 \
    --tp_size=2 \
    --batch_per_device=4 \
    --grad_accum_steps=8
```

How it works:

- `astrai/parallel/tp.py` shards the one operator that carries weights: `Linear`. A plan maps module keys (fnmatch patterns over `named_modules()`) to split styles — `colwise` chunks the weight along dim 0 (attention q/k/v over heads, MLP up/gate over ffn channels; each rank computes its own output features, no communication) and `rowwise` chunks along dim 1 (attention o_proj, MLP down; each rank computes a partial sum, one all-reduce ends the forward).
- Everything between a colwise and a rowwise projection is per-feature elementwise (RoPE per head, SiLU per channel, norms over the fully reduced hidden), so the model code between the two ends of a shard never observes the split. The `lm_head` and the embedding stay replicated.
- Communication rides megatron-style autograd functions: rowwise all-reduces on forward and passes gradients through; colwise passes the forward and all-reduces gradients on backward.
- Every rank evaluates the **full** loss (the hidden state is whole at each rowwise boundary), so a rank's gradient on its own shard is already the exact single-device gradient — no loss rescaling. DDP/FSDP gradient sync is forced onto the dp group: tp peers hold different shards, and a world-group default would all-reduce unrelated shards together.
- Sharding happens after the full weights load and before the executor wraps and builds the optimizer (`train_context.before_wrap`); the shard is final, never restored. A tp=2 test pins forward logits/loss and gradients exactly against the single-device run.

Constraints (enforced with explicit errors at launch):

- `--tp_size > 1` cannot be combined with `--cp_size > 1` — the ring-attention patch and head-sharded projections interact on the SDPA inputs and the combination is unverified.
- MoE is refused (expert dispatch and the per-token aux loss assume full-feature router inputs), and so is LoRA (the `LoRALinear` wrapper bypasses the sharded `Linear` forward).

## Gradient Accumulation

Gradient accumulation lets you simulate a larger effective batch size by accumulating gradients over multiple micro-batches before calling `optimizer.step()`.

```
Effective batch = dp_size × batch_per_device × grad_accum_steps
```

cp ranks share each batch (they shard the sequence), so `--cp_size` does not
scale the effective batch. The launcher starts `dp_size × cp_size` processes.

Example: dp_size 4 × batch 4 × accum 8 = effective batch 256.

### How it works

Three cooperating layers:

1. **`GradientState`** — tracks the micro-step counter. Fires `sync_gradients=True` every `grad_accum_steps` micro-batches.
2. **`executor._no_sync(model)`** — suppresses gradient synchronization on non-sync micro-steps:
   - `none`: `nullcontext` (nothing to skip)
   - `ddp`: `model.no_sync()` (skips all-reduce)
   - `fsdp`: `set_requires_gradient_sync(False)` on each `FSDPModule`
3. **`AccumOptimizer` / `AccumScheduler`** — gate `step()` and `zero_grad()` on `sync_gradients`, so the optimizer only fires on the last micro-step.

The loss is divided by `grad_accum_steps` before `backward()`, so gradients sum to the correct mean.

## Process Launching

AstrAI auto-detects the launch method:

| Detection | Strategy | Use Case |
|-----------|----------|----------|
| `torchelastic` / `torchrun` env vars | `TorchrunStrategy` | External orchestrator (`torchrun`, K8s) |
| `RANK` + `WORLD_SIZE` env vars | `TorchrunStrategy` | External launch |
| Neither | `LocalStrategy` | `python scripts/tools/train.py` (in-process spawn) |

### Local (default)

When you run `python scripts/tools/train.py --dp_size=4`, AstrAI derives the process count (`dp_size × cp_size`) and uses `torch.multiprocessing.start_processes` to spawn the child processes. The parent process manages signal forwarding (SIGTERM/SIGINT) and waits for all children to finish.

### Torchrun

For multi-node or SLURM environments:

```bash
torchrun --nproc_per_node=4 scripts/tools/train.py \
    --train_type=sft \
    --dp_mode=ddp \
    --dp_size=4 \
    --param_path ./params \
    --data_root_path ./dataset \
    --batch_per_device=4
```

When launched via `torchrun`, the launcher creates the worker processes. AstrAI reads `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` from the environment and uses `TorchrunStrategy`; the parallelism degrees do not control process creation in this mode.

The parallelism degrees still drive scheduler `total_steps`. Set them so `dp_size × cp_size` equals the global `WORLD_SIZE` (including multi-node runs); a mismatched decomposition fails loudly at startup.

Raw Slurm variables such as `SLURM_PROCID`, `SLURM_NTASKS`, and `SLURM_LOCALID` are not recognized automatically. Launch through `torchrun`, or map the scheduler's variables to `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and `MASTER_PORT` before starting AstrAI. The same requirement applies to launchers that expose only OpenMPI-specific variables.

## NCCL Troubleshooting

The following variables are troubleshooting options for hardware or network configurations where NCCL hangs or fails. They are not general requirements and can reduce performance by disabling peer-to-peer or GPUDirect RDMA paths:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
```

Apply them only after confirming the relevant NCCL transport is the source of the failure. AstrAI does not set them in Python.

## Checkpoint Saving

Checkpoints are saved by **rank-0 only**. The flow:

1. `executor.checkpoint_context(model)` — wraps with `dist.barrier()` before and after (distributed only).
2. `executor.unwrap_model(model)` — gathers the full state dict:
   - `none`: `model.state_dict()`
   - `ddp`: `model.module.state_dict()`
   - `fsdp`: `unshard()` → `full_tensor()` → `reshard()` (collective on all ranks, result kept only on rank-0)
3. Non-rank-0 ranks get `None` — the save is skipped.
4. Rank-0 writes metadata, weights, optional optimizer/scheduler state, and a
   checksum manifest to a hidden sibling directory, then atomically renames the
   complete checkpoint into place.

> **FSDP note**: Even though only rank-0 saves, all ranks must participate in `unwrap_model` because `unshard()` and `full_tensor()` are collective operations. The barriers in `checkpoint_context` keep all ranks in lockstep.

## Total Steps Calculation

The scheduler's total step count accounts for data-parallel sharding:

```
samples_per_replica = ceil(dataset_len / dp_size)
batches_per_replica  = ceil(samples_per_replica / batch_per_device)
total_steps          = (batches_per_replica // grad_accum_steps) * n_epoch
```

This ensures the LR schedule is correctly scaled regardless of the number of GPUs.

## Real Examples

### Pretraining (seq, DDP, 4 GPUs)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python scripts/tools/train.py \
    --train_type=seq \
    --param_path ./params \
    --data_root_path ./dataset/cached \
    --dp_mode=ddp \
    --dp_size=4 \
    --n_epoch=1 \
    --max_lr=2e-4 \
    --schedule_type=wsd \
    --warmup_ratio=0.02 \
    --window_size=2048 \
    --batch_per_device=4 \
    --grad_accum_steps=32 \
    --ckpt_interval=2000
# Effective batch = 4 × 4 × 32 = 512
```

### SFT (DDP, 4 GPUs)

```bash
python scripts/tools/train.py \
    --train_type=sft \
    --param_path ./AstrAI-V1-base \
    --data_root_path ./dataset/cached_sft \
    --dp_mode=ddp \
    --dp_size=4 \
    --n_epoch=2 \
    --max_lr=2e-5 \
    --schedule_type=cosine \
    --warmup_ratio=0.02 \
    --min_rate=0.05 \
    --window_size=2048 \
    --batch_per_device=4 \
    --grad_accum_steps=8
# Effective batch = 4 × 4 × 8 = 128
```

### DPO (Single GPU)

```bash
python scripts/tools/train.py \
    --train_type=dpo \
    --param_path ./checkpoint/epoch_1_step_6000 \
    --data_root_path ./alpaca_dpo.jsonl \
    --dp_mode=none \
    --dp_size=1 \
    --max_lr=5e-6 \
    --schedule_type=cosine \
    --warmup_ratio=0.1 \
    --min_rate=0.1 \
    --window_size=1024 \
    --batch_per_device=4 \
    --grad_accum_steps=8 \
    --dpo_beta=0.1 \
    --max_grad_norm=50
```

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dp_size` | 1 | Data-parallel replicas; the launcher starts `dp_size × cp_size × tp_size` processes. Under `torchrun`, keep the product equal to the global `WORLD_SIZE` |
| `--dp_mode` | `fsdp` | `none`, `ddp`, or `fsdp` |
| `--cp_size` | 1 | Context-parallel group size: shards sequences across contiguous ranks (seq pretraining, requires `ddp`/`fsdp` and the torch attention backend) |
| `--tp_size` | 1 | Tensor-parallel group size: shards Linear projections over features (attention heads / ffn channels; refuses cp+tp, MoE, and LoRA) |
| `--start_method` | `spawn` | Multiprocessing start method (`spawn`, `fork`, `forkserver`) |
| `--backend` | `nccl` | Distributed backend (`nccl`, `gloo`) |
| `--master_addr` | `localhost` | Master node address |
| `--master_port` | `29500` | Master node port |
| `--device_type` | `cuda` | Device type |


Full parameter reference: [CLI Reference](params.md). Training loop and strategies: [Training Guide](training.md).

> Document Update Time: 2026-09-05
