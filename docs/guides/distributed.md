# Distributed Training

AstrAI supports three parallel modes: **single GPU** (`none`), **Data Parallel** (`ddp`), and **Fully Sharded Data Parallel** (`fsdp`). This guide covers when to use each, how to launch multi-GPU training, and how gradient accumulation works.

## Contents

- [Quick Start](#quick-start)
- [Parallel Modes](#parallel-modes)
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
    --parallel_mode=none \
    --nprocs=1 \
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
    --parallel_mode=ddp \
    --nprocs=4 \
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
    --parallel_mode=fsdp \
    --nprocs=4 \
    --batch_per_device=4 \
    --grad_accum_steps=8
```

> `--parallel_mode` defaults to `fsdp`. You can omit it for FSDP.

## Parallel Modes

| Mode | `--parallel_mode` | Param Layout | Memory | When to Use |
|------|-------------------|--------------|--------|-------------|
| Single GPU | `none` | Full, replicated | Highest | Small models, DPO/GRPO, debugging |
| DDP | `ddp` | Full, replicated | High | Most multi-GPU training |
| FSDP | `fsdp` | Sharded (DTensor) | Lowest | Large models that don't fit in single GPU |

### NoneExecutor

No wrapping. The model runs as-is on a single device. Gradient accumulation still works via `AccumOptimizer`/`AccumScheduler` (they gate `step()` on the sync counter). Checkpoint saving is a plain `state_dict()` call.

### DDPExecutor

Wraps the model with `torch.nn.parallel.DistributedDataParallel`. Each rank has a full copy of the model; gradients are all-reduced across ranks. Uses `gradient_as_bucket_view=True` and `broadcast_buffers=False` by default (hardcoded in `train.py`).

During gradient accumulation, non-sync micro-steps use `model.no_sync()` to skip gradient all-reduce. Only the final micro-step triggers the all-reduce.

For `online_grpo` and `online_dpo`, each rank gives the in-process inference
scheduler an explicit view of the replicated `DDP.module`. Rollout never
depends on DDP forwarding unknown attributes such as `config`, and training
continues to use the wrapped model. The inference view does not issue DDP
collectives, so ranks may generate different response lengths without
deadlocking; the following training forward/backward step returns to the DDP
wrapper and synchronizes gradients normally.

### FSDPExecutor (FSDP2 / `fully_shard`)

Uses PyTorch's FSDP2 per-module API (`torch.distributed.fsdp.fully_shard`). Each model child (e.g., each `DecoderBlock`) is individually sharded — parameters become `DTensor`s distributed across ranks. No `FlatParameter`, original parameter names are preserved.

Key differences from DDP:
- **Lower memory**: parameters are sharded, not replicated.
- **Custom grad norm**: FSDP gradients are `DTensor`s, so `clip_grad_norm` computes the local norm, then all-reduces to get the global norm.
- **Collective checkpoint ops**: `unshard()` and `full_tensor()` are collective — all ranks must call them even though only rank-0 saves. The executor handles this via `dist.barrier()` in `checkpoint_context`.
- **Root skipped**: `fully_shard` is applied to direct children only (not the root model) due to an `ABC + Generic[T]` MRO incompatibility.

Distributed FSDP cannot currently provide the in-process inference scheduler
with a replicated model view. Online rollout therefore fails before tokenizer,
KV-cache, or scheduler construction instead of passing sharded `DTensor`
parameters into an unsupported generation path. `torch.compile` plus online
rollout is likewise rejected before scheduler construction. These combinations
need an explicit weight-materialization/lifecycle design before support can be
enabled.

## Gradient Accumulation

Gradient accumulation lets you simulate a larger effective batch size by accumulating gradients over multiple micro-batches before calling `optimizer.step()`.

```
Effective batch = nprocs × batch_per_device × grad_accum_steps
```

Example: 4 GPUs × batch 4 × accum 8 = effective batch 256.

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

When you run `python scripts/tools/train.py --nprocs=4`, AstrAI uses `torch.multiprocessing.start_processes` to spawn 4 child processes. The parent process manages signal forwarding (SIGTERM/SIGINT) and waits for all children to finish.

### Torchrun

For multi-node or SLURM environments:

```bash
torchrun --nproc_per_node=4 scripts/tools/train.py \
    --train_type=sft \
    --parallel_mode=ddp \
    --nprocs=4 \
    --param_path ./params \
    --data_root_path ./dataset \
    --batch_per_device=4
```

When launched via `torchrun`, the launcher creates the worker processes. AstrAI reads `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` from the environment and uses `TorchrunStrategy`; `--nprocs` does not control process creation in this mode.

The current training CLI still uses `--nprocs` when calculating scheduler `total_steps`. Set it to the global `WORLD_SIZE` so the step count reflects data-parallel sharding, including multi-node runs.

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
samples_per_replica = ceil(dataset_len / nprocs)
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
    --parallel_mode=ddp \
    --nprocs=4 \
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
    --parallel_mode=ddp \
    --nprocs=4 \
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
    --parallel_mode=none \
    --nprocs=1 \
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
| `--nprocs` | 1 | Local process count for AstrAI's launcher; under `torchrun`, set it to global `WORLD_SIZE` for total-step calculation |
| `--parallel_mode` | `fsdp` | `none`, `ddp`, or `fsdp` |
| `--start_method` | `spawn` | Multiprocessing start method (`spawn`, `fork`, `forkserver`) |
| `--backend` | `nccl` | Distributed backend (`nccl`, `gloo`) |
| `--master_addr` | `localhost` | Master node address |
| `--master_port` | `29500` | Master node port |
| `--device_type` | `cuda` | Device type |

> `--tp_size` is accepted by the CLI but discarded before configuration. Tensor parallelism is not implemented, and there is no tensor-parallel module or model integration.

Full parameter reference: [CLI Reference](params.md). Training loop and strategies: [Training Guide](training.md).

> Document Update Time: 2026-08-02
