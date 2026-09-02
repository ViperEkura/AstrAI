# CLI Parameter Reference

## Contents

- [Training Parameters](#training-parameters)
- [Inference Server](#inference-server-serverpy)
- [Generate](#generate-generatepy)
- [Preprocess](#preprocess-preprocesspy)

## Training Parameters

### Basic Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config`, `-c` | YAML config file; explicit CLI options override YAML values | None |
| `--train_type` | Training type (`seq`, `sft`, `dpo`, `grpo`, `online_grpo`, `online_dpo`) | required |
| `--data_root_path` | Dataset root directory | required |
| `--param_path` | Model parameters or checkpoint path | required |
| `--resume` | Resume training from `--param_path` | False |
| `--n_epoch` | Total training epochs | 1 |
| `--batch_per_device` | Batch size per device | 1 |
| `--grad_accum_steps` | Gradient accumulation steps between optimizer steps | 1 |

### Learning Rate Scheduling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--warmup_ratio` | Fraction of total steps used for LR warmup | 0.05 |
| `--max_lr` | Maximum learning rate (cosine decay after warmup) | 3e-4 |
| `--max_grad_norm` | Maximum gradient norm for clipping; `TrainConfig` validates it as positive (or `None`) | 1.0 |

### Optimizer

The default `muon_adamw` optimizer sends matrix parameters through **Muon** and
non-matrix parameters through **AdamW** (`fused=True`).

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--optimizer` | Built-in optimizer (`muon_adamw`, `nora_nadamw`, `mano_adamw`) | `muon_adamw` |
| `--weight_decay` | Weight decay for optimizer parameter groups that are eligible for decay | 0.1 |
| `--muon_momentum` | Muon momentum factor | 0.95 |
| `--muon_nesterov`, `--no-muon_nesterov` | Enable or disable Nesterov momentum for Muon | enabled |
| `--muon_ns_steps` | Newton-Schulz iteration steps for Muon | 5 |
| `--muon_adjust_lr` | Muon LR adjustment strategy (`original`, `match_rms_adamw`) | `match_rms_adamw` |

`nora_nadamw` routes internal `Linear.weight` matrices to **Nora** and
embeddings, the LM head, norms, biases, LoRA factors, and fallback parameters to
**NAdamW**. Parameters are classified by module role and identity, so tied
embedding/head weights occur in exactly one group. Nora requires complete rows
under DTensor sharding and rejects layouts sharded along the last dimension.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--nora_lr` | Nora learning rate | 5e-3 |
| `--nora_beta` | Nora momentum-buffer EMA factor | 0.95 |
| `--nora_momentum` | Nora Nesterov interpolation factor | 0.95 |
| `--nora_weight_decay` | Nora matrix weight decay | 0.0 |

`mano_adamw` routes internal `Linear.weight` matrices to **Mano** (manifold
normalized optimizer) and the remaining parameters to **AdamW**. Mano projects
the momentum onto the tangent space of the Oblique manifold and normalizes it,
alternating the projection axis (row/column) each step — replacing Muon's
Newton-Schulz iteration with a cheaper normalization.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mano_momentum` | Accepted by the CLI but currently ignored by optimizer construction | 0.95 |
| `--mano_nesterov`, `--no-mano_nesterov` | Accepted by the CLI but currently ignored by optimizer construction | enabled |

The two Mano-specific flags are reserved for future wiring; do not rely on them
to change optimizer behavior in the current release.

Optimizer identity and hyperparameters are saved in checkpoint metadata. Optimizer
states are intentionally not interchangeable: resume older MuonAdamW checkpoints
with `--optimizer=muon_adamw`.

### Data Loading

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--window_size` | Max input sequence length | model config `max_position_embeddings` |
| `--stride` | Stride for sliding window over sequences | None |
| `--random_seed` | Random seed for reproducibility | 3407 |
| `--num_workers` | DataLoader worker processes | 4 |
| `--pin_memory`, `--no-pin_memory` | Enable or disable DataLoader pinned memory | enabled |

### Checkpoint & Resume

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ckpt_interval` | Iterations between checkpoints | 5000 |
| `--ckpt_dir` | Checkpoint save directory | checkpoint |
| `--start_epoch` | Resume from epoch (0 = from scratch) | 0 |
| `--start_samples` | Resume from sample count per rank | 0 |

### Validation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--val_split` | Ratio to split from training dataset for validation (e.g. 0.05) | None |
| `--val_step` | Number of optimizer steps between validation runs | 1000 |

### Logging

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--metrics` | Repeatable metric option (for example, `--metrics loss --metrics lr --metrics val_loss`) | `loss`, `lr`, `grad_norm`, `grad_snr` |

### Gradient Checkpointing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gradient_checkpointing`, `--no-gradient_checkpointing` | Enable or disable activation checkpointing for DecoderBlock modules | disabled |

### Miscellaneous

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--compile` | Enable `torch.compile` with mode `default`, `reduce-overhead`, or `max-autotune`; omit to disable | None |
| `--dry-run` | Validate the merged configuration and print the training plan without training | False |

### Distributed Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--nprocs` | Number of GPUs / processes | 1 |
| `--parallel_mode` | Parallel strategy (`none`, `ddp`, `fsdp`) | fsdp |
| `--device_type` | Device type | cuda |
| `--start_method` | Multiprocessing start method (`spawn`, `fork`, `forkserver`) | spawn |
| `--backend` | Distributed training backend | nccl |
| `--master_addr` | Master node address | localhost |
| `--master_port` | Master node port | 29500 |
| `--tp_size` | Reserved tensor-parallel size; accepted but currently ignored | None |

### Strategy-specific

| Parameter | Description | Default | Used by |
|-----------|-------------|---------|---------|
| `--dpo_beta` | DPO beta value | 0.1 | `dpo`, `online_dpo` |
| `--label_smoothing` | Label smoothing for cross-entropy loss | 0.0 | `seq`, `sft` |
| `--group_size` | GRPO/rollout group size | 4 | `grpo`, `online_grpo`, `online_dpo` |
| `--grpo_clip_eps` | GRPO clipping epsilon | 0.2 | `grpo`, `online_grpo` |
| `--grpo_kl_coef` | GRPO KL penalty coefficient | 0.01 | `grpo`, `online_grpo` |
| `--grpo_loss_variant` | Objective variant (`grpo` or `dr_grpo`) | `grpo` | `grpo`, `online_grpo` |
| `--grpo_max_completion_length` | Fixed completion budget used by Dr.GRPO; online mode defaults to `rollout_max_tokens` | None | `grpo`, `online_grpo` |
| `--neftune_alpha` | NEFTune noise alpha (0=disabled, typical: 5.0) | 0.0 | `sft` |

### Online Rollout

`online_grpo` and `online_dpo` are factory aliases for the existing `grpo` and
`dpo` strategy classes; online behavior is enabled by rollout components rather
than separate strategy subclasses. These options apply to the online aliases.
Online strategies require
a `BaseRewardModel` factory in `TrainConfig`; `train.py` does not currently
provide a command-line option for configuring one.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--rollout_interval` | Optimizer steps between rollout refreshes | 512 |
| `--rollout_temperature` | Rollout sampling temperature | 0.7 |
| `--rollout_top_k` | Rollout top-k filtering (`0` disables) | 0 |
| `--rollout_top_p` | Rollout nucleus sampling threshold | 0.9 |
| `--rollout_max_tokens` | Maximum generated tokens per response | 1024 |

### Scheduler

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--schedule_type` | LR scheduler type (`cosine`, `sgdr`, `wsd`) | cosine |
| `--min_rate` | Minimum LR as fraction of base LR | None (all current schedulers use their effective default of 0.01) |
| `--cycle_length` | SGDR first cycle length in steps | None (total_steps - warmup_steps) |
| `--t_mult` | SGDR cycle length multiplier per restart | 2 |
| `--stable_steps` | WSD stable plateau steps | None (80% of post-warmup steps) |
| `--decay_steps` | WSD decay steps | None (total_steps - warmup_steps - stable_steps) |

### Usage Example

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

---

## Inference Server (`server.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config`, `-c` | path | `None` | Serving YAML config. CLI flags override YAML values |
| `--host` | str | `0.0.0.0` | Host address |
| `--port` | int | `8000` | Port number |
| `--param_path` | path | `project_root/params` | Path to model parameters |
| `--device` | str | `cuda` | Device to load model on |
| `--dtype` | str | `bfloat16` | Model weights dtype (`bfloat16`, `float16`, `float32`) |
| `--max_batch_size` | int | `16` | Maximum batch size for continuous batching |
| `--max_seq_len` | int | model config `max_position_embeddings` | Maximum sequence length (KV cache size + prompt truncation) |
| `--reload` | flag | `False` | Enable auto-reload for development |

Usage:
```bash
python scripts/tools/server.py --param_path ./params --device cuda --dtype bfloat16
```

YAML config (a `server:` section; explicit CLI flags override YAML values):
```bash
python scripts/tools/server.py --config serve.yaml
```
```yaml
server:
  host: 0.0.0.0
  port: 8000
  device: cuda
  dtype: bfloat16
  max_batch_size: 16
  max_seq_len: null
```
`serve.yaml` also carries a `runtime:` section for the Docker wrapper; see
[Docker Serving](../developer/docker-serving.md).

See [Inference Guide](inference.md) for HTTP API documentation.

## Generate (`generate.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--param_path` | str | required | Path to the model directory |
| `--input_json_file` | str | required | Path to the input JSONL file |
| `--output_json_file` | str | required | Path to the output JSONL file |
| `--question_key` | str | `question` | Key for the question in input JSON |
| `--response_key` | str | `response` | Key for the response in output JSON |
| `--temperature` | float | `0.8` | Sampling temperature |
| `--top_k` | int | `50` | Top-k filtering |
| `--top_p` | float | `0.95` | Nucleus sampling threshold |
| `--batch_size` | int | `1` | Batch size for generation |
| `--num_samples` | int | `1` | Responses per prompt |
| `--max_seq_len` | int | `2048` | KV cache sequence length |
| `--frequency_penalty` | float | `0.0` | Frequency penalty |
| `--rep_window` | int | `64` | Window size for frequency penalty |

Usage:
```bash
python scripts/tools/generate.py \
    --param_path ./params \
    --input_json_file input.jsonl \
    --output_json_file output.jsonl
```

## Preprocess (`preprocess.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_files` | path(s) | required | One or more existing `.jsonl` or `.json` paths. Wildcards work only when expanded by the invoking shell; the CLI does not expand globs itself. |
| `--output_dir`, `-o` | path | required | Output directory for processed data |
| `--config`, `-c` | path | required | Preprocessing pipeline config (JSON) |
| `--tokenizer_path` | str | `params` | Path to tokenizer directory |
| `--batch_size` | int | config value (`256` by default) | Override records processed per batch; must be at least 1 |

Usage:
```bash
python scripts/tools/preprocess.py data/part-000.jsonl data/part-001.jsonl -o output/ -c sft.json
```

See [Preprocessing Guide](preprocessing.md) for config file format and examples.

---

> Document Update Time: 2026-08-22
