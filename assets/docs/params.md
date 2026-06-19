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
| `--train_type` | Training type (`seq`, `sft`, `dpo`, `grpo`) | required |
| `--data_root_path` | Dataset root directory | required |
| `--param_path` | Model parameters or checkpoint path | required |
| `--n_epoch` | Total training epochs | 1 |
| `--batch_per_device` | Batch size per device | 1 |
| `--grad_accum_steps` | Gradient accumulation steps between optimizer steps | 1 |

### Learning Rate Scheduling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--warmup_ratio` | Fraction of total steps used for LR warmup | 0.05 |
| `--max_lr` | Maximum learning rate (cosine decay after warmup) | 3e-4 |
| `--max_grad_norm` | Maximum gradient norm for clipping | 1.0 |

### Optimizer (AdamW)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--adamw_beta1` | AdamW beta1 | 0.9 |
| `--adamw_beta2` | AdamW beta2 | 0.95 |
| `--adamw_weight_decay` | AdamW weight decay | 0.01 |

### Data Loading

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--window_size` | Max input sequence length | model config `max_len` |
| `--stride` | Stride for sliding window over sequences | None |
| `--random_seed` | Random seed for reproducibility | 3407 |
| `--num_workers` | DataLoader worker processes | 4 |
| `--no_pin_memory` | Disable pin_memory (enabled by default) | (flag) |

### Checkpoint & Resume

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ckpt_interval` | Iterations between checkpoints | 5000 |
| `--ckpt_dir` | Checkpoint save directory | checkpoint |
| `--start_epoch` | Resume from epoch (0 = from scratch) | 0 |
| `--start_batch` | Resume from batch iteration | 0 |

### Validation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--val_split` | Ratio to split from training dataset for validation (e.g. 0.05) | None |
| `--val_step` | Number of optimizer steps between validation runs | 1000 |

### Logging

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--log_dir` | Directory for metric logs | checkpoint/logs |
| `--log_interval` | Number of batch iterations between metric logs | 100 |
| `--metrics` | Metrics to log (e.g. --metrics loss lr val_loss) | ["loss", "lr"] |

### Gradient Checkpointing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gradient_checkpointing` | Enable activation checkpointing for DecoderBlock modules | False |

### Distributed Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--nprocs` | Number of GPUs / processes | 1 |
| `--parallel_mode` | Parallel strategy (`none`, `ddp`, or `fsdp`) | none |
| `--device_type` | Device type | cuda |
| `--start_method` | Multiprocessing start method (`spawn`, `fork`, `forkserver`) | spawn |
| `--backend` | Distributed training backend | nccl |
| `--master_addr` | Master node address | localhost |
| `--master_port` | Master node port | 29500 |

### Strategy-specific

| Parameter | Description | Default | Used by |
|-----------|-------------|---------|---------|
| `--dpo_beta` | DPO beta value | 0.1 | `dpo` |
| `--label_smoothing` | Label smoothing for cross-entropy loss | 0.0 | `seq`, `sft` |
| `--group_size` | GRPO group size | 4 | `grpo` |
| `--grpo_clip_eps` | GRPO clipping epsilon | 0.2 | `grpo` |
| `--grpo_kl_coef` | GRPO KL penalty coefficient | 0.01 | `grpo` |
| `--grpo_sync_interval` | GRPO ref_model sync interval (steps) | 200 | `grpo` |
| `--neftune_alpha` | NEFTune noise alpha (0=disabled, typical: 5.0) | 0.0 | `sft` |

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
    --adamw_beta1=0.9 \
    --adamw_beta2=0.95 \
    --adamw_weight_decay=0.01 \
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
| `--host` | str | `0.0.0.0` | Host address |
| `--port` | int | `8000` | Port number |
| `--param_path` | path | `project_root/params` | Path to model parameters |
| `--device` | str | `cuda` | Device to load model on |
| `--dtype` | str | `bfloat16` | Model weights dtype (`bfloat16`, `float16`, `float32`) |
| `--max_batch_size` | int | `16` | Maximum batch size for continuous batching |
| `--reload` | flag | `False` | Enable auto-reload for development |

Usage:
```bash
python scripts/tools/server.py --param_path ./params --device cuda --dtype bfloat16
```

See [Inference Guide](inference.md) for HTTP API documentation.

## Generate (`generate.py`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--param_path` | str | required | Path to the model directory |
| `--input_json_file` | str | required | Path to the input JSONL file |
| `--output_json_file` | str | required | Path to the output JSONL file |
| `--question_key` | str | `question` | Key for the question in input JSON |
| `--response_key` | str | `response` | Key for the response in output JSON |
| `--temperature` | float | `0.60` | Sampling temperature |
| `--top_k` | int | `30` | Top-k filtering |
| `--top_p` | float | `0.95` | Nucleus sampling threshold |
| `--batch_size` | int | `1` | Batch size for generation |
| `--max_tokens` | int | model config `max_len` | Maximum tokens to generate |

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
| `input_files` | path(s) | required | Input JSONL file(s), supports glob (`data/*.jsonl`) |
| `--output_dir`, `-o` | path | required | Output directory for processed data |
| `--config`, `-c` | path | required | Preprocessing pipeline config (JSON) |
| `--num_workers` | int | `4` | Number of parallel workers |

Usage:
```bash
python scripts/tools/preprocess.py data/*.jsonl -o output/ -c sft.json
```

See [Preprocessing Guide](preprocessing.md) for config file format and examples.

---

> Document Update Time: 2026-06-19