# Getting Started

This guide walks you through installing AstrAI, downloading a model, running inference, preprocessing data, and launching your first training job.

## Contents

- [Prerequisites](#prerequisites)
- [1. Install](#1-install)
- [2. Download Model Weights](#2-download-model-weights)
- [3. Run Inference](#3-run-inference)
- [4. Preprocess Data](#4-preprocess-data)
- [5. Train](#5-train)
- [6. Evaluate](#6-evaluate)
- [7. Docker](#7-docker)
- [Next Steps](#next-steps)

## Prerequisites

- **Python 3.12+**
- **PyTorch 2.11.0** (the exact version pinned by AstrAI; CUDA 12.8 build recommended for GPU support)
- NVIDIA GPU with CUDA for training, `scripts/tools/generate.py`, generation evaluations, and demos. The HTTP server and direct-scoring evaluations can run on CPU where their CLI exposes a CPU device.

## 1. Install

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI

# Kernels auto-build when nvcc + CUDA are detected; skip with CSRC_KERNELS=false
pip install -e .

# Force the CUDA kernel build (fused attention, rotary embedding, FP8 GEMM)
# CSRC_KERNELS=true pip install -e . --no-build-isolation

# With dev dependencies (pytest, ruff)
# pip install -e ".[dev]"
```

> **CUDA kernels** build automatically when `nvcc` is on `PATH` and
> `torch.cuda.is_available()` returns `True`; set `CSRC_KERNELS=false` to skip
> them, or `CSRC_KERNELS=true` to force them (required when building in an
> isolated environment with `--no-build-isolation`). Once built, `CudaBackend`
> is the default attention backend on GPU (cuda > flash > torch priority).
> Override via `ASTR_BACKEND` env var or `attn_backend()` context manager.
> Fused rotary embedding kernel is auto-dispatched when available. Skip for
> CPU-only usage.

## 2. Download Model Weights

AstrAI uses HuggingFace-style model directories. Download the default 1B instruction-tuned model:

```bash
python scripts/demo/download.py
# → Downloads to params/
```

To use a different model:

```bash
python scripts/demo/download.py --repo-id <HF_REPO_ID> --local-dir ./my_model
```

The model directory contains:
- `config.json` — model architecture configuration
- `model.safetensors` — model weights
- `tokenizer.json` + `tokenizer_config.json` — tokenizer files (including chat template)

External HuggingFace checkpoints of the LLaMA layout (e.g. `meta-llama/...`,
`mistralai/...`, `Qwen/Qwen2-...`) can be loaded directly: `AutoModel.from_pretrained`
auto-detects HF `model_type` / key names (`input_layernorm`, `gate_proj`, MoE
`experts.<j>` ...) and converts config and weights in place. Dense and MoE
(Mixtral / DeepSeek-V3 layout) FFNs are supported; MLA attention
(DeepSeek-V2/V3) and biased projections (`attention_bias`) are not. Pass
`weights_format="astrai"` to skip conversion, or `"hf"` to force it.

## 3. Run Inference

### Interactive Chat (Simplest)

```bash
python scripts/demo/stream_chat.py
# Type your message after >>, type !exit to quit
```

This starts a single-turn interactive prompt loop with streaming output. Each prompt is independent; conversation history is not retained.

### Start an HTTP Server

```bash
# Terminal 1: start server
python scripts/tools/server.py --param_path ./params --device cuda

# Terminal 2: query (OpenAI-compatible API)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

The server also supports the Anthropic API at `/v1/messages`. See [Inference Guide](guides/inference.md) for full API documentation.

### Batch Generation from a File

Create an input JSONL file (one JSON object per line):

```json
{"question": "What is machine learning?"}
{"question": "Explain gradient descent."}
```

```bash
python scripts/tools/generate.py \
    --param_path ./params \
    --input_json_file input.jsonl \
    --output_json_file output.jsonl
```

## 4. Preprocess Data

AstrAI uses a declarative JSON config to define the preprocessing pipeline. Create a config file for your training type:

### Pretraining (seq)

Input JSONL:
```json
{"text": "Artificial intelligence is..."}
```

Config (`pretrain.json`):
```json
{
    "input": {
        "sections": [{"field": "text", "action": "train"}]
    },
    "preprocessing": {"max_seq_len": 2048},
    "output": {"storage_format": "bin"}
}
```

### SFT (Supervised Fine-Tuning)

Input JSONL:
```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

Config (`sft.json`):
```json
{
    "input": {
        "sections": [{"field": "messages", "action": "$role", "template": true}]
    },
    "mask": {
        "system": "mask",
        "user": "mask",
        "assistant": "train"
    },
    "mask_default": "mask",
    "preprocessing": {"max_seq_len": 2048},
    "output": {"storage_format": "bin", "dtype": {"loss_mask": "bool"}}
}
```

### Run Preprocessing

```bash
python scripts/tools/preprocess.py data/*.jsonl -o output/ -c pretrain.json
```

See [Preprocessing Guide](guides/preprocessing.md) for DPO/GRPO configs and all options.

## 5. Train

### Single GPU

```bash
python scripts/tools/train.py \
    --train_type=seq \
    --data_root_path=/path/to/dataset \
    --param_path=./params \
    --batch_per_device=4 \
    --grad_accum_steps=8 \
    --max_lr=1e-4 \
    --window_size=2048 \
    --ckpt_dir=./checkpoint \
    --dp_size=1 \
    --dp_mode=none
```

### Multi-GPU (DDP)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Only if this host's NCCL transport is broken; see docs/guides/distributed.md:
# export NCCL_P2P_DISABLE=1
# export NCCL_NET_GDR_LEVEL=0

python scripts/tools/train.py \
    --train_type=seq \
    --data_root_path=/path/to/dataset \
    --param_path=./params \
    --dp_mode=ddp \
    --dp_size=4 \
    --batch_per_device=4 \
    --grad_accum_steps=8 \
    --max_lr=1e-4 \
    --window_size=2048 \
    --ckpt_dir=./checkpoint
```

### Training Types

| `--train_type` | Description | Data Keys |
|----------------|-------------|-----------|
| `seq` | Pre-training (next-token prediction) | `sequence` |
| `sft` | Supervised fine-tuning (masked loss) | `sequence`, `loss_mask` |
| `dpo` | Direct Preference Optimization | `chosen`, `rejected`, `*_mask` |
| `grpo` | Group Relative Policy Optimization | `prompts`, `responses`, `masks`, `rewards` |

See [Training Guide](guides/training.md) for loss formulas and strategies. See [Distributed Guide](guides/distributed.md) for DDP/FSDP details.

## 6. Evaluate

HumanEval and MMLU download their benchmark data through HuggingFace
`datasets`, which is not part of the base install:

```bash
pip install datasets
```

```bash
# HumanEval (code generation, auto-downloads dataset)
python scripts/eval/evaluate_humaneval.py --param_path ./params --num_samples 20

# MMLU (knowledge, auto-downloads dataset)
python scripts/eval/evaluate_mmlu.py --param_path ./params --n_shot 5

# Perplexity on custom data
python scripts/eval/evaluate_ppl.py --param_path ./params --input_path data.jsonl --output_dir ppl_results/
```

See [Evaluation Guide](guides/evaluation.md) for all benchmarks.

## 7. Docker

```bash
# Build
docker build -t astrai:latest .

# Run inference server with GPU
docker run --gpus all -p 8000:8000 astrai:latest \
  python scripts/tools/server.py --port 8000 --device cuda

# Docker Compose (GPU)
docker compose up -d
```

## Next Steps

| Topic | Document |
|-------|----------|
| CLI parameters (train, server, generate, preprocess) | [CLI Reference](guides/params.md) |
| Preprocessing pipeline details | [Preprocessing Guide](guides/preprocessing.md) |
| Training loop, strategies, schedulers | [Training Guide](guides/training.md) |
| KV cache, continuous batching, HTTP API | [Inference Guide](guides/inference.md) |
| Evaluation benchmarks | [Evaluation Guide](guides/evaluation.md) |
| Multi-GPU DDP / FSDP | [Distributed Guide](guides/distributed.md) |
| System architecture | [Architecture](developer/architecture.md) |
| Data pipeline internals | [Data Flow](developer/dataflow.md) |
| YAML-driven containerized serving | [Docker Serving](developer/docker-serving.md) |
| YAML-driven containerized training | [Docker Training](developer/docker-training.md) |

> Document Update Time: 2026-08-22
