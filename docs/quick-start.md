# Quick Start

This guide walks you through installing AstrAI, downloading a model, running inference, preprocessing data, and launching your first training job.

## Prerequisites

- **Python 3.12+**
- **PyTorch 2.13.0** (the exact version pinned by AstrAI; CUDA 13.0 build recommended for GPU support)
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

Optional kernel build and backend selection are documented in
[CUDA kernels](developer/kernels/README.md).

## 2. Download Model Weights

Run the download script; files are saved to `params/`:

```bash
python scripts/demo/download.py
```

For another repository or destination:

```bash
python scripts/demo/download.py --repo-id <HF_REPO_ID> --local-dir ./my_model
```

Use the downloaded directory directly. File conventions and external checkpoint
imports are described in [HF files](developer/huggingface.md).

## 3. Run Inference

### Interactive Chat (Simplest)

```bash
python scripts/demo/stream_chat.py
# Type your message after >>, press Ctrl+C or Ctrl+D to exit
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

For SFT, DPO and GRPO schemas, use the
[preprocessing guide](guides/preprocessing.md); their masks and record fields
are part of the data contract.

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

For multi-GPU launch settings, see [distributed training](guides/distributed.md).
For other objectives, precision settings and resume behavior, see
[training](guides/training.md). Run `python scripts/tools/train.py --help` for
all CLI options.

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

Use the repository wrappers for image, device and volume configuration:
[Docker serving](developer/docker/serving.md) and
[Docker training](developer/docker/training.md).

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
| YAML-driven containerized serving | [Docker Serving](developer/docker/serving.md) |
| YAML-driven containerized training | [Docker Training](developer/docker/training.md) |
