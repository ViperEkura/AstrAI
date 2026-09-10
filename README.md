<div align="center">
  
  <img src="docs/images/logo.png" width="auto" alt="Logo">
  <p>
    <strong>A lightweight Transformer training & inference framework</strong>
  </p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="license">
  <img src="https://img.shields.io/github/v/tag/ViperEkura/AstrAI?label=Release&color=76bad9" alt="release">
  <img src="https://img.shields.io/github/stars/ViperEkura/AstrAI?style=flat&label=Stars&color=76bad9" alt="stars">
  <img src="https://img.shields.io/github/forks/ViperEkura/AstrAI?style=flat&label=Forks&color=76bad9" alt="forks">
</div>
<br>

<div align="center">
  <a href="#english">English</a> •
  <a href="docs/README-zh-CN.md">中文</a> •
  <a href="https://github.com/ViperEkura/AstrAI/issues">Issue Tracker</a> •
  <a href="https://github.com/ViperEkura/AstrAI/discussions">Discussions</a> •
  <a href="https://huggingface.co/ViperEkura">HuggingFace</a>
</div>

<br>

## 📖 Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Demo](#demo)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Community](#community)
- [License](#license)

---

<a id="english"></a>
## English

### Overview

AstrAI is an end-to-end Transformer framework for building, training, evaluating, and serving models. It provides a compact PyTorch codebase for the complete model lifecycle, from declarative data preprocessing and distributed training to continuous-batching inference and OpenAI/Anthropic-compatible APIs.

| Area | Capabilities |
|---|---|
| **Models** | Autoregressive language models and embedding models with GQA, MLA, MoE, RoPE, and extensible attention/FFN components |
| **Training** | Pre-training (`seq`), supervised fine-tuning (`sft`), DPO, and GRPO with gradient accumulation, checkpointing, DDP, and FSDP |
| **Data** | Declarative JSON preprocessing, configurable masking and packing, binary/JSONL storage, and streaming datasets |
| **Inference** | Continuous batching, paged KV cache, radix prefix caching, streaming generation, and Torch/CUDA/FlashAttention backends |
| **Serving** | FastAPI server with OpenAI and Anthropic chat completion protocols, including SSE streaming and tool calls |
| **Evaluation** | Perplexity, MMLU, HumanEval, IFEval, IFD, ROUGE, and weight-analysis evaluation tools |
| **Extensibility** | Factory and registry architecture for models, datasets, training strategies, callbacks, kernels, and protocol components |

### Getting Started

End-to-end walkthrough in 5 steps:

**1. Install**

AstrAI requires Python 3.12+ and pins PyTorch exactly to `2.11.0`. Training, `scripts/tools/generate.py`, generation evaluations, and the generation demos require CUDA; CPU support is limited to components with an explicit CPU device path, such as the HTTP server and direct-scoring evaluations.

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .                                          # kernels auto-build when nvcc + CUDA are detected
# CSRC_KERNELS=false pip install -e .                     # skip kernels (pure PyTorch)
# CSRC_KERNELS=true pip install -e . --no-build-isolation  # force the fused CUDA kernel build
# pip install -e ".[dev]"                                  # dev dependencies (pytest, ruff)
```

**2. Download model**

```bash
python scripts/demo/download.py    # downloads 1B checkpoint to params/
```

**3. Preprocess data**

Create `pretrain.json` (preprocessing config for `seq` strategy):

```json
{
    "version": 1,
    "input": {"sections": [{"field": "text", "action": "train"}]},
    "preprocessing": {"max_seq_len": 2048},
    "output": {"storage_format": "bin"}
}
```

```bash
python scripts/tools/preprocess.py data/*.jsonl -o output/ -c pretrain.json
```

**4. Train**

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup python scripts/tools/train.py \
    --dp_size=4 \
    --dp_mode=ddp \
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

**5. Serve & query**

```bash
# Terminal 1: start server
python scripts/tools/server.py --param_path ./params --device cuda

# Terminal 2: query
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'
```

### Demo

Check out the demos in the `scripts/demo/` folder:

```bash
# Download model weights (required before running demos)
python scripts/demo/download.py                      # model → params/

# Single-turn interactive streaming prompt loop (no conversation history)
python scripts/demo/stream_chat.py
# Type your message after >>, type !exit to quit

# Batch generation (5 hardcoded prompts, non-streaming)
python scripts/demo/generate_batch.py

# Single-prompt autoregressive streaming
python scripts/demo/generate_ar.py
```

All generation demos use `temperature=0.8`, `top_p=0.95`, `top_k=50`, `max_tokens=2048` by default and require `params/` to contain model weights (run `download.py` first).

Watch a video walkthrough on [bilibili](https://www.bilibili.com/video/BV1fuLB6yEj6).

---

See [Documentation](#documentation) for full references beyond the examples above.

#### Text Generation

Batch generation from a JSONL file:

```bash
python scripts/tools/generate.py \
    --param_path ./params \
    --input_json_file input.jsonl \
    --output_json_file output.jsonl
```

#### Docker

Build and run with Docker (recommended for GPU environments):

```bash
# Build image
docker build -t astrai:latest .

# Run with GPU support
docker run --gpus all -it astrai:latest

# Run inference server
docker run --gpus all -p 8000:8000 astrai:latest \
  python -m scripts.tools.server --port 8000 --device cuda

# Run with volume mount for data
docker run --gpus all -v /path/to/data:/data -it astrai:latest

# Docker Compose (GPU, default)
docker compose up -d

# Docker Compose CPU server profile (CUDA-only generation scripts/demos are unavailable)
docker compose --profile cpu up -d

# YAML-driven serving (see serve.yaml; up/run/down/logs/status...)
bash scripts/serve.sh up
```

> **Note**: `--gpus all` is required for CUDA support. Without it, `torch.cuda.is_available()` will return `False`.

#### HTTP API Examples

Additional request examples beyond the [Getting Started](#getting-started) flow:

```bash
# OpenAI-compatible streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell a story"}],"stream":true,"max_tokens":500}'

# Anthropic-compatible
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"astrai","system":"You are a helpful assistant.","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'

# Anthropic-compatible streaming with stop sequences
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"astrai","messages":[{"role":"user","content":"Write a story"}],"max_tokens":500,"stream":true,"stop_sequences":["The end"]}'

# Health check
curl http://localhost:8000/health
```

See [Inference Guide](docs/guides/inference.md) for SSE streaming format, error codes, and stats endpoint.

### Documentation

| Document | Description |
|----------|-------------|
| [Get Started](./docs/get-started.md) | Installation and quickstart |
| [CLI Reference](./docs/guides/params.md) | Parameters for all CLI tools (train, server, generate, preprocess) |
| [Preprocessing](./docs/guides/preprocessing.md) | Declarative JSON-driven data preprocessing |
| [Training](./docs/guides/training.md) | Training loop, strategies & formulas |
| [Inference](./docs/guides/inference.md) | KVCache, continuous batching, sampling & HTTP API |
| [Evaluation](./docs/guides/evaluation.md) | HumanEval, MMLU, PPL, ROUGE, IFD, IFEval |
| [Distributed](./docs/guides/distributed.md) | Multi-GPU DDP / FSDP training |
| [Architecture](./docs/developer/architecture.md) | System architecture, class diagram & design patterns |
| [Data Flow](./docs/developer/dataflow.md) | Data pipeline, storage backends & dataset architecture |
| [Internals](./docs/developer/internals.md) | Training internals: loss formulas, callback lifecycle, KV cache |
| [CUDA Kernels](./docs/developer/cuda_kernels.md) | Custom CUDA attention kernels & benchmarks |
| [Docker Serving](./docs/developer/docker-serving.md) | YAML-driven containerized serving (`serve.yaml`, `serve.sh`) |
| [Docker Training](./docs/developer/docker-training.md) | YAML-driven containerized training (`train.yaml`, `train.sh`) |

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

For major changes, please open an issue first to discuss what you would like to change.

### Community

- **GitHub Issues**: [Issue Tracker](https://github.com/ViperEkura/AstrAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ViperEkura/AstrAI/discussions)
- **HuggingFace**: [Model Hub](https://huggingface.co/ViperEkura)

### License

This project is licensed under the [Apache-2.0 License](LICENSE).

---

<div align="center">
  <em>A lightweight Transformer framework designed for both high performance and ease of use.</em>
</div>
