<div align="center">
  
  <img src="assets/images/logo.png" width="auto" alt="Logo">
  <p>
    <strong>A lightweight Transformer training & inference framework</strong>
  </p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-GPL--3.0-blue.svg" alt="license">
  <img src="https://img.shields.io/github/v/tag/ViperEkura/AstrAI?label=Release&color=76bad9" alt="release">
  <img src="https://img.shields.io/github/stars/ViperEkura/AstrAI?style=flat&label=Stars&color=76bad9" alt="stars">
  <img src="https://img.shields.io/github/forks/ViperEkura/AstrAI?style=flat&label=Forks&color=76bad9" alt="forks">
</div>
<br>

<div align="center">
  <a href="#english">English</a> •
  <a href="assets/docs/README-zh-CN.md">中文</a> •
  <a href="https://github.com/ViperEkura/AstrAI/issues">Issue Tracker</a> •
  <a href="https://github.com/ViperEkura/AstrAI/discussions">Discussions</a> •
  <a href="https://huggingface.co/ViperEk/">HuggingFace</a>
</div>

<br>

## 📖 Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
- [Demo](#demo)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Community](#community)
- [License](#license)

---

<a id="english"></a>
## English

### Features

- 🚀 **High Performance**: Optimized for both training and inference with efficient parallelization.
- 🔧 **Flexible**: Support for seq/sft/dpo/grpo training, customizable model architectures.
- 💡 **Easy to Use**: Simple API with comprehensive examples and demos.
- 📦 **Lightweight**: Minimal dependencies, easy to deploy.
- 🔬 **Research‑Friendly**: Modular design, easy to experiment with new ideas.
- 🤗 **HuggingFace-Style API**: AutoModel/AutoTokenizer APIs inspired by HuggingFace for easy model and tokenizer loading.
- 🔌 **Dual API Compatibility**: Supports both OpenAI and Anthropic chat completion APIs out of the box.

### Getting Started

End-to-end walkthrough in 5 steps:

**1. Install**

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .                                          # pure PyTorch (no CUDA kernels)
# CSRC_KERNELS=true pip install -e . --no-build-isolation  # optional: fused CUDA kernels
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

# Interactive streaming chat (multi-turn, maintains history)
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

# Docker Compose (CPU only)
docker compose --profile cpu up -d
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

See [Inference Guide](assets/docs/inference.md) for SSE streaming format, error codes, and stats endpoint.

### Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](./assets/docs/params.md) | Parameters for all CLI tools (train, server, generate, preprocess) |
| [Architecture](./assets/docs/architecture.md) | System architecture, class diagram & design patterns |
| [Training](./assets/docs/training.md) | Training loop, strategies & formulas |
| [Inference](./assets/docs/inference.md) | KVCache, continuous batching, sampling & HTTP API |
| [Data Flow](./assets/docs/dataflow.md) | Data pipeline, storage backends & dataset architecture |
| [Preprocessing](./assets/docs/preprocessing.md) | Declarative JSON-driven data preprocessing |

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
- **HuggingFace**: [Model Hub](https://huggingface.co/ViperEk)

### License

This project is licensed under the [GPL-3.0 License](LICENSE).

---

<div align="center">
  <em>A lightweight Transformer framework designed for both high performance and ease of use.</em>
</div>