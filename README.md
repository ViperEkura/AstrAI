<div align="center">
  
  <img src="assets/images/logo.png" width="auto" alt="Logo">
  <p>
    <strong>A lightweight Transformer training & inference framework</strong>
  </p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/license-GPL--3.0-blue.svg" alt="license">
  <img src="https://img.shields.io/github/v/release/ViperEkura/AstrAI?label=Release&color=76bad9" alt="release">
  <img src="https://img.shields.io/github/stars/ViperEkura/AstrAI?style=flat&label=Stars&color=76bad9" alt="stars">
  <img src="https://img.shields.io/github/forks/ViperEkura/AstrAI?style=flat&label=Forks&color=76bad9" alt="forks">
  <img src="https://img.shields.io/github/actions/workflow/status/ViperEkura/AstrAI/tests.yml?label=CI&color=76bad9" alt="ci">
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
- [Quick Start](#quick-start)
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

### Quick Start

#### Installation

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

#### Download Pre-trained Model

Download pre-trained model weights (1B bilingual checkpoint) to `params/`:

```bash
python scripts/demo/download.py
```

Or download manually from [HuggingFace](https://huggingface.co/ViperEk/KHAOSZ) into `params/`.

#### Train a Model

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

Full reference at [Parameter Guide](assets/docs/params.md).

#### Generate Text

```bash
python scripts/tools/generate.py \
    --param_path /path/to/model \
    --input_json_file /path/to/input.jsonl \
    --output_json_file /path/to/output.jsonl
```

#### Docker

Build and run with Docker (recommended for GPU environments):

```bash
# Build image
docker build -t astrai:latest .

# Run with GPU support
docker run --gpus all -it astrai:latest

# Run with specific GPUs
docker run --gpus '"device=0,1"' -it astrai:latest

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

#### Start HTTP Server

Start the inference server with OpenAI and Anthropic-compatible HTTP API:

```bash
python -m scripts.tools.server --port 8000 --device cuda
```

Make requests:

```bash
# OpenAI-compatible
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 512
  }'

# OpenAI-compatible streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell a story"}],
    "stream": true,
    "max_tokens": 500
  }'

# Anthropic-compatible
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "astrai",
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 512
  }'

# Anthropic-compatible streaming with stop sequences
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "astrai",
    "messages": [{"role": "user", "content": "Write a story"}],
    "max_tokens": 500,
    "stream": true,
    "stop_sequences": ["The end"]
  }'

# Health check
curl http://localhost:8000/health
```

#### Demo

Check out the demos in the `scripts/demo/` folder:

```bash
# Download model weights (required before running demos)
python scripts/demo/download.py

# Interactive streaming chat
python scripts/demo/stream_chat.py

# Batch generation
python scripts/demo/generate_batch.py

# Auto‑regressive generation
python scripts/demo/generate_ar.py
```

Watch a video walkthrough on [bilibili](https://www.bilibili.com/video/BV1fuLB6yEj6).

### Documentation

| Document | Description |
|----------|-------------|
| [Parameter Guide](./assets/docs/params.md) | Training & inference parameters |
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