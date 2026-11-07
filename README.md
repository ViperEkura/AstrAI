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

<a id="english"></a>

## Overview

AstrAI is a PyTorch framework for preparing data, training models, evaluating
checkpoints and serving inference. It brings these steps into one codebase, with
CLI tools for running experiments and Python components for custom workflows.

Use it to train on your own data, compare training stages, run streaming generation,
or develop training objectives and execution backends. The
[Quick Start](docs/quick-start.md) walks through the full workflow.

## Features

| Area | What you can do |
| --- | --- |
| **Data preparation** | Tokenize JSON/JSONL data, configure role-based loss masks, pack sequences and write binary datasets with explicit record boundaries. |
| **Training** | Run pretraining, SFT, DPO and online GRPO/PPO; accumulate gradients, validate during training and resume from committed checkpoints. |
| **Optimization and parallelism** | Configure compute, gradient and optimizer-state precision; use CPU optimizer-state offload, DDP/FSDP and supported CP/TP layouts. |
| **Inference** | Generate synchronously or asynchronously, stream text, batch requests continuously and optionally use paged KV allocation with prefix reuse. |
| **Serving** | Expose OpenAI and Anthropic chat endpoints with streaming responses, or call the Python inference engine directly. |
| **Evaluation** | Compare checkpoints with MMLU, HumanEval, MBPP, IFEval and other evaluation tools, including perplexity and weight analysis. |

Training objectives, datasets, optimizers and schedulers can be selected through
configuration or extended in Python. Attention and other operations have selectable
Torch and accelerated backends; see [architecture](docs/developer/architecture.md)
for module boundaries and extension points.

## Typical workflows

- **Train on your data:** [preprocess](docs/guides/preprocessing.md) raw records,
  [train or fine-tune](docs/guides/training.md), then [evaluate](docs/guides/evaluation.md)
  saved checkpoints.
- **Run a checkpoint:** use the demo below for interactive input, or start an
  [HTTP service](docs/guides/inference.md) for application requests.
- **Modify the framework:** follow the [module design](docs/developer/architecture.md)
  and [testing guide](docs/developer/testing.md) when changing an objective or backend.

## Run

Requires Python 3.12+ and PyTorch 2.13.0. The demo requires CUDA.

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .
python scripts/demo/download.py
python scripts/demo/stream_chat.py
```

The download is saved to `params/`. Each prompt is independent; press Ctrl+C
or Ctrl+D to exit. To use another local checkpoint, pass `--model_path /path/to/checkpoint`.

```bash
# Raw continuation (omit --prompt for interactive input)
python scripts/demo/stream_chat.py --raw --prompt "Once upon a time"

# HTTP server
python scripts/tools/server.py --param_path ./params --device cuda
```

The optional CUDA kernels build when CUDA and nvcc are detected.
Use `CSRC_KERNELS=false pip install -e .` to skip the extension.
See [kernel build instructions](docs/developer/kernels/README.md) for details.

## Documentation

| Task | Guide |
| --- | --- |
| Install and run the full workflow | [Quick Start](docs/quick-start.md) |
| Prepare training data | [Preprocessing](docs/guides/preprocessing.md) |
| Train, resume and configure precision | [Training](docs/guides/training.md) |
| Run on multiple GPUs | [Distributed training](docs/guides/distributed.md) |
| Generate text and serve HTTP requests | [Inference](docs/guides/inference.md) |
| Evaluate checkpoints | [Evaluation](docs/guides/evaluation.md) |
| Look up CLI options | [CLI reference](docs/guides/params.md) |
| Understand subsystem structure and responsibilities | [Architecture](docs/developer/architecture.md) |
| Work with tokenizer and checkpoint files | [HF file conventions](docs/developer/huggingface.md) |
| Run checks | [Testing](docs/developer/testing.md) |
| Deploy with Docker | [Serving](docs/developer/docker/serving.md), [training](docs/developer/docker/training.md) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Discuss major changes in an
[issue](https://github.com/ViperEkura/AstrAI/issues) before opening a pull request.
Questions and examples belong in [Discussions](https://github.com/ViperEkura/AstrAI/discussions).

## License

[Apache-2.0](LICENSE).
