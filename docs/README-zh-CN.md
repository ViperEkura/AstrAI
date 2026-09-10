<div align="center">
  
  <img src="./images/logo.png" width="auto" alt="Logo">
  
  <div>
    <a href="../README.md">English</a> • 
    <a href="#chinese">中文</a>
  </div>
  
  <p>
    <strong>轻量级 Transformer 训练与推理框架</strong>
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
  <a href="../README.md">English</a> •
  <a href="#chinese">中文</a> •
  <a href="https://github.com/ViperEkura/AstrAI/issues">问题追踪</a> •
  <a href="https://github.com/ViperEkura/AstrAI/discussions">讨论区</a> •
  <a href="https://huggingface.co/ViperEkura">HuggingFace</a>
</div>
<br>

## 📖 目录

- [项目概览](#项目概览)
- [快速上手](#快速上手)
- [演示](#演示)
- [文档](#文档)
- [贡献](#贡献)
- [社区](#社区)
- [许可证](#许可证)

---

<a id="chinese"></a>
## 中文

### 项目概览

AstrAI 是一个覆盖模型构建、训练、评测与部署的端到端 Transformer 框架。项目以精简的 PyTorch 代码实现完整模型生命周期，包括声明式数据预处理、分布式训练、连续批处理推理，以及兼容 OpenAI 和 Anthropic 的服务接口。

| 领域 | 能力 |
|---|---|
| **模型** | 自回归语言模型与嵌入模型，支持 GQA、MLA、MoE、RoPE，以及可扩展的 Attention/FFN 组件 |
| **训练** | 预训练（`seq`）、监督微调（`sft`）、DPO 和 GRPO，支持梯度累积、检查点、DDP 与 FSDP |
| **数据** | 声明式 JSON 预处理、可配置掩码与样本打包、二进制/JSONL 存储和流式数据集 |
| **推理** | 连续批处理、分页 KV Cache、Radix 前缀缓存、流式生成，以及 Torch/CUDA/FlashAttention 后端 |
| **服务** | 基于 FastAPI 的 OpenAI 与 Anthropic 聊天补全协议，支持 SSE 流式输出和工具调用 |
| **评测** | Perplexity、MMLU、HumanEval、IFEval、IFD、ROUGE 和权重分析评测工具 |
| **扩展** | 基于工厂与注册表扩展模型、数据集、训练策略、回调、内核和协议组件 |

### 快速上手

端到端演示，只需 5 步：

**1. 安装**

AstrAI 需要 Python 3.12+，并精确固定 PyTorch 版本为 `2.11.0`。训练、`scripts/tools/generate.py`、生成式评估和生成演示需要 CUDA；CPU 支持仅适用于提供明确 CPU 设备路径的组件，例如 HTTP 服务和直接打分评估。

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .                                          # 检测到 nvcc + CUDA 时自动构建内核
# CSRC_KERNELS=false pip install -e .                     # 跳过内核（纯 PyTorch）
# CSRC_KERNELS=true pip install -e . --no-build-isolation  # 强制构建融合 CUDA 内核
# pip install -e ".[dev]"                                  # 可选：开发依赖（pytest, ruff）
```

**2. 下载模型**

```bash
python scripts/demo/download.py    # 下载 1B 检查点到 params/
```

**3. 预处理数据**

创建 `pretrain.json`（`seq` 策略的预处理配置）：

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

**4. 训练**

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

**5. 启动服务并调用**

```bash
# 终端 1：启动服务
python scripts/tools/server.py --param_path ./params --device cuda

# 终端 2：发起请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":512}'
```

### 演示

查看 `scripts/demo/` 文件夹中的演示：

```bash
# 下载模型权重（运行演示前必需）
python scripts/demo/download.py                      # model → params/

# 单轮交互式流式提示循环（不保留对话历史）
python scripts/demo/stream_chat.py
# 在 >> 后输入消息，输入 !exit 退出

# 批量生成（5 条硬编码提示词，非流式）
python scripts/demo/generate_batch.py

# 单条提示词自回归流式生成
python scripts/demo/generate_ar.py
```

所有生成演示默认使用 `temperature=0.8`、`top_p=0.95`、`top_k=50`、`max_tokens=2048`，需要 `params/` 目录包含模型权重（请先运行 `download.py`）。

观看 [bilibili](https://www.bilibili.com/video/BV1fuLB6yEj6) 上的视频演示。

---

更多选项请参考[文档](#文档)。

#### 文本生成

从 JSONL 文件批量生成：

```bash
python scripts/tools/generate.py \
    --param_path ./params \
    --input_json_file input.jsonl \
    --output_json_file output.jsonl
```

#### Docker

使用 Docker 构建和运行（推荐用于 GPU 环境）：

```bash
# 构建镜像
docker build -t astrai:latest .

# 启用 GPU 运行
docker run --gpus all -it astrai:latest

# 运行推理服务
docker run --gpus all -p 8000:8000 astrai:latest \
  python scripts/tools/server.py --port 8000 --device cuda

# 挂载数据卷
docker run --gpus all -v /path/to/data:/data -it astrai:latest

# Docker Compose（GPU，默认）
docker compose up -d

# Docker Compose CPU 服务配置（不支持仅限 CUDA 的生成脚本和演示）
docker compose --profile cpu up -d
```

> **注意**: 必须使用 `--gpus all` 才能启用 CUDA 支持，否则 `torch.cuda.is_available()` 将返回 `False`。

#### HTTP API 示例

除[快速上手](#快速上手)流程外，更多请求示例：

```bash
# OpenAI 兼容流式
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"讲个故事"}],"stream":true,"max_tokens":500}'

# Anthropic 兼容
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"astrai","system":"你是一个乐于助人的助手。","messages":[{"role":"user","content":"你好"}],"max_tokens":512}'

# Anthropic 兼容流式并设置停止序列
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"astrai","messages":[{"role":"user","content":"写个故事"}],"max_tokens":500,"stream":true,"stop_sequences":["结束"]}'

# 健康检查
curl http://localhost:8000/health
```

SSE 流式格式、错误码和统计端点详见[推理文档](guides/inference.md)。

### 文档

| 文档 | 说明 |
|------|------|
| [快速上手](./get-started.md) | 安装与快速入门 |
| [CLI 参考](./guides/params.md) | 所有 CLI 工具参数（训练、服务、生成、预处理） |
| [数据预处理](./guides/preprocessing.md) | 声明式 JSON 驱动数据预处理 |
| [训练文档](./guides/training.md) | 训练循环、策略与公式 |
| [推理文档](./guides/inference.md) | KVCache、连续批处理、采样与 HTTP API |
| [评估文档](./guides/evaluation.md) | HumanEval、MMLU、PPL、ROUGE、IFD、IFEval |
| [分布式训练](./guides/distributed.md) | 多卡 DDP / FSDP 训练 |
| [架构文档](./developer/architecture.md) | 系统架构、类图与设计模式 |
| [数据流程](./developer/dataflow.md) | 数据管道、存储后端与数据集架构 |
| [内部实现](./developer/internals.md) | 训练原理：损失公式、回调生命周期、KV Cache |
| [CUDA 内核](./developer/cuda_kernels.md) | 自定义 CUDA 注意力内核与基准测试 |
| [Docker 服务部署](./developer/docker-serving.md) | YAML 驱动的容器化服务（`serve.yaml`、`serve.sh`） |
| [Docker 训练部署](./developer/docker-training.md) | YAML 驱动的容器化训练（`train.yaml`、`train.sh`） |

### 贡献

我们欢迎贡献！请参阅[贡献指南](../CONTRIBUTING.md)了解详情。

1. Fork 本仓库。
2. 创建功能分支。
3. 提交更改。
4. 发起 Pull Request。

重大更改请先开 issue 讨论。

### 社区

- **GitHub Issues**: [问题追踪](https://github.com/ViperEkura/AstrAI/issues)
- **Discussions**: [GitHub 讨论区](https://github.com/ViperEkura/AstrAI/discussions)
- **HuggingFace**: [模型中心](https://huggingface.co/ViperEkura)

### 许可证

本项目采用 [Apache-2.0 许可证](../LICENSE)。

---

<div align="center">
  <em>专为高性能与易用性设计的轻量级 Transformer 框架。</em>
</div>
