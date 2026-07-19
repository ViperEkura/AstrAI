<div align="center">
  
  <img src="../images/logo.png" width="auto" alt="Logo">
  
  <div>
    <a href="../../README.md">English</a> • 
    <a href="#chinese">中文</a>
  </div>
  
  <p>
    <strong>轻量级 Transformer 训练与推理框架</strong>
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
  <a href="../../README.md">English</a> •
  <a href="#chinese">中文</a> •
  <a href="https://github.com/ViperEkura/AstrAI/issues">问题追踪</a> •
  <a href="https://github.com/ViperEkura/AstrAI/discussions">讨论区</a> •
  <a href="https://huggingface.co/ViperEkura">HuggingFace</a>
</div>
<br>

## 📖 目录

- [特性](#特性)
- [快速上手](#快速上手)
- [演示](#演示)
- [文档](#文档)
- [贡献](#贡献)
- [社区](#社区)
- [许可证](#许可证)

---

<a id="chinese"></a>
## 中文

### 特性

- 🚀 **高性能**: 训练与推理双向优化，高效并行。
- 🔧 **灵活**: 支持 seq/sft/dpo/grpo 多种训练方式，可定制模型架构。
- 💡 **易用**: 简洁的 API 与丰富的示例、演示。
- 📦 **轻量**: 依赖少，部署简单。
- 🔬 **研究友好**: 模块化设计，便于实验新想法。
- 🤗 **HuggingFace 风格 API**: 类 HuggingFace 的 AutoModel/AutoTokenizer 接口，方便加载模型和分词器。
- 🔌 **双 API 兼容**: 同时支持 OpenAI 和 Anthropic 聊天补全 API，开箱即用。

### 快速上手

端到端演示，只需 5 步：

**1. 安装**

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .                                          # 纯 PyTorch（不含 CUDA 内核）
# CSRC_KERNELS=true pip install -e . --no-build-isolation  # 可选：融合 CUDA 内核加速
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

# 交互式流式聊天（多轮对话，保持历史记录）
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
  python -m scripts.tools.server --port 8000 --device cuda

# 挂载数据卷
docker run --gpus all -v /path/to/data:/data -it astrai:latest

# Docker Compose（GPU，默认）
docker compose up -d

# Docker Compose（仅 CPU）
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

SSE 流式格式、错误码和统计端点详见[推理文档](./inference.md)。

### 文档

| 文档 | 说明 |
|------|------|
| [CLI 参考](./params.md) | 所有 CLI 工具参数（训练、服务、生成、预处理） |
| [架构文档](./architecture.md) | 系统架构、类图与设计模式 |
| [训练文档](./training.md) | 训练循环、策略与公式 |
| [推理文档](./inference.md) | KVCache、连续批处理、采样与 HTTP API |
| [数据流程](./dataflow.md) | 数据管道、存储后端与数据集架构 |
| [数据预处理](./preprocessing.md) | 声明式 JSON 驱动数据预处理 |

### 贡献

我们欢迎贡献！请参阅[贡献指南](../../CONTRIBUTING.md)了解详情。

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

本项目采用 [GPL-3.0 许可证](../../LICENSE)。

---

<div align="center">
  <em>专为高性能与易用性设计的轻量级 Transformer 框架。</em>
</div>