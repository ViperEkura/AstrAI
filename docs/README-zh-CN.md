<div align="center">

  <img src="./images/logo.png" width="auto" alt="Logo">

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

<a id="chinese"></a>

## 概述

AstrAI 是基于 PyTorch 的数据处理、模型训练、检查点评测和推理服务框架。
这些流程共用一套代码：可以通过命令行运行实验，也可以使用 Python 组件组织自己的工作流。

你可以用它训练自己的数据、对比不同训练阶段、运行流式生成，或开发新的训练目标和执行后端。
完整流程见[快速上手](quick-start.md)。

## 主要功能

| 领域 | 可以做什么 |
| --- | --- |
| **数据处理** | 对 JSON/JSONL 数据分词，配置按角色监督的 loss mask，打包序列，并保存带有明确记录边界的二进制数据集。 |
| **训练** | 运行预训练、SFT、DPO 和在线 GRPO/PPO，支持梯度累积、训练中验证，以及从已提交状态的检查点恢复。 |
| **优化与并行** | 分别配置计算、梯度和优化器状态精度，使用优化器状态 CPU 卸载、DDP/FSDP，以及受支持的 CP/TP 并行配置。 |
| **推理** | 同步或异步生成、流式输出、连续批处理，并可选用分页 KV 分配与前缀复用。 |
| **服务** | 提供 OpenAI 和 Anthropic 聊天接口及流式响应，也可以直接调用 Python 推理引擎。 |
| **评测** | 使用 MMLU、HumanEval、MBPP、IFEval 等工具对比检查点，也可计算困惑度和分析权重。 |

训练目标、数据集、优化器和学习率调度器可以通过配置选择，也可以在 Python 中扩展。
Attention 等算子提供可选择的 Torch 和加速后端；模块边界与扩展入口见[架构说明](developer/architecture.md)。

## 典型流程

- **训练自己的数据：** [预处理](guides/preprocessing.md)原始记录，
  [训练或微调](guides/training.md)，再[评测](guides/evaluation.md)保存的检查点。
- **运行检查点：** 使用下方演示进行交互输入，或启动 [HTTP 服务](guides/inference.md)处理应用请求。
- **修改框架：** 扩展训练目标或后端时，参考[模块设计](developer/architecture.md)和[测试指南](developer/testing.md)。

## 运行

需要 Python 3.12+ 和 PyTorch 2.13.0；演示脚本需要 CUDA。

```bash
git clone https://github.com/ViperEkura/AstrAI.git
cd AstrAI
pip install -e .
python scripts/demo/download.py
python scripts/demo/stream_chat.py
```

下载文件保存在 `params/`。每次提问独立处理，按 Ctrl+C 或 Ctrl+D 退出。
使用其他本地检查点时，传入 `--model_path /path/to/checkpoint`。

```bash
# 原始文本续写（省略 --prompt 则交互输入）
python scripts/demo/stream_chat.py --raw --prompt "很久很久以前"

# HTTP 服务
python scripts/tools/server.py --param_path ./params --device cuda
```

检测到 CUDA 和 nvcc 时会构建可选 CUDA 扩展。
使用 `CSRC_KERNELS=false pip install -e .` 跳过扩展，
详见[内核构建说明](developer/kernels/README.md)。

## 文档

| 任务 | 指南 |
| --- | --- |
| 安装并运行完整流程 | [快速上手](quick-start.md) |
| 准备训练数据 | [预处理](guides/preprocessing.md) |
| 训练、恢复与精度配置 | [训练](guides/training.md) |
| 多 GPU 训练 | [分布式训练](guides/distributed.md) |
| 生成文本与提供 HTTP 服务 | [推理](guides/inference.md) |
| 评测检查点 | [评测](guides/evaluation.md) |
| 查询命令行参数 | [CLI 参数](guides/params.md) |
| 理解模块结构与职责 | [架构](developer/architecture.md) |
| 使用 tokenizer 和检查点文件 | [HF 文件约定](developer/huggingface.md) |
| 运行检查 | [测试](developer/testing.md) |
| Docker 部署 | [推理服务](developer/docker/serving.md)、[训练](developer/docker/training.md) |

## 贡献

请阅读 [CONTRIBUTING.md](../CONTRIBUTING.md)。
较大改动请先提交 [Issue](https://github.com/ViperEkura/AstrAI/issues)，再发起 PR。
使用问题和示例可发布到 [Discussions](https://github.com/ViperEkura/AstrAI/discussions)。

## 许可证

[Apache-2.0](../LICENSE)。
