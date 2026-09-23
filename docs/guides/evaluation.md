# Evaluation

AstrAI provides 9 evaluation scripts in `scripts/eval/` covering code generation, knowledge QA, commonsense QA, perplexity, summarization, data quality, instruction following, and weight analysis, plus suite tooling (`run_suite.py`, `collect.py`) built on the shared `astrai.bench` module.

## Contents

- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [HumanEval](#humaneval-code-generation)
- [MBPP](#mbpp-code-generation)
- [MMLU](#mmlu-knowledge-qa)
- [HellaSwag](#hellaswag-commonsense-qa)
- [Perplexity](#perplexity-ppl)
- [ROUGE](#rouge)
- [IFD](#ifd-instruction-following-difficulty)
- [IFEval](#ifeval-instruction-following)
- [Weight Analysis](#weight-analysis)
- [Suite runner](#suite-runner)
- [Online evaluation](#online-evaluation-checkpoint-watcher)
- [Collecting results](#collecting-results)
- [Tips](#tips)

## Prerequisites

HumanEval, MMLU, and IFEval import HuggingFace `datasets` to download their benchmark data. This package is not installed by AstrAI's base dependencies, so install it before running those scripts:

```bash
pip install datasets
```

The generation-based scripts require CUDA because they load the model on `cuda` with `bfloat16`. Direct-scoring and metric scripts support the devices shown below.

## Overview

| Script | Metric | Model Invocation | External Dataset |
|--------|--------|-------------------|-------------------|
| `evaluate_humaneval.py` | Code-gen pass@1/10/100 | `InferenceEngine.generate` | HF `openai/openai_humaneval` (auto-download) |
| `evaluate_mbpp.py` | Code-gen pass@1/10 (chat zero-shot) | `InferenceEngine.generate` | HF `google-research-datasets/mbpp` (auto-download) |
| `evaluate_mmlu.py` | MCQ accuracy (log-likelihood) | Direct `model()` forward | HF `cais/mmlu` (auto-download) |
| `evaluate_hellaswag.py` | Commonsense acc / acc_norm (log-likelihood) | Direct `model()` forward | HF `Rowan/hellaswag` (auto-download) |
| `evaluate_ppl.py` | Perplexity / token loss | Direct `model()` forward | User JSONL |
| `evaluate_rouge.py` | ROUGE-1/2/L | None (pure metric) | User JSONL |
| `evaluate_ifd.py` | Instruction-Following Difficulty | Direct `model()` forward | User JSONL |
| `evaluate_ifeval.py` | Instruction-following constraints | `InferenceEngine.generate` | HF `google/IFEval` (auto-download) |
| `analyze_weights.py` | SVD effective rank / weight stats | None (loads safetensors) | Checkpoint dir |
| `run_suite.py` | Standard suite per checkpoint, with retries | Launches the above | Local data caches |
| `collect.py` | Markdown comparison table from results JSONs | None | `results/` directory |

Two invocation patterns exist:
- **Generation benchmarks** (HumanEval, IFEval): use `InferenceEngine` to generate responses, then score them.
- **Scoring benchmarks** (MMLU, PPL, IFD): call `model()` directly under `torch.inference_mode()` for log-likelihood computation.

| Script | Device support |
|--------|----------------|
| HumanEval | CUDA for generation; `--test_only` can score existing completions without loading a model |
| IFEval | CUDA only |
| MMLU | CUDA or CPU via `--device`; auto-selects CUDA when available |
| PPL | CUDA or CPU via `--device`; auto-selects CUDA when available |
| IFD | CUDA or CPU via `--device`; auto-selects CUDA when available |
| ROUGE | CPU-only metric computation; no model is loaded |
| Weight analysis | CUDA by default; CPU supported via `--device cpu` |

---

## HumanEval (Code Generation)

Generates completions for 164 programming problems, executes them against hidden tests, and reports pass@k.

```bash
python scripts/eval/evaluate_humaneval.py \
    --param_path ./params \
    --num_samples 20 \
    --batch_size 64 \
    --max_tokens 512 \
    --output results/humaneval.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | `./params` | Model directory |
| `--data_path` | `./humaneval/HumanEval.jsonl` | HumanEval JSONL (auto-downloaded if missing) |
| `--output` | None | Save results JSON (also writes `_completions.json`) |
| `--test_only` | None | Test an existing completions JSON (skip generation) |
| `--generate_only` | False | Only generate, skip execution/testing |
| `--num_samples` | 200 | Completions per problem (pass@k needs >= k) |
| `--max_tokens` | 512 | Max generation length |
| `--temperature` | 0.8 | Sampling temperature |
| `--top_p` | 0.95 | Nucleus sampling threshold |
| `--top_k` | 50 | Top-k sampling |
| `--batch_size` | 64 | Generation batch size |
| `--max_seq_len` | 4096 | KV cache sequence length |
| `--test_workers` | 8 | ProcessPoolExecutor workers for test execution |
| `--test_timeout` | 3.0 | Per-subprocess timeout (seconds) |
| `--problems` | None | Restrict to specific problem indices |

**Output**: stdout prints `pass@1`, `pass@10`, `pass@100`. With `--output`, writes per-problem results + `_summary` aggregate and a `_completions.json` file.

**Data**: Auto-downloads `openai/openai_humaneval` from HuggingFace on first run. Each problem has `task_id`, `entry_point`, `prompt`, `test`.

---

## MBPP (Code Generation)

Mostly Basic Python Problems, full test split (500 tasks, ids 11-510), chat zero-shot: task text + asserts + the canonical function signature go through the chat template. Raw few-shot completion transcripts derail SFT models, and without the signature ~65% of failures are invented function names — so absolute numbers are not comparable to published few-shot MBPP; cross-checkpoint comparisons are the payload.

```bash
python scripts/eval/evaluate_mbpp.py \
    --param_path ./params \
    --num_samples 20 \
    --batch_size 64 \
    --output results/mbpp.json
```

Key parameters mirror HumanEval (`--num_samples`, `--temperature`, `--batch_size`, `--test_only`, `--problems`); data auto-downloads to `mbpp/mbpp_test.jsonl` on first run. Execution = completion + `test_list` asserts (challenge tests excluded).

**Output**: stdout prints `pass@1`/`pass@10`; with `--output`, per-problem results + `_summary` and a `_completions.json` file.

---

## MMLU (Knowledge QA)

57-subject multiple-choice accuracy via log-likelihood comparison. Supports n-shot few-shot prompting and option permutation.

```bash
python scripts/eval/evaluate_mmlu.py \
    --param_path ./params \
    --n_shot 5 \
    --subjects abstract_algebra high_school_us_history \
    --output results/mmlu.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | `./params` | Model directory |
| `--data_dir` | `./mmlu_data` | MMLU data directory (per-subject CSVs) |
| `--download` | False | Force re-download |
| `--n_shot` | 5 | Few-shot examples (0 = zero-shot) |
| `--subjects` | all 57 | Specific subjects to evaluate |
| `--output` | None | Output JSON path |
| `--split` | `test` | `test` or `val` |
| `--device` | auto | Device (`cuda` / `cpu`) |
| `--dtype` | auto | `bfloat16` on CUDA, `float32` on CPU |
| `--seed` | 0 | Seed for option permutation (0 = enabled, -1 = disabled) |
| `--batch_size` | 4 | Questions per batch; each question produces four choice rows |

**How it works**: For each question, builds a prompt with n-shot examples, then scores each choice (A/B/C/D) by computing the summed log-likelihood of the choice token given the context. The choice with the highest log-prob is the prediction.

**Output**: stdout prints per-subject accuracy and overall. With `--output`, writes per-subject `{accuracy, correct, total}` + `_overall` aggregate.

**Data**: Auto-downloads `cais/mmlu` from HuggingFace. Stored as per-subject CSVs in `<data_dir>/<split>/` and `<data_dir>/dev/` (for few-shot). `--subjects` accepts canonical MMLU names such as `abstract_algebra`, `college_computer_science`, `high_school_us_history`, and `world_religions`.

---

## HellaSwag (Commonsense QA)

Sentence-completion MCQ over the full validation split (10,042 questions), zero-shot, following the lm-evaluation-harness protocol (same text normalization; each ending scored as a continuation). Reports raw accuracy and length-normalized `acc_norm`.

```bash
python scripts/eval/evaluate_hellaswag.py \
    --param_path ./params \
    --batch_size 16 \
    --output results/hellaswag.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | `./params` | Model directory |
| `--data_path` | `./hellaswag/val.jsonl` | HellaSwag validation JSONL (auto-downloaded if missing) |
| `--limit` | 0 | Score only the first N questions (0 = all) |
| `--batch_size` | 8 | Questions per batch (4 continuation rows each) |
| `--device` / `--dtype` | cuda / bfloat16 | Device and dtype |
| `--output` | None | Save `_summary` JSON |

**Output**: stdout prints `acc` and `acc_norm`; with `--output`, a `_summary` JSON. A full run takes ~2.5 minutes per arm on one RTX 5090 — short-sequence forward passes are cheap; prefer full runs over sampling.

**Data**: Auto-downloads `Rowan/hellaswag` (validation split) on first run.

---

## Perplexity (PPL)

Token-level negative-log-likelihood and perplexity on arbitrary text data. Supports streaming mode (memory-efficient) and non-streaming mode (exact per-token stats).

```bash
python scripts/eval/evaluate_ppl.py \
    --param_path ./params \
    --input_path data.jsonl \
    --output_dir ppl_results/ \
    --batch_size 64 \
    --max_length 2048
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | required | Model directory |
| `--input_path` | required | Input file, glob, or directory |
| `--output_dir` | required | Output directory for `summary.json` + token JSONL |
| `--text_key` | `text` | Key for the text field in input data |
| `--batch_size` | 64 | Batch size |
| `--max_length` | 2048 | Max sequence length (tokens) |
| `--token_level` | False | Store per-token log_probs + token-type analysis |
| `--max_samples` | None | Random subsample per file |
| `--device` | auto | Device |
| `--dtype` | auto | Torch dtype |

**Input**: JSONL or JSON files. Each item must have a field named by `--text_key` (default `text`). If `--input_path` is a directory, recursively collects `*.jsonl` and `*.json`.

**Output**: `summary.json` with per-file token count, mean loss, perplexity, and p50/p90/p95/p99 loss. Median loss is included only with `--token_level`; that mode also writes per-token JSONL with token IDs and log-probs.

---

## ROUGE

ROUGE-1/2/L (precision, recall, F1) for summarization. Self-contained implementation with no external dependencies.

```bash
python scripts/eval/evaluate_rouge.py \
    --data_path predictions.jsonl \
    --output results/rouge.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | required | JSONL with `reference`/`candidate` per line |
| `--output` | None | Output JSON path |

**Input**: JSONL, one object per line:
```json
{"reference": "Ground truth text", "candidate": "Model output text"}
```

**Output**: stdout prints `rouge-1`, `rouge-2`, `rouge-l` each as P/R/F1. With `--output`, writes JSON with `aggregate` and `per_item` scores.

Can also be imported as a library:
```python
from scripts.eval.evaluate_rouge import compute_rouge
scores = compute_rouge(reference, candidate)
```

---

## IFD (Instruction-Following Difficulty)

Data quality metric: `IFD = L_conditional / L_unconditional`. Measures how much harder it is to predict a response given its instruction vs. without it. Useful for filtering instruction-tuning data.

```bash
python scripts/eval/evaluate_ifd.py \
    --param_path ./params \
    --input_path sft_data.jsonl \
    --output_dir ifd_results/ \
    --format messages \
    --batch_size 8
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | required | Model directory |
| `--input_path` | required | Input file, glob, or directory |
| `--output_dir` | required | Output directory |
| `--max_len` | 2048 | Max token length |
| `--format` | `plain` | `plain` (instruction/response fields) or `messages` (chat format) |
| `--instr_key` | `instruction` | Instruction field key (plain format) |
| `--resp_key` | `response` | Response field key (plain format) |
| `--batch_size` | 8 | Items per model-forward flush |
| `--device` | auto | Device |
| `--dtype` | auto | Torch dtype |
| `--sentinel_text` | `\n` | Prefix for unconditional pass (`""` → bos/pad fallback) |
| `--per_token` | False | Include per-token IFD breakdown |
| `--max_samples` | None | Random subsample per file |
| `--append_eos` / `--no-append_eos` | `True` | Append (or skip) EOS token to instruction/response |

**How it works**: Two forward passes per batch — (1) conditional: packed BFD sequence with context + response, (2) unconditional: response prefixed with a sentinel. IFD = mean_conditional_loss / mean_unconditional_loss. IFD > 1 means the instruction makes the response harder to predict (higher quality data).

**Output**: Per-file `<label>_ifd.jsonl` with IFD scores per item. `summary.json` aggregates per-file stats.

---

## IFEval (Instruction Following)

Google's IFEval benchmark: generates responses and verifies 27 types of constraints (keywords, format, length, case, punctuation, etc.).

```bash
python scripts/eval/evaluate_ifeval.py \
    --param_path ./params \
    --num_samples 1 \
    --max_tokens 512 \
    --output results/ifeval.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--param_path` | `./params` | Model directory |
| `--data_path` | `./ifeval/input_data.jsonl` | IFEval JSONL (auto-downloaded if missing) |
| `--output` | None | Output JSON path |
| `--max_tokens` | 512 | Max generation tokens |
| `--temperature` | 0.1 | Sampling temperature (low for instruction-following) |
| `--top_p` | 0.95 | Top-p sampling |
| `--top_k` | 50 | Top-k sampling |
| `--num_samples` | 1 | Samples per problem (best-of-n scoring) |
| `--batch_size` | 64 | Inference batch size |
| `--max_seq_len` | 4096 | KV cache sequence length |
| `--limit` | None | Limit to first N problems (quick testing) |
| `--dump_responses` | None | Path to dump raw responses as JSONL |

**Output**: stdout prints overall accuracy + per-constraint-type accuracy table. With `--output`, writes per-problem results + `_summary`.

**Data**: Auto-downloads `google/IFEval` from HuggingFace. Each problem has `key`, `prompt`, `instruction_id_list`, `kwargs`.

---

## Weight Analysis

SVD-based effective rank and weight statistics for checkpoint diagnostics. Does not load the model graph or run any forward pass.

```bash
python scripts/eval/analyze_weights.py \
    --ckpt_dir ./checkpoint/epoch_1_step_6000 \
    --output results/weights.json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ckpt_dir` | required | Checkpoint directory containing `model.safetensors` |
| `--compare` | None | Additional checkpoint dirs to compare |
| `--no_svd` | False | Skip SVD; show only weight stats (faster) |
| `--output` | None | Save results as JSON |
| `--device` | `cuda` | Device for SVD |

**Output**: SVD effective rank by component (ER@90/95/99%, entropic rank, condition number), per-layer effective rank grid, and weight value statistics (mean/std/min/max). Provides a utilization verdict (HIGH >0.85 / MODERATE >0.5 / LOW).

---

## Suite runner

`run_suite.py` runs the standard benchmark set for one checkpoint, one benchmark per free GPU, with per-job retries (a job is retried when its process dies without writing the output file — idempotent, so rerunning skips completed benchmarks).

```bash
nohup python -u scripts/eval/run_suite.py \
    --ckpt checkpoints/sft-mix3/epoch_1_step_4385 \
    --tag mix3 \
    --benchmarks mmlu,ifeval,humaneval,mbpp,hellaswag \
    > logs/suite_mix3.log 2>&1 &
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ckpt` | required | Checkpoint path, passed through as `--param_path` |
| `--tag` | required | Output naming: `results/<bench>_<tag>.json` |
| `--benchmarks` | `mmlu,ifeval,humaneval` | Comma-separated subset (`mbpp`, `mbpp2`, `hellaswag` available) |
| `--gpus` | auto | Preferred GPU indices (free GPUs below 100 MiB; otherwise 4-7 first) |
| `--attempts` | 3 | Retries per benchmark |
| `--smoke` | False | Cheap per-benchmark invocations for a plumbing test (IFEval has no cheap mode and is skipped) |
| `--dry-run` | False | Print the job plan and exit |

Standard per-benchmark parameters are baked in (MMLU 5-shot bs4; IFEval ns1 bs64 temp 0.1; HumanEval/MBPP ns20 bs64; HellaSwag bs16) and match the eval reports. Wrap in `nohup`: if the runner itself dies, rerunning the same command skips finished benchmarks. A summary table prints at the end.

## Online evaluation (checkpoint watcher)

`watch_ckpts.py` polls a training `ckpt_dir` and evaluates each new checkpoint as soon as it lands, on GPUs that training is **not** using — the training run itself is never touched. Results land under `results/` while training continues, so the score curve across checkpoints is visible without waiting for the run to finish.

```bash
# terminal 1: training on GPUs 4-7
# terminal 2 (from the repo root): watcher on the spare GPUs
nohup python -u scripts/eval/watch_ckpts.py \
    --ckpt_dir checkpoints/sft-run1 \
    --tag run1 \
    --gpus 0,1,2 \
    > logs/watch_run1.log 2>&1 &
```

Each checkpoint produces `<bench>_<tag>_epoch_<e>_step_<n>.json` (bare `epoch_<e>_step_<n>` tag without `--tag`), directly readable by `collect.py`. A summary table prints after each checkpoint's suite finishes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ckpt_dir` | required | Training checkpoint directory to watch |
| `--gpus` | required | Comma-separated spare GPU ids — must not overlap the training GPUs (the watcher never auto-picks, to avoid OOMing a training card) |
| `--benchmarks` | `ifeval,humaneval,mbpp2` | Comma-separated subset of `mmlu,ifeval,humaneval,mbpp,mbpp2,hellaswag` |
| `--tag` | none | Prefix for result tags |
| `--every` | 1 | Only evaluate checkpoints whose step is a multiple of N (align with `--ckpt_interval`) |
| `--once` | False | Evaluate existing checkpoints and exit instead of polling |
| `--poll` | 30 | Poll interval in seconds |
| `--attempts` | 3 | Retries per benchmark |
| `--smoke` | False | Cheap per-benchmark invocations for a plumbing test |

Behavior notes:

- **Default benchmark set is the generation trio** (`ifeval`, `humaneval`, `mbpp2`). MMLU and HellaSwag score through the log-likelihood path, which sits at chance for every current checkpoint (a pretraining-side context pathology — see those sections), so they carry no online signal; pass them explicitly in `--benchmarks` if wanted anyway.
- **Idempotent**: benchmarks whose output file already exists are skipped, so restarting the watcher (or rerunning `--once`) resumes where it left off, including after a crash.
- **GPU contention is retried, not fatal**: if the spare GPUs are busy when a checkpoint lands, the watcher logs it and retries on the next poll.
- More benchmarks than GPUs is fine: they run in waves of `len(--gpus)` per checkpoint, one checkpoint's suite at a time (oldest first).
- **One watcher per `ckpt_dir`, one run per `ckpt_dir`**: checkpoint dir names (`epoch_<e>_step_<n>`) are reused across runs, and a second training into the same dir overwrites the old one — pointing two runs at one dir mixes their results.
- Stop with Ctrl-C (or kill): benchmarks already in flight run to completion; the watcher itself needs no cleanup.

## Collecting results

`collect.py` walks `results/` for `<bench>_<benchmark>.json` files (skipping `*_completions.json`) and renders a markdown comparison table — rows are standardized metrics per benchmark, columns are tags.

```bash
python scripts/eval/collect.py \
    --results-dir results \
    --benchmarks mmlu,ifeval,humaneval,mbpp2,hellaswag \
    --tags mix3,mix3_lr5e5,mix3_lr2e5 \
    --output results/compare.md
```

Metric names: `acc`; `pass@1`/`pass@10` plus `zero_pass`/`strong_pass` counts (code benchmarks); `acc`/`acc_norm` (HellaSwag); IFEval `json_format`/`num_bullet_lists` sub-items. `mbpp2` denotes the signature-protocol MBPP (see the MBPP section). Omitting `--output` prints to stdout.

## Tips

- **Quick test**: Use `--limit` (IFEval, HellaSwag) or `--problems` (HumanEval, MBPP) to run on a small subset first — or `run_suite.py --smoke`.
- **Auto-download**: After installing `datasets`, HumanEval, MBPP, MMLU, HellaSwag, and IFEval auto-download their datasets on first run (set `HF_ENDPOINT` if the default endpoint is unreachable). The other scripts expect user-provided data.
- **Output formats**: `--output` writes a single JSON for most scripts. PPL and IFD write an `--output_dir` containing `summary.json` plus per-file artifacts.
- **CPU mode**: MMLU, PPL, and IFD support `--device cpu --dtype float32`; weight analysis supports `--device cpu`. HumanEval generation and IFEval are CUDA-only.

> Document Update Time: 2026-07-30
