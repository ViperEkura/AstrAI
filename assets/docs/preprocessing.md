# Preprocessing Pipeline

Declarative JSON-driven data preprocessing. `MaskBuilderFactory` supports three registered builders: `"single"` (single-output via `input.sections`), `"multi"` (multi-output via `input.sources`), and `"sectioned"` (façade dispatching to `single` or `multi` based on config).

## Contents

- [Philosophy](#philosophy)
- [Config Structure](#config-structure)
- [Quick Start](#quick-start) — SFT Chat, SFT Instruction, Pretrain, DPO, GRPO examples
- [Configuration Reference](#configuration-reference) — all fields
- [Mask Algorithm](#mask-algorithm)
- [Output Layout](#output-layout)
- [CLI](#cli)
- [Python API](#python-api)

## Philosophy

| Component | Responsibility |
|-----------|---------------|
| `tokenizer_config.json` (`chat_template`) | Formatting -- how roles become tokens |
| `pipeline.json` (`mask`) | Masking -- which roles participate in training |

A single config file captures the entire pipeline, reusable and version-controllable.

## Config Structure

```json
{
  "version":       1,
  "input":         {},   // sections (single) or sources (multi)
  "mask":          {},   // role -> "train" | "mask"
  "mask_default":  "mask",
  "preprocessing": {},
  "output":        {}
}
```

### Section Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `field` | str | -- | JSONL key to read |
| `action` | str | -- | `"train"` / `"mask"` / `"$role"` |
| `template` | bool | `false` | Apply `chat_template` per message |
| `add_special_tokens` | bool | `true` for first non-template section | Add special tokens during encode |

### Source Fields (multi-output mode)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sections` | list[dict] | -- | Same as single-output section list |
| `list_field` | bool | `false` | JSONL field holds a list; tokenise each element |
| `mask_key` | str | `"{key}_mask"` | Explicit output key for loss mask |

---

## Quick Start

### SFT Chat

Input JSONL:

```json
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
```

Config:

```json
{
  "input": {
    "sections": [
      {"field": "messages", "action": "$role", "template": true}
    ]
  },
  "mask": {
    "system": "mask",
    "user": "mask",
    "assistant": "train"
  },
  "mask_default": "mask",
  "preprocessing": {
    "max_seq_len": 2048
  },
  "output": {
    "storage_format": "bin",
    "dtype": {"loss_mask": "bool"}
  }
}
```

Output keys: `sequence` (int32), `loss_mask` (bool)

### SFT Instruction

Input JSONL:

```json
{"prompt": "Translate to French: Hello", "response": "Bonjour"}
```

Config:

```json
{
  "input": {
    "sections": [
      {"field": "prompt",   "action": "mask", "add_special_tokens": true},
      {"field": "response", "action": "train"}
    ]
  },
  "mask_default": "mask",
  "preprocessing": {
    "max_seq_len": 2048
  }
}
```

Output keys: `sequence`, `loss_mask`

### Pretrain

Input JSONL:

```json
{"text": "Artificial Intelligence is a field of computer science..."}
```

Config:

```json
{
  "input": {
    "sections": [
      {"field": "text", "action": "train"}
    ]
  },
  "preprocessing": {
    "max_seq_len": 8192,
    "min_chars": 100
  }
}
```

Output keys: `sequence` (no `loss_mask` — all tokens trained)

### DPO

Input JSONL:

```json
{"chosen": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}], "rejected": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "5"}]}
```

Config:

```json
{
  "input": {
    "sources": {
      "chosen": {
        "sections": [
          {"field": "chosen", "action": "$role", "template": true}
        ]
      },
      "rejected": {
        "sections": [
          {"field": "rejected", "action": "$role", "template": true}
        ]
      }
    }
  },
  "mask": {
    "user": "mask",
    "assistant": "train"
  },
  "mask_default": "mask"
}
```

Output keys: `chosen`, `chosen_mask`, `rejected`, `rejected_mask`

### GRPO

Input JSONL:

```json
{"prompt": [{"role": "user", "content": "What is 2+2?"}], "responses": ["4", "Five", "Four"], "rewards": [1.0, 0.3, 0.8]}
```

Config:

```json
{
  "input": {
    "sources": {
      "prompts": {
        "sections": [
          {"field": "prompt", "action": "mask", "template": true}
        ]
      },
      "responses": {
        "sections": [
          {"field": "responses", "action": "train"}
        ],
        "list_field": true,
        "mask_key": "masks"
      },
      "rewards": {
        "sections": [
          {"field": "rewards", "action": "value"}
        ]
      }
    }
  },
  "mask": {
    "user": "mask",
    "assistant": "train"
  },
  "mask_default": "mask"
}
```

Output keys: `prompts`, `prompts_mask`, `responses`, `masks`, `rewards` (float32)

- `action: "value"` — extract raw values from JSONL without tokenisation
- `list_field: true` — tokenise each list element independently, then concatenate
- `mask_key: "masks"` — rename the auto-generated mask key (default: `responses_mask`)
- `prompts_mask` is auto-generated (all masked) and unused by GRPOStrategy

---

## Configuration Reference

### `input`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sections` | list[dict] or null | `null` | Section specs for single-output mode |
| `sources` | dict[str, dict] or null | `null` | Source specs for multi-output mode (DPO/GRPO) |

When `sources` is set, `sections` is ignored.

### `mask`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mask` | dict | `{}` | `{role: "train" \| "mask"}` |
| `mask_default` | str | `"mask"` | Default action for unlisted roles |

### `preprocessing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_len` | int | `2048` | Truncate sequences to this length |
| `min_chars` | int | `50` | Skip text-mode items shorter than this |
| `max_chars` | int | `2000000` | Skip text-mode items longer than this |
| `max_items` | int or null | `null` | Stop after N documents |
| `packing_strategy` | str | `"simple"` | Packing strategy: `"simple"`, `"bfd"`, `"bfd_split"` |
| `max_packed_len` | int | `8192` | Maximum length of a packed bin |
| `truncation_mode` | str | `"keep_start"` | How to truncate sequences: `"keep_start"` or `"keep_end"` |

### `output`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `domain_key` | str or null | `null` | JSONL key for domain grouping |
| `storage_format` | str | `"bin"` | `"bin"` (mmap) or `"h5"` |
| `max_tokens_per_shard` | int | `100000000` | Flush threshold in cumulative tokens |
| `dtype` | dict[str, str] | `{}` | Per-key tensor dtype override (e.g. `{"loss_mask": "bool"}`) |
| `position_ids_mode` | str | `"doc_reset"` | How to compute position_ids: `"none"`, `"doc_reset"`, `"continuous"` |

---

## Mask Algorithm

### Template mode (`template: true`)

1. Prepend BOS token (masked)
2. For each message in the field's array:
   1. Render through `chat_template` for that single message
   2. Encode rendered text
   3. Apply mask rule for the message's role

### Non-template mode

Encode the field value as text. Mask value is 1 (train) or 0 (mask) per the section's `action`.

### Text config detection

When no section uses `template` and all sections have `action: "train"`, the builder omits `loss_mask` from the output — all tokens are trained.

---

## Output Layout

### Single-Shard (`bin`)

```
output/
  __default__/
    shard_0000/
      meta.json
      sequence.bin
      loss_mask.bin
  wiki/
    shard_0000/
      meta.json
      sequence.bin
      loss_mask.bin
```

### Multi-Shard (`bin`)

When `max_tokens_per_shard` is exceeded:

```
output/
  __default__/
    shard_0000/
      meta.json
      sequence.bin
      loss_mask.bin
    shard_0001/
      meta.json
      sequence.bin
      loss_mask.bin
```

For `bin` format, `MmapStore` discovers all shards under the domain directory via `rglob("meta.json")`. For `h5` format, `H5Store` discovers `.h5`/`.hdf5` files via recursive glob.

---

## CLI

```bash
# SFT
python scripts/tools/preprocess.py data/sft/*.jsonl -o output/sft/ -c configs/sft_chat.json

# DPO
python scripts/tools/preprocess.py data/dpo/*.jsonl -o output/dpo/ -c configs/dpo.json --tokenizer_path params

# GRPO
python scripts/tools/preprocess.py data/grpo/*.jsonl -o output/grpo/ -c configs/grpo.json
```

---

## Python API

```python
from astrai.preprocessing.pipeline import Pipeline
from astrai.config.preprocess_config import PipelineConfig

config = PipelineConfig.from_file("sft.json")
Pipeline(
    config,
    ["data_part1.jsonl", "data_part2.jsonl"],
    output_dir="output/",
    tokenizer_path="params",
).run()
```

> Document Update Time: 2026-07-09
