# Preprocessing Pipeline

Declarative JSON-driven data preprocessing. The built-in pipeline directly constructs `SectionedMaskBuilder`, which selects `SingleOutputMaskBuilder` for `input.sections` or `MultiOutputMaskBuilder` for `input.sources`. All three builders remain available for direct construction and custom renderer injection.

## Contents

- [Philosophy](#philosophy)
- [Config Structure](#config-structure)
- [Quick Start](#quick-start) — SFT Chat, SFT Instruction, Pretrain, DPO, GRPO examples
- [Configuration Reference](#configuration-reference) — all fields
- [Mask Algorithm](#mask-algorithm)
- [Output Layout](#output-layout)
- [Training Compatibility](#training-compatibility)
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
| `action` | str | -- | `"train"` / `"mask"` / `"$role"` / `"value"`; `"value"` copies raw values without tokenization |
| `template` | bool | `false` | Apply `chat_template` per message |
| `add_special_tokens` | bool | `true` when the first retained section is non-template; `false` for list elements | Add special tokens during encode; template sections always encode without additional special tokens |

### Source Fields (multi-output mode)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sections` | list[dict] | -- | Same as single-output section list |
| `list_field` | bool | `false` | JSONL field holds a list; tokenise each element |
| `mask_key` | str | `"{key}_mask"` | Explicit output key for loss mask |

Scalar and batch rendering use the same filtering, masking and truncation rules.
An empty or filtered section does not consume the first-section special-token
default. Top-level template sequences retain the existing masked BOS prefix;
individual template messages and list elements do not add another prefix.

For `list_field`, every element defaults to `add_special_tokens: false`; an
explicit `true` or `false` is honored for non-template elements. This fixes an
earlier scalar/batch discrepancy: batch encoding used to add special tokens by
default, while scalar encoding ignored an explicit `true`. Template elements
continue to use the tokens supplied by `chat_template`, without extra encoder
special tokens.

Custom `SectionRenderer` implementations can override `process_sections` and
`process_sections_batch`, or the corresponding `process_list_field` methods.
Builders keep these scalar and batch entry points separate: scalar rendering
uses `encode(str)` and supports scalar-only tokenizers; batch rendering uses
`encode(list)` grouped by special-token policy. A batch renderer/tokenizer must
return one result per input record/text in the same order.

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

Output keys: `sequence` (int32), `loss_mask` (bool), `position_ids` (int32)

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

Output keys: `sequence`, `loss_mask`, `position_ids`

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

Output keys: `sequence`, `position_ids` (no `loss_mask` — all tokens trained)

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
  "mask_default": "mask",
  "output": {
    "record_keys": ["chosen", "rejected", "chosen_mask", "rejected_mask"]
  }
}
```

Output keys: `chosen`, `chosen_mask`, `rejected`, `rejected_mask`

The offline `Pipeline` preserves each preference pair in `.bin` output by
writing per-record offsets for the four keys explicitly listed in
`output.record_keys`. Single and multiple shards are supported. When both a
source and its mask are listed, omitted all-ones masks are materialized using
that source's own length; explicit masks are preserved and checked for matching
lengths. Custom preprocessing may use another `mask_key`, but `DPODataset`
expects the four standard names shown above. Pairs are not packed together.
Binary DPO data produced without offsets must be preprocessed again.
See [Training Compatibility](#training-compatibility).

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
- `list_field: true` — tokenise each list element independently; builders retain one token and mask list per element
- `mask_key: "masks"` — rename the auto-generated mask key (default: `responses_mask`)
- `prompts_mask` is auto-generated (all masked) and unused by GRPOStrategy

The offline `Pipeline` flattens GRPO response groups for `.bin` output without
preserving their boundaries, and there is no automatic raw-JSONL GRPO processor
in `DatasetFactory`. See
[Training Compatibility](#training-compatibility) for the supported routes.

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
| `max_seq_len` | int | `2048` | Maximum single segment length; overflow runs after full tokenization |
| `min_chars` | int | `50` | Skip text-mode items shorter than this |
| `max_chars` | int | `2000000` | Skip text-mode items longer than this |
| `max_items` | int or null | `null` | Stop after N documents |
| `batch_size` | int | `256` | Records per tokenization batch |
| `overflow` | str | `"truncate"` | Long single-output samples: `"truncate"` or `"split"` |
| `packing` | str | `"bfd"` | `"bfd"` merges segments by best fit; `"none"` keeps each segment separate |
| `max_packed_len` | int | `8192` | Maximum length of a packed bin |
| `truncation_mode` | str | `"keep_start"` | How to truncate sequences: `"keep_start"` or `"keep_end"` |

### Tokenizer parallelism

Batch inputs passed to AutoTokenizer.encode use the Rust tokenizers backend.
Enable its Rayon thread pool before starting preprocessing with:

```bash
TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=2 \
python scripts/tools/preprocess.py data/part-000.jsonl -o output/ -c configs/pretrain.json
```

RAYON_NUM_THREADS limits threads per process. In a job that also uses
multiple worker processes, each process can create its own Rayon pool;
start with a small thread limit and tune the process-count times
thread-count against available CPU capacity. Set these variables before
launching the job; they do not reconfigure already-running workers.

### `output`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `domain_key` | str or null | `null` | JSONL key for domain grouping |
| `storage_format` | str | `"bin"` | Pipeline output format. Only `"bin"` is supported; other formats are rejected before preprocessing starts. JSONL input is still supported. |
| `max_tokens_per_shard` | int | `100000000` | Flush threshold counted from each record's primary flat sequence: `sequence` for single-output data, otherwise the first flat source output |
| `dtype` | dict[str, str] | `{}` | Per-key tensor dtype override (e.g. `{"loss_mask": "bool"}`) |
| `position_ids_mode` | str | `"doc_reset"` | How to compute position_ids: `"none"`, `"doc_reset"`, `"continuous"` |
| `record_keys` | list[str] | `[]` | Explicit flat record fields to write with offsets. Names must be unique and non-empty; selected fields must exist and have equal record counts. No fields are inferred from DPO-like names. |

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
      position_ids.bin
  wiki/
    shard_0000/
      meta.json
      sequence.bin
      loss_mask.bin
      position_ids.bin
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
      position_ids.bin
    shard_0001/
      meta.json
      sequence.bin
      loss_mask.bin
      position_ids.bin
```

`MmapStore` discovers binary shards recursively through their `meta.json` files.
Each shard's metadata is a top-level object keyed by tensor name:

```json
{
  "sequence": {"shape": [123456], "dtype": "int32"},
  "loss_mask": {"shape": [123456], "dtype": "bool"},
  "position_ids": {"shape": [123456], "dtype": "int32"}
}
```

An optional `offsets` array appears for record-oriented binary data written
through `save_bin(..., record_keys=...)`. `Pipeline` forwards the explicitly
configured `output.record_keys` to `BinWriter`; the default empty list preserves
stream-only output. Offsets start at zero in each shard, end at the field's token
count, and have one more entry than the number of records.
`MmapStore` preserves the existing filesystem discovery order (not guaranteed to
be sorted) and locates records using cumulative shard record counts. Record
index validation is deferred until record access or record-count queries;
invalid offsets do not block token-stream access to otherwise valid tensors.
DPO validates offsets and token/mask boundaries when the dataset is constructed.
In record mode, `store[i]` requests all fields and raises if any lack offsets;
use `fetch_record(i, key)` to select an indexed field from a mixed record/stream
store. Field names are not silently dropped from `store[i]`.

---

## Training Compatibility

| Training type | Supported input route |
|---------------|-----------------------|
| `seq` | Offline preprocessed `.bin`, or raw `.jsonl` eagerly transformed by `JsonlStore` using `dataset_config.json` or the built-in `messages` config |
| `sft` | Offline preprocessed `.bin`, or raw `.jsonl` through the same eager transform routes |
| `dpo` | Offline preprocessed `.bin` with record offsets (single or multiple shards), raw `.jsonl` through the automatic lazy DPO processor when `tokenizer_path` is supplied, or a caller-provided record store |
| `grpo` | A caller-provided, already-loaded `Store` with record-shaped `prompts`, `responses`, `masks`, and `rewards`; no automatic raw-JSONL processor is currently wired |

Offline GRPO `.bin` output is not currently loadable for training. GRPO response
groups are flattened before the binary writer and their record/group boundaries
are not preserved.

---

## CLI

```bash
# SFT
python scripts/tools/preprocess.py data/sft/part-000.jsonl -o output/sft/ -c configs/sft_chat.json --batch_size 128

# DPO
python scripts/tools/preprocess.py data/dpo/part-000.jsonl -o output/dpo/ -c configs/dpo.json --tokenizer_path params

# GRPO
python scripts/tools/preprocess.py data/grpo/part-000.jsonl -o output/grpo/ -c configs/grpo.json
```

Inputs may be `.jsonl` files or `.json` files containing one object or a list of
objects. Each positional path must exist. A wildcard such as `data/*.jsonl`
works only when the invoking shell expands it before Click receives the
arguments; otherwise pass the files explicitly.

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

## Python API migration

Construct `SectionedMaskBuilder()`, `SingleOutputMaskBuilder()` or
`MultiOutputMaskBuilder()` directly instead of using `MaskBuilderFactory`.
Their `build` / `build_batch` methods and renderer injection remain available.

Position generation is now the pure function
`generate_position_ids(sequences, mode)` from `astrai.preprocessing`.
It returns `[]` for `none`, resets numbering for each document in `doc_reset`,
and returns one continuous range for `continuous`. Empty sequences produce no
positions; an unsupported mode raises `ValueError`. The position strategy
classes and their factory have been removed. Online transforms omit the
`position_ids` key when there are no positions. Offline packing still attaches
per-document resets before packing and continuous positions after packing.
Configuration names and stored data formats are unchanged.


### Writer construction

The pipeline constructs `BinWriter()` directly. `StoreWriter` and
`StoreWriterFactory` have been removed; direct callers can import `BinWriter`
from `astrai.preprocessing`. Record validation and partial-shard cleanup remain
part of the writer. `output.storage_format` accepts only `"bin"`, including
when loading a configuration file. This restriction concerns output;
`JsonlStore` and JSONL input tokenization remain available.

### SFT packs with explicit boundaries

Use the independent defaults `overflow: truncate` and `packing: bfd`.
Set both `max_seq_len` and `max_packed_len` to the training context length
(for example, 2048). Overflow uses the smaller limit; BFD uses the packed
limit as bin capacity. `split` preserves token chunks but does not copy
previous context or repair incomplete conversation turns. `none` still
applies the selected overflow policy.

For SFT, set `output.record_keys` to
`["sequence", "loss_mask", "seq_lengths"]` and keep
`output.position_ids_mode: doc_reset`. Each bin is one binary record;
`seq_lengths` holds the lengths of its independent segments. No padding
or per-token position IDs are stored. SFTDataset reads every record,
including the final partial pack. The trainer selects `sft_collate_fn`
automatically for packed SFT datasets; custom DataLoaders must select it too.

The collator pads to the longest pack in the batch, resets positions for
each segment, and shifts labels within segments. Segment-ending targets
and padding are ignored by the loss. Token ID zero is a valid masked
filler, not an assumed tokenizer padding symbol. Existing SFT attention
constructs a same-segment causal mask from the position resets. Resets
and EOS tokens alone do not isolate attention. This path uses the
existing dense attention mask and requires a compatible model/backend;
packed context parallelism remains unsupported.

The old `packing_strategy` option and strategy classes have been removed.
Migrate `simple` to `packing: none`, `bfd` to `packing: bfd`, and
`bfd_split` to `overflow: split, packing: bfd`. Replace the temporary
`bfd_pad_sft` option with the record format above and regenerate data.
Old stream datasets remain readable, but cannot recover discarded pack
boundaries. Single-output builders now return full tokenized samples;
use `prepare_sequences` to apply length and packing policies.
DPO/GRPO outputs retain independent per-output truncation and are not
split or merged by these single-output policies.
