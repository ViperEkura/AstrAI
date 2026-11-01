# Data Flow

This document describes the data pipeline: from raw text to model input tensors. For creating preprocessing configs, see [Preprocessing Guide](../guides/preprocessing.md).

## Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation) — tokenization, format detection, backends
- [Data Keys by Training Type](#data-keys-by-training-type)
- [Dataset Architecture](#dataset-architecture)
- [Sampler](#sampler)
- [DataLoader](#dataloader)

## Overview

```
JSON / JSONL Records → Pipeline (mask builder) → Tokenized Tensors
                                               ↓
                                           .bin storage
                                               ↓
                                       Store.load()
                                              ↓
                                      Store.fetch(begin, end, keys)
                                              ↓
                                       Dataset.__getitem__(idx)
                                               ↓
                                       RDSampler → DataLoader → Training
```

## Data Preparation

The offline `Pipeline` accepts `.jsonl` records and `.json` files containing one
object or a list of objects. It tokenizes them and writes binary shards (`.bin`
plus `meta.json`) with keyed tensor groups. Binary is the only registered output
writer; the pipeline cannot emit JSONL.

### Tokenization

The `Pipeline` reads JSON/JSONL records, applies the mask builder (see
[Preprocessing](../guides/preprocessing.md)), and produces token sequences:

```python
# Per JSONL line: messages → chat template → token IDs + loss mask
tokens = tokenizer.encode(rendered_text)        # List[int]
loss_mask = [0, 0, 0, 1, 1, 1, 1, 1, 1]        # 0=masked, 1=train
# Stored as flat tensors, packed with other lines by packing strategy
```

For default single-output preprocessing, the stored keys are `sequence` and
`position_ids`, plus `loss_mask` when masking is required. Packing is supported
for single-output data with a `sequence` key. Shard flushing counts the primary
flat sequence for each record: `sequence` in single-output mode, otherwise the
first flat source output.

The exact shard `meta.json` schema is a top-level mapping from key to tensor
metadata. It does not contain a storage-format or total-token field:

```json
{
  "sequence": {"shape": [123456], "dtype": "int32"},
  "loss_mask": {"shape": [123456], "dtype": "bool"},
  "position_ids": {"shape": [123456], "dtype": "int32"}
}
```

Record-aware binary data includes `"offsets": [0, ...]` inside each record key's
metadata. `Pipeline` forwards `output.record_keys` to `BinWriter`, which writes
offsets only for explicitly selected fields. DPO configs list all four standard
fields, preserving each pair independently within each shard. The default empty
list preserves the original stream-only writer behavior, including custom
outputs whose field names happen to resemble DPO fields.

### Format Detection

`detect_format(load_path)` inspects the path:

- If `load_path` is a file: `.jsonl` selects `"jsonl"`; other suffixes raise `ValueError`.
- If `load_path` is a directory: any recursive `*.bin` plus a `meta.json` selects `"bin"`; otherwise any recursive `*.jsonl` selects `"jsonl"`.
- Detection does not require `dataset_config.json`; configuration is selected later when `DatasetFactory.load()` constructs a transform via `_build_jsonl_transform()` and passes it to `JsonlStore.load()`.

### Store Backends

Storage format is auto-detected by `detect_format()`; backends are dispatched via registry:

```
StoreFactory.create("bin")   → MmapStore
StoreFactory.create("jsonl") → JsonlStore
```

Both stores inherit `Store` and compose the `Streamable` and `Recordable`
access methods.

**MmapStore**: Memory-maps `.bin` files. OS page cache sharing is native — no explicit `share_memory_()` needed. Uses `torch.from_numpy(np.memmap(...))`. `segments_are_records=False` — bin segments are contiguous streams; record access is driven by `_offsets` (written when `save_bin(..., record_keys=...)` was used at preprocessing time).

**JsonlStore**: Reads a `.jsonl` file or the sorted top-level `*.jsonl` files in
a directory. Eager transform selection is owned by
`DatasetFactory._build_jsonl_transform()` (called from `DatasetFactory.load()`) —
the factory picks the first available route:

1. An explicit `transform=` argument passed through `store.load()`.
2. `dataset_config.json` in the JSONL directory. It follows `PipelineConfig` and may add `tokenizer_path`; when omitted, the config directory is used.
3. The built-in `messages` transform when `tokenizer_path=` is supplied. It masks system/user turns, trains assistant turns, and emits document-reset position IDs.

`JsonlStore.load()` requires `transform=` to be passed explicitly for eager mode
(raises `ValueError` if missing).

Only DPO gets an automatic lazy route from `DatasetFactory`: raw JSONL plus
`tokenizer_path` installs `dpo_processor` and tokenizes each record in
`fetch_record`. GRPO does not currently have an automatic lazy processor.

Eager-loaded stores normalize tensors into `Store._data[Dict[str, List[Tensor]]]`
and `Store._cum[Dict[str, List[int]]]` (cumulative lengths for stream indexing).
Binary record indexing uses `Store._offsets[Dict[str, List[List[int]]]]`, with
one offset array per shard and key, plus `_record_cum` (cumulative shard record
counts). A binary search maps each global record index to a shard before slicing
its mmap tensor. `MmapStore` retains shard discovery order rather than sorting
paths. It builds and validates the binary record index lazily when records or
their count are requested; `DPODataset` requests validation during construction.
Token-stream access does not require a valid record index. Record fields must
have offsets and equal record counts in every shard. In record mode, `store[i]`
requests all fields and fails explicitly if any lack offsets, while
`fetch_record(i, key)` can select an indexed field from a mixed store.
Nested JSONL keys such as GRPO `responses`/`masks` remain record
values and are excluded from stream bookkeeping. Lazy DPO instead retains raw
records and processes them in `fetch_record`.

## Data Keys by Training Type

| Type | Storage Keys | Access Mode |
|------|-------------|-------------|
| `seq` | `sequence`, `position_ids` by default (`SEQDataset` consumes only `sequence`) | stream (`fetch`) |
| `sft` | `sequence`, `loss_mask`, `position_ids` | stream (`fetch`) |
| `dpo` | `chosen`, `rejected`, `chosen_mask`, `rejected_mask` | record (`fetch_record`) |
| `grpo` | `prompts`, `responses`, `masks`, `rewards` | record (`fetch_record`) |

Offline DPO `.bin` output supports single and multiple shards with per-record
offsets. Legacy DPO binary output without offsets requires reprocessing. GRPO
binary response groups are still flattened without preserving record/group
boundaries. Supported raw routes are eager JSONL for SEQ/SFT and automatic lazy
JSONL for DPO. GRPO requires a caller-built, already-loaded record store.

## Dataset Architecture

```
DatasetFactory.load(...)
  → detect_format(load_path)
  → optionally build dpo_processor for raw JSONL
  → StoreFactory.create(storage_type, window_size, stride)
  → Store.load(load_path, transform=... or processor=...)
  → DatasetFactory.create(train_type, store=store)

Stream datasets (SEQ/SFT):
  BaseDataset.__getitem__(idx)
    → Store.sample_window(idx) → [begin, end)
    → Store.fetch(begin, end, keys) → Tensor / Dict[str, Tensor]

Record datasets (DPO/GRPO):
  DPODataset/GRPODataset.__getitem__(idx)
    → Store.fetch_record(idx, keys) → Tensor / Dict[str, Tensor]
```

Class hierarchy: `BaseDataset` is the direct base of `SEQDataset`, `SFTDataset`,
`DPODataset`, and `GRPODataset`. There is no `RecordDataset` class.

`window_size` = max input length, `stride` = step between consecutive samples (defaults to `window_size`, optional). Only meaningful for stream datasets — record datasets ignore both. `storage_type` defaults to `None` (auto-detect via `detect_format`).

For raw JSONL, `tokenizer_path` builds the lazy processor only for DPO. For
SEQ/SFT it is forwarded to `JsonlStore` so the built-in eager `messages`
transform can be selected when no `dataset_config.json` exists. GRPO receives no
automatic processor. A pre-built `store` bypasses path, format, tokenizer,
window, and stride setup entirely.

`Store.fetch(begin, end, keys)` (stream mode, on `Streamable`): accepts a single key (`str`) returning a `Tensor`, or a list of keys returning `Dict[str, Tensor]`. Internally uses `bisect` across multi-segment tensors. Raises `RuntimeError("Store not loaded")` if called before `load()`.

`Store.fetch_record(index, keys)` (record mode, on `Recordable`): same key API. Uses `_offsets[key]` when present for binary record layouts; otherwise it indexes per-record JSONL tensors directly.

## Sampler

`RDSampler` supports checkpoint-aware distributed sampling:

- Tracks `start_epoch` / `start_iter` for resume
- Shuffle via `torch.Generator(seed + epoch)`
- Per-replica index slicing for DDP

## DataLoader

Standard PyTorch `DataLoader` with configurable `batch_size`, `num_workers`, `pin_memory`, `prefetch_factor`. Sampler produces indices; dataloader fetches tensor batches via `__getitem__`.

> Document Update Time: 2026-07-19
