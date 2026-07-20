# Data Flow

This document describes the data pipeline: from raw text to model input tensors. For creating preprocessing configs, see [Preprocessing Guide](preprocessing.md).

## Contents

- [Overview](#overview)
- [Data Preparation](#data-preparation) — tokenization, format detection, backends
- [Data Keys by Training Type](#data-keys-by-training-type)
- [Dataset Architecture](#dataset-architecture)
- [Sampler](#sampler)
- [DataLoader](#dataloader)

## Overview

```
JSONL Lines → Pipeline (mask builder) → Tokenized Tensors
                                              ↓
                                      .h5 or .bin storage
                                              ↓
                                      Store.load()
                                              ↓
                                      Store.fetch(begin, end, keys)
                                              ↓
                                      BaseDataset.__getitem__(idx)
                                              ↓
                                      Sampler → DataLoader → Training / Inference
```

## Data Preparation

Raw text is tokenized via `AutoTokenizer.encode()` and saved as HDF5 (`.h5`) or binary (`.bin` + `meta.json`) files with keyed tensor groups.

### Tokenization

The `Pipeline` reads JSONL lines, applies the mask builder (see [Preprocessing](preprocessing.md)), and produces flat token sequences:

```python
# Per JSONL line: messages → chat template → token IDs + loss mask
tokens = tokenizer.encode(rendered_text)        # List[int]
loss_mask = [0, 0, 0, 1, 1, 1, 1, 1, 1]        # 0=masked, 1=train
# Stored as flat tensors, packed with other lines by packing strategy
```

The output `meta.json` records the storage format, key names, dtype, total token count, and tensor shapes for each shard.

### Format Detection

`detect_format(load_path)` inspects the path:

- If `load_path` is a file: checks suffix — `.h5`/`.hdf5` → `"h5"`, `.jsonl` → `"jsonl"`, unknown suffix raises `ValueError`
- If `load_path` is a directory: recursively globs for `*.h5`/`*.hdf5` files → `"h5"`, `*.bin` + `**/meta.json` → `"bin"`, or `*.jsonl` + `dataset_config.json` → `"jsonl"`

### Store Backends

Storage format is auto-detected by `detect_format()`; backends are dispatched via registry:

```
StoreFactory.create("h5")    → H5Store
StoreFactory.create("bin")   → MmapStore
StoreFactory.create("jsonl") → JsonlStore
```

All three inherit `Store` (base, owns `_data`/`_cum`/`_offsets`/`_normalize`) plus the `Streamable` and `Recordable` mixins, so every backend supports both `fetch(begin, end, keys)` (stream) and `fetch_record(index, keys)` (record) APIs.

**H5Store**: Reads HDF5 files. Tensors are loaded into host memory and normalized into segmented storage. `segments_are_records=True` — each `data_i` dataset is one record.

**MmapStore**: Memory-maps `.bin` files. OS page cache sharing is native — no explicit `share_memory_()` needed. Uses `torch.from_numpy(np.memmap(...))`. `segments_are_records=False` — bin segments are contiguous streams; record access is driven by `_offsets` (written when `save_bin(..., record_keys=...)` was used at preprocessing time).

**JsonlStore**: On-the-fly tokenization of raw JSONL files at load time. Requires a `dataset_config.json` alongside the `.jsonl` files following the same `PipelineConfig` schema with an additional `tokenizer_path` field. Two modes: eager (default, applies `TokenizeTransform` to all records at load) and lazy (`processor=fn` given, defers tokenisation to `fetch_record` — used by DPO/GRPO).

All backends normalise tensors into `Store._data[Dict[str, List[Tensor]]]` + `Store._cum[Dict[str, List[int]]]` (cumulative lengths for bisect-based stream indexing) + `Store._offsets[Dict[str, List[int]]]` (per-record offsets for record-mode indexing). Nested keys (GRPO `responses`/`masks` as `List[List[Tensor]]`) are stored as-is and excluded from both bookkeepings — they are only accessed record-by-record.

## Data Keys by Training Type

| Type | Storage Keys | Access Mode |
|------|-------------|-------------|
| `seq` | `sequence` (→ input_ids, target_ids via offset-by-1) | stream (`fetch`) |
| `sft` | `sequence`, `loss_mask`, `position_ids` | stream (`fetch`) |
| `dpo` | `chosen`, `rejected`, `chosen_mask`, `rejected_mask` | record (`fetch_record`) |
| `grpo` | `prompts`, `responses`, `masks`, `rewards` | record (`fetch_record`) |

## Dataset Architecture

```
DatasetFactory.load(train_type, load_path, window_size, stride=None,
                    storage_type=None, tokenizer_path=None,
                    max_position_embeddings=2048, store=None)
  → BaseDataset.load(load_path, storage_type=None)
    → detect_format(load_path)
    → StoreFactory.create(storage_type)
    → Store.load(load_path)
      → _normalize(raw)  # base Store, shared by both backends
        → Store._data[Dict[str, List[Tensor]]]
          + _cum[Dict[str, List[int]]]   (stream mode)
          + _offsets[Dict[str, List[int]]]  (record mode)

Stream datasets (SEQ/SFT):
  BaseDataset.__getitem__(idx)
    → get_index(idx) → [begin, end)
    → Store.fetch(begin, end, keys) → Tensor / Dict[str, Tensor]

Record datasets (DPO/GRPO via RecordDataset):
  RecordDataset.__getitem__(idx)
    → Store.fetch_record(idx, keys) → Tensor / Dict[str, Tensor]
```

Class hierarchy: `BaseDataset` ← `SEQDataset` / `SFTDataset` (stream); `BaseDataset` ← `RecordDataset` ← `DPODataset` / `GRPODataset` (record).

`window_size` = max input length, `stride` = step between consecutive samples (defaults to `window_size`, optional). Only meaningful for stream datasets — record datasets ignore both. `storage_type` defaults to `None` (auto-detect via `detect_format`).

`tokenizer_path` triggers lazy on-the-fly tokenisation for record datasets on raw JSONL (DPO builds a `dpo_processor`; SEQ/SFT/pre-tokenised backends ignore it). `store` (pre-built `Store`) bypasses `load_path`/`storage_type`/`tokenizer_path` entirely — the caller controls Store construction.

`Store.fetch(begin, end, keys)` (stream mode, on `Streamable`): accepts a single key (`str`) returning a `Tensor`, or a list of keys returning `Dict[str, Tensor]`. Internally uses `bisect` across multi-segment tensors. Raises `RuntimeError("Store not loaded")` if called before `load()`.

`Store.fetch_record(index, keys)` (record mode, on `Recordable`): same key API. Uses `_offsets[key]` when present (bin layout with per-record offsets), otherwise indexes `_data[key]` directly (H5/JSONL where each segment is one record).

## Sampler

`ResumableDistributedSampler` supports checkpoint-aware distributed sampling:

- Tracks `start_epoch` / `start_iter` for resume
- Shuffle via `torch.Generator(seed + epoch)`
- Per-replica index slicing for DDP

## DataLoader

Standard PyTorch `DataLoader` with configurable `batch_size`, `num_workers`, `pin_memory`, `prefetch_factor`. Sampler produces indices; dataloader fetches tensor batches via `__getitem__`.

> Document Update Time: 2026-07-19
