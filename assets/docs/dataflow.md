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

**H5Store**: Reads HDF5 files, supports `share_memory_()` for multi-process DataLoader workers (copies tensors to shared memory).

**MmapStore**: Memory-maps `.bin` files. OS page cache sharing is native — no explicit `share_memory_()` needed. Uses `torch.from_numpy(np.memmap(...))`.

**JsonlStore**: On-the-fly tokenization of raw JSONL files at load time. Requires a `dataset_config.json` alongside the `.jsonl` files following the same `PipelineConfig` schema with an additional `tokenizer_path` field.

All backends normalise tensors into `Store._data[Dict[str, List[Tensor]]]` + `Store._cum[Dict[str, List[int]]]` (cumulative lengths for bisect-based indexing).

## Data Keys by Training Type

| Type | Storage Keys |
|------|-------------|
| `seq` | `sequence` (→ input_ids, target_ids via offset-by-1) |
| `sft` | `sequence`, `loss_mask`, `position_ids` |
| `dpo` | `chosen`, `rejected`, `chosen_mask`, `rejected_mask` |
| `grpo` | `prompts`, `responses`, `masks`, `rewards` |

## Dataset Architecture

```
DatasetFactory.load(train_type, load_path, window_size, stride=None, storage_type=None)
  → BaseDataset.load(load_path, storage_type=None)
    → detect_format(load_path)
    → StoreFactory.create(storage_type)
    → Store.load(load_path)
      → _normalize(raw)  # base Store, shared by both backends
        → Store._data[Dict[str, List[Tensor]]] + _cum[Dict[str, List[int]]]
          → BaseDataset.__getitem__(idx)
            → get_index(idx) → [begin, end)
            → Store.fetch(begin, end, keys) → Tensor / Dict[str, Tensor]
```

`window_size` = max input length, `stride` = step between consecutive samples (defaults to `window_size`, optional). `storage_type` defaults to `None` (auto-detect via `detect_format`).

`Store.fetch(begin, end, keys)` accepts a single key (`str`) returning a `Tensor`, or a list of keys returning `Dict[str, Tensor]`. Internally uses `bisect` across multi-segment tensors. Raises `RuntimeError("Store not loaded")` if called before `load()`.

## Sampler

`ResumableDistributedSampler` supports checkpoint-aware distributed sampling:

- Tracks `start_epoch` / `start_iter` for resume
- Shuffle via `torch.Generator(seed + epoch)`
- Per-replica index slicing for DDP

## DataLoader

Standard PyTorch `DataLoader` with configurable `batch_size`, `num_workers`, `pin_memory`, `prefetch_factor`. Sampler produces indices; dataloader fetches tensor batches via `__getitem__`.

> Document Update Time: 2026-07-05
