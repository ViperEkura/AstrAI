"""Config-driven JSONL preprocessing pipeline.

Composes a :class:`BaseMaskBuilder` (selected by ``input.type``) with
sharding and flush to ``.bin`` storage.  Packing, position-id
generation and storage writing are each delegated to pluggable strategies,
dispatched by configuration keys.

Record iteration, mask building, primary-id extraction and per-key
accumulation are shared with :class:`TokenizeTransform` via the
:mod:`astrai.preprocessing.core` helpers.
"""

import json
import logging
import os
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional

import torch
import tqdm

from astrai.config.preprocess_config import PipelineConfig
from astrai.preprocessing.core import (
    accumulate_record,
    build_preprocessing_components,
    prepare_sequences,
    primary_ids,
)
from astrai.preprocessing.writer import BinWriter

logger = logging.getLogger(__name__)

_STR_TO_DTYPE: Dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def filter_by_length(text: str, min_len: int = 50, max_len: int = 2_000_000) -> bool:
    return min_len <= len(text) <= max_len


class Pipeline:
    """Tokenization pipeline driven by a declarative :class:`PipelineConfig`.

    Usage::

        config = PipelineConfig.from_file("sft_pipeline.json")
        Pipeline(config, ["data.jsonl"], output_dir="out", tokenizer_path="params").run()
    """

    def __init__(
        self,
        config: PipelineConfig,
        input_paths: List[str],
        output_dir: str,
        tokenizer_path: str,
    ):
        if config.output.storage_format != "bin":
            raise ValueError("Pipeline output storage_format only supports 'bin'")
        os.makedirs(output_dir, exist_ok=True)
        self.config = config
        self.paths = input_paths
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path

        self.tokenizer, self.mask_builder = build_preprocessing_components(
            tokenizer_path
        )
        self._writer = BinWriter()

    def transform(self, item: Dict) -> Optional[Dict]:
        return self.mask_builder.build(item, self.config, self.tokenizer)

    def transform_batch(self, items: List[Dict]) -> List[Optional[Dict]]:
        return self.mask_builder.build_batch(items, self.config, self.tokenizer)

    def run(self):
        domains: Dict = defaultdict(lambda: defaultdict(list))
        total_tokens = 0
        shard_idx: Dict[str, int] = defaultdict(int)
        count = 0

        pp = self.config.preprocessing

        progress = tqdm.tqdm(desc="Tokenizing", unit="docs", mininterval=0.5)
        stop = False
        for items in self._iter_batches(pp.batch_size):
            progress.update(len(items))
            try:
                results = self.transform_batch(items)
            except Exception:
                logger.warning(
                    "Failed to process batch, retrying records individually",
                    exc_info=True,
                )
                results = []
                for item in items:
                    try:
                        results.append(self.transform(item))
                    except Exception:
                        logger.warning(
                            "Failed to process item, skipping", exc_info=True
                        )
                        results.append(None)

            for result in results:
                if pp.max_items and count >= pp.max_items:
                    stop = True
                    break
                if result is None:
                    continue

                self._complete_record_masks(result)
                domain = result.pop("domain", "__default__")
                ids = primary_ids(result)
                if not ids:
                    continue

                bucket = domains[domain]
                accumulate_record(bucket, result, ids)

                count += 1
                total_tokens += len(ids)

                if total_tokens >= self.config.output.max_tokens_per_shard:
                    self._flush(domains, shard_idx)
                    domains.clear()
                    total_tokens = 0
            if stop:
                break

        progress.close()

        if total_tokens > 0:
            self._flush(domains, shard_idx)

    def _complete_record_masks(self, result: Dict) -> None:
        record_keys = set(self.config.output.record_keys)
        if not record_keys:
            return
        sources = self.config.input.sources or {}
        for name, spec in sources.items():
            mask_key = spec.get("mask_key", f"{name}_mask")
            if not {name, mask_key}.issubset(record_keys) or name not in result:
                continue
            if mask_key not in result:
                result[mask_key] = [1] * len(result[name])
            if len(result[name]) != len(result[mask_key]):
                raise ValueError(
                    f"Record mask '{mask_key}' must match '{name}' lengths"
                )
        if "sequence" in result and "loss_mask" in record_keys:
            result.setdefault("loss_mask", [1] * len(result["sequence"]))
        missing = record_keys.difference(result).difference({"seq_lengths"})
        if missing:
            raise ValueError(f"Record output is missing fields: {sorted(missing)}")

    def _validate_record_masks(self, tensors: Dict[str, List[torch.Tensor]]) -> None:
        record_keys = set(self.config.output.record_keys)
        for name, spec in (self.config.input.sources or {}).items():
            mask_key = spec.get("mask_key", f"{name}_mask")
            if not {name, mask_key}.issubset(record_keys):
                continue
            sequences, masks = tensors[name], tensors[mask_key]
            if len(sequences) != len(masks) or any(
                sequence.numel() != mask.numel()
                for sequence, mask in zip(sequences, masks)
            ):
                raise ValueError(
                    f"Record mask '{mask_key}' must match '{name}' lengths"
                )

    def _iter_items(self):
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as f:
                if path.endswith(".json"):
                    data = json.load(f)
                    if isinstance(data, dict):
                        yield data
                    elif isinstance(data, list):
                        yield from data
                else:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        yield json.loads(line)

    def _iter_batches(self, batch_size: int):
        batch_size = max(1, batch_size)
        batch = []
        for item in self._iter_items():
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _flush(self, domains, shard_idx):
        for domain, keys in domains.items():
            idx = shard_idx[domain]

            keys = prepare_sequences(dict(keys), self.config)
            tensors = self._to_tensors(keys)

            self._validate_record_masks(tensors)
            writer_kwargs = (
                {"record_keys": self.config.output.record_keys}
                if self.config.output.record_keys
                else {}
            )
            self._writer.save(self.output_dir, domain, idx, tensors, **writer_kwargs)
            shard_idx[domain] = idx + 1

            first_key = "sequence" if "sequence" in tensors else next(iter(tensors))
            tqdm.tqdm.write(
                f"  saved {domain}/shard_{idx:04d}  "
                f"({sum(t.numel() for t in tensors[first_key]):,} tokens)"
            )

    def _to_tensors(self, keys: Dict[str, List]) -> Dict[str, List[torch.Tensor]]:
        """Convert packed per-key id lists to tensors.

        Honours ``config.output.dtype`` overrides per key; falls back to
        ``int32``.  Handles three shapes (see
        :func:`astrai.preprocessing.core.to_per_record_tensors` for the
        equivalent online-path helper):
        - ``List[int]`` per record → one tensor per record.
        - ``List[List[int]]`` per record (GRPO responses/masks) → one tensor
          per record, inner lists flattened.
        - ``List[int]`` for the whole shard (pre-packed keys) → single tensor.
        """
        tensors: Dict[str, List[torch.Tensor]] = {}
        for key, ids_list in keys.items():
            dt = _STR_TO_DTYPE.get(
                self.config.output.dtype.get(key, "int32"), torch.int32
            )
            if ids_list and isinstance(ids_list[0], list):
                tensors[key] = [
                    torch.tensor(
                        list(chain.from_iterable(ids))
                        if ids and isinstance(ids[0], list)
                        else ids,
                        dtype=dt,
                    )
                    for ids in ids_list
                ]
            else:
                tensors[key] = [
                    torch.tensor(list(chain.from_iterable(ids_list)), dtype=dt)
                ]
        return tensors
