"""Config-driven JSONL preprocessing pipeline.

Composes a :class:`BaseMaskBuilder` (selected by ``input.type``) with
sharding and flush to ``.h5`` / ``.bin`` storage.  Packing, position-id
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
    build_preprocessing_components,
    primary_ids,
)
from astrai.preprocessing.packing import PackingStrategyFactory
from astrai.preprocessing.writer import StoreWriterFactory

logger = logging.getLogger(__name__)

_STR_TO_DTYPE: dict[str, torch.dtype] = {
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
        input_paths: list[str],
        output_dir: str,
        tokenizer_path: str,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.config = config
        self.paths = input_paths
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path

        self.tokenizer, self.mask_builder, self._position_id = (
            build_preprocessing_components(config, tokenizer_path)
        )
        self._packer = PackingStrategyFactory.create(
            config.preprocessing.packing_strategy
        )
        self._writer = StoreWriterFactory.create(config.output.storage_format)

    def transform(self, item: dict) -> Optional[dict]:
        return self.mask_builder.build(item, self.config, self.tokenizer)

    def transform_batch(self, items: list[dict]) -> list[Optional[dict]]:
        return self.mask_builder.build_batch(items, self.config, self.tokenizer)

    def run(self):
        domains: dict = defaultdict(lambda: defaultdict(list))
        total_tokens = 0
        shard_idx: dict[str, int] = defaultdict(int)
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

                domain = result.pop("domain", "__default__")
                ids = primary_ids(result)
                if not ids:
                    continue

                bucket = domains[domain]
                self._align_bucket(bucket, result, ids)
                for key, val in result.items():
                    bucket[key].append(val)

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

    @staticmethod
    def _align_bucket(bucket: dict, result: dict, ids: list):
        """Pad previously-accumulated keys that are missing from *result*."""
        for key in list(bucket.keys()):
            if key in result:
                continue
            bucket[key].append([0] * len(ids))

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

            pp = self.config.preprocessing
            original_sequences = keys.get("sequence", [])
            mode = self.config.output.position_ids_mode

            keys = self._inject_doc_reset_position_ids(keys, mode, original_sequences)
            keys = self._packer.apply(dict(keys), pp.max_packed_len, pp.truncation_mode)
            tensors = self._to_tensors(keys)
            tensors = self._inject_continuous_position_ids(
                tensors, mode, keys.get("sequence", [])
            )

            self._writer.save(self.output_dir, domain, idx, tensors)
            shard_idx[domain] = idx + 1

            first_key = "sequence" if "sequence" in tensors else next(iter(tensors))
            tqdm.tqdm.write(
                f"  saved {domain}/shard_{idx:04d}  "
                f"({tensors[first_key][0].numel():,} tokens)"
            )

    def _inject_doc_reset_position_ids(
        self,
        keys: Dict[str, list],
        mode: str,
        original_sequences: List[List[int]],
    ) -> Dict[str, list]:
        """Attach per-document position_ids before packing (``doc_reset``).

        ``doc_reset`` position ids must enter the packer so that each
        packed bin concatenates the per-doc ranges in bin order.  The
        per-record structure ``[range(len(s)) for s in seqs]`` is required
        by the packer (it concatenates per-record lists per bin); the
        ``PositionIdStrategy.generate`` flattens, so it cannot be used
        directly here — it is only consulted for the ``continuous``
        post-packing path.
        """
        if mode != "doc_reset" or not original_sequences:
            return keys
        keys["position_ids"] = [list(range(len(s))) for s in original_sequences]
        return keys

    def _inject_continuous_position_ids(
        self,
        tensors: Dict[str, List[torch.Tensor]],
        mode: str,
        packed_sequences: List[List[int]],
    ) -> Dict[str, List[torch.Tensor]]:
        """Attach a single continuous position_ids tensor after packing.

        ``continuous`` mode spans the whole shard (post-packing), so it
        cannot participate in bin packing — it is computed from the
        packed sequences and appended directly to the tensor dict.
        """
        if mode != "continuous" or not packed_sequences:
            return tensors
        pos_ids = self._position_id.generate(packed_sequences)
        if pos_ids:
            tensors["position_ids"] = [torch.tensor(pos_ids, dtype=torch.int32)]
        return tensors

    def _to_tensors(self, keys: Dict[str, list]) -> Dict[str, List[torch.Tensor]]:
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
