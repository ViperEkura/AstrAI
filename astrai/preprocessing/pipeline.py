"""Config-driven JSONL preprocessing pipeline.

Composes a :class:`BaseMaskBuilder` (selected by ``input.type``) with
sharding and flush to ``.h5`` / ``.bin`` storage.  Packing, position-id
generation and storage writing are each delegated to pluggable strategies,
dispatched by configuration keys.
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
from astrai.preprocessing.builder import MaskBuilderFactory
from astrai.preprocessing.packing import PackingStrategyFactory
from astrai.preprocessing.position_id import PositionIdStrategyFactory
from astrai.preprocessing.writer import StoreWriterFactory
from astrai.tokenize import AutoTokenizer

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

        self.mask_builder = MaskBuilderFactory.create("sectioned")
        self._packer = PackingStrategyFactory.create(
            config.preprocessing.packing_strategy
        )
        self._position_id = PositionIdStrategyFactory.create(
            config.output.position_ids_mode
        )
        self._writer = StoreWriterFactory.create(config.output.storage_format)

    def transform(self, item: dict) -> Optional[dict]:
        return self.mask_builder.build(item, self.config, self._tokenizer)

    def run(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        domains: dict = defaultdict(lambda: defaultdict(list))
        total_tokens = 0
        shard_idx: dict[str, int] = defaultdict(int)
        count = 0

        pp = self.config.preprocessing

        for item in tqdm.tqdm(
            self._iter_items(), desc="Tokenizing", unit="docs", mininterval=0.5
        ):
            if pp.max_items and count >= pp.max_items:
                break

            try:
                result = self.transform(item)
            except Exception:
                logger.warning(
                    "Failed to process item #%d, skipping", count + 1, exc_info=True
                )
                continue
            if result is None:
                continue

            domain = result.pop("domain", "__default__")

            is_multi = bool(getattr(self.config.input, "sources", None))
            if is_multi:
                ids = self._primary_ids(result)
            else:
                ids = result.pop("sequence")
                result["sequence"] = ids

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

        if total_tokens > 0:
            self._flush(domains, shard_idx)

    @staticmethod
    def _primary_ids(result: dict) -> list:
        """Return the first list-valued entry in *result* as the primary id
        sequence for token counting."""
        for val in result.values():
            if isinstance(val, list) and val and isinstance(val[0], int):
                return val
        return []

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
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

    def _flush(self, domains, shard_idx):
        for domain, keys in domains.items():
            idx = shard_idx[domain]

            pp = self.config.preprocessing
            original_sequences = keys.get("sequence", [])
            mode = self.config.output.position_ids_mode

            if mode == "doc_reset" and original_sequences:
                keys["position_ids"] = [list(range(len(s))) for s in original_sequences]

            keys = self._packer.apply(dict(keys), pp.max_packed_len, pp.truncation_mode)

            tensors: Dict[str, List[torch.Tensor]] = {}
            for key, ids_list in keys.items():
                dt = _STR_TO_DTYPE.get(
                    self.config.output.dtype.get(key, "int32"), torch.int32
                )
                tensors[key] = [
                    torch.tensor(list(chain.from_iterable(ids_list)), dtype=dt)
                ]

            if mode == "continuous" and original_sequences:
                pos_ids = self._position_id.generate(keys.get("sequence", []))
                if pos_ids:
                    tensors["position_ids"] = [torch.tensor(pos_ids, dtype=torch.int32)]

            self._writer.save(self.output_dir, domain, idx, tensors)
            shard_idx[domain] = idx + 1

            first_key = "sequence" if "sequence" in tensors else next(iter(tensors))
            tqdm.tqdm.write(
                f"  saved {domain}/shard_{idx:04d}  "
                f"({tensors[first_key][0].numel():,} tokens)"
            )
