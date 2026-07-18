"""Tokenization transform for JSONL record streams.

Bridges the Reader layer (``JsonlStore`` reads raw JSON records) and the
Dataset layer (expects per-record tensors).  Holds the tokenizer,
mask-builder and position-id strategy together so that I/O code stays
free of model dependencies.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch

from astrai.config.preprocess_config import PipelineConfig
from astrai.preprocessing.builder import MaskBuilderFactory
from astrai.preprocessing.position_id import PositionIdStrategyFactory
from astrai.tokenize import AutoTokenizer


class TokenizeTransform:
    """Tokenize raw JSONL record dicts into per-key tensor lists.

    Owns the three preprocessing concerns that were previously inlined in
    ``JsonlStore``: tokenization, loss-mask construction and position-id
    generation.  Constructing it loads the tokenizer, so it is intentionally
    cheap to pass around once built.

    Args:
        config: Pipeline config describing sections / masks / position mode.
        tokenizer_path: Path passed to ``AutoTokenizer.from_pretrained``.
    """

    def __init__(self, config: PipelineConfig, tokenizer_path: str):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.mask_builder = MaskBuilderFactory.create("sectioned")
        self.position_strategy = PositionIdStrategyFactory.create(
            config.output.position_ids_mode
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "TokenizeTransform":
        """Build from a ``dataset_config.json`` file path.

        The config file follows :class:`PipelineConfig` schema with an
        extra ``tokenizer_path`` field.  When omitted, the config's
        parent directory is used as the tokenizer path.
        """
        root = Path(config_path).parent
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        tokenizer_path = raw_config.pop("tokenizer_path", None) or str(root)
        config = PipelineConfig.from_dict(raw_config)
        return cls(config, tokenizer_path)

    def apply(self, records: List[dict]) -> Dict[str, list]:
        """Tokenize a list of raw record dicts.

        Returns a dict mapping key (``sequence``, ``chosen``, ``responses``,
        …) to a list of per-record tensors (or nested tensor lists for
        multi-response keys such as GRPO ``responses``).
        """
        raw: Dict[str, list] = {}
        doc_sequences: List[List[int]] = []

        for item in records:
            result = self.mask_builder.build(item, self.config, self.tokenizer)
            if result is None:
                continue
            result.pop("domain", None)
            primary_ids = self._primary_ids(result)
            if not primary_ids:
                continue
            doc_sequences.append(primary_ids)
            for key, ids in result.items():
                if key not in raw:
                    raw[key] = []
                if ids and isinstance(ids[0], list):
                    raw[key].append(
                        [torch.tensor(sub, dtype=self._infer_dtype(sub)) for sub in ids]
                    )
                else:
                    raw[key].append(torch.tensor(ids, dtype=self._infer_dtype(ids)))

        pos_ids = self.position_strategy.generate(doc_sequences)
        if pos_ids:
            raw["position_ids"] = [torch.tensor(pos_ids, dtype=torch.int32)]

        return raw

    @staticmethod
    def _primary_ids(result: dict) -> List[int]:
        for val in result.values():
            if isinstance(val, list) and val and isinstance(val[0], int):
                return val
        return []

    @staticmethod
    def _infer_dtype(ids: List) -> torch.dtype:
        if ids and isinstance(ids[0], float):
            return torch.float32
        return torch.int32
