"""Tokenization transform for JSONL record streams.

Bridges the Reader layer (``JsonlStore`` reads raw JSON records) and the
Dataset layer (expects per-record tensors).  Holds the tokenizer,
mask-builder and position-id strategy together so that I/O code stays
free of model dependencies.

The record-processing core (mask building, primary-id extraction,
per-key tensorisation, position-id generation) is shared with
:class:`astrai.preprocessing.pipeline.Pipeline` via the
:mod:`astrai.preprocessing.core` helpers.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch

from astrai.config.preprocess_config import PipelineConfig
from astrai.preprocessing.core import (
    build_position_ids,
    build_preprocessing_components,
    iter_raw_records,
    to_per_record_tensors,
)


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
        self.tokenizer, self.mask_builder, self.position_strategy = (
            build_preprocessing_components(config, tokenizer_path)
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

        for result in iter_raw_records(
            records, self.mask_builder, self.config, self.tokenizer
        ):
            primary = None
            for val in result.values():
                if isinstance(val, list) and val and isinstance(val[0], int):
                    primary = val
                    break
            if primary is not None:
                doc_sequences.append(primary)
            for key, ids in result.items():
                raw.setdefault(key, []).append(ids)

        tensors = to_per_record_tensors(raw)

        pos_ids = build_position_ids(doc_sequences, self.position_strategy)
        if pos_ids is not None:
            tensors["position_ids"] = [torch.tensor(pos_ids, dtype=torch.int32)]

        return tensors
