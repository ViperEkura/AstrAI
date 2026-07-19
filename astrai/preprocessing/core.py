"""Shared preprocessing kernel used by both :class:`Pipeline` and
:class:`TokenizeTransform`.

The two entry points previously duplicated ~60 % of their logic:
record iteration, mask-builder invocation, primary-id extraction,
per-key accumulation, dtype inference and position-id generation.
This module factors out the common core as pure functions so that
the online (``TokenizeTransform``) and offline (``Pipeline``) paths
stay in lockstep.
"""

from itertools import chain
from typing import Dict, Iterator, List, Optional

import torch

from astrai.config.preprocess_config import PipelineConfig
from astrai.preprocessing.builder import MaskBuilderFactory
from astrai.preprocessing.position_id import PositionIdStrategyFactory
from astrai.tokenize import AutoTokenizer


def build_preprocessing_components(config: PipelineConfig, tokenizer_path: str):
    """Load tokenizer, mask builder and position-id strategy together.

    Both ``Pipeline`` and ``TokenizeTransform`` need the same triple;
    centralising the construction avoids drift (e.g. one path forgetting
    to create the position-id strategy).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    mask_builder = MaskBuilderFactory.create("sectioned")
    position_strategy = PositionIdStrategyFactory.create(
        config.output.position_ids_mode
    )
    return tokenizer, mask_builder, position_strategy


def primary_ids(result: dict) -> List[int]:
    """Return the first flat int-list value in *result*.

    Used for token counting and position-id generation when the
    primary key name is not known (DPO uses ``chosen``, GRPO uses
    ``prompts``, SFT uses ``sequence``).
    """
    for val in result.values():
        if isinstance(val, list) and val and isinstance(val[0], int):
            return val
    return []


def infer_dtype(ids: List) -> torch.dtype:
    """Float values become float32, everything else int32."""
    if ids and isinstance(ids[0], float):
        return torch.float32
    return torch.int32


def iter_raw_records(
    records: List[dict],
    mask_builder,
    config: PipelineConfig,
    tokenizer,
) -> Iterator[dict]:
    """Yield mask-builder output dicts for each record, skipping failures.

    Drops ``domain`` from the result (callers that need it should read
    it before calling this).  Each yielded dict maps a key
    (``sequence``, ``chosen``, ``responses``…) to either a flat
    ``List[int]`` or a nested ``List[List[int]]`` (GRPO responses/masks).
    """
    for item in records:
        result = mask_builder.build(item, config, tokenizer)
        if result is None:
            continue
        result.pop("domain", None)
        if not primary_ids(result):
            continue
        yield result


def to_per_record_tensors(
    raw: Dict[str, list],
) -> Dict[str, List[torch.Tensor]]:
    """Convert an accumulated ``{key: [per-record ids]}`` dict to tensors.

    Handles three shapes transparently:

    - ``List[int]`` per record (``sequence``, ``chosen``…) → one tensor per record.
    - ``List[List[int]]`` per record (GRPO ``responses``/``masks``) → one
      ``List[Tensor]`` per record (nested), preserving the per-response
      boundary so downstream code can index responses individually.
    - ``List[int]`` for the whole shard (pre-packed keys) → single tensor.

    The detection mirrors the previous inline logic in
    ``Pipeline._flush`` and ``TokenizeTransform.apply``.
    """
    tensors: Dict[str, List[torch.Tensor]] = {}
    for key, ids_list in raw.items():
        if ids_list and isinstance(ids_list[0], list):
            tensors[key] = [
                [torch.tensor(sub, dtype=infer_dtype(sub)) for sub in ids]
                if ids and isinstance(ids[0], list)
                else torch.tensor(ids, dtype=infer_dtype(ids))
                for ids in ids_list
            ]
        else:
            tensors[key] = [
                torch.tensor(list(chain.from_iterable(ids_list)), dtype=torch.int32)
            ]
    return tensors


def build_position_ids(
    sequences: List[List[int]],
    strategy,
) -> Optional[List[int]]:
    """Generate position ids for *sequences* using *strategy*.

    Returns ``None`` when the strategy produces no ids (e.g. ``none``
    mode), so callers can skip attaching the key instead of storing
    an empty list.
    """
    pos_ids = strategy.generate(sequences)
    return pos_ids or None
