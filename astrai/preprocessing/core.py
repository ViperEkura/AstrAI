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
from typing import Dict, Iterator, List

import torch

from astrai.config.preprocess_config import PipelineConfig
from astrai.preprocessing.builder import SectionedMaskBuilder
from astrai.preprocessing.packing import pack_sequences
from astrai.tokenize import AutoTokenizer


def build_preprocessing_components(tokenizer_path: str):
    """Load the tokenizer and built-in section builder for both input paths."""
    return AutoTokenizer.from_pretrained(tokenizer_path), SectionedMaskBuilder()


def primary_ids(result: Dict) -> List[int]:
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
    records: List[Dict],
    mask_builder,
    config: PipelineConfig,
    tokenizer,
) -> Iterator[Dict]:
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
    raw: Dict[str, List],
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


def accumulate_record(bucket: Dict, result: Dict, ids: List[int]) -> None:
    """Fill omitted all-ones masks in both earlier and incoming records."""
    for key in result.keys() - bucket.keys():
        fill = 1 if key == "loss_mask" or key.endswith("_mask") else 0
        bucket[key] = [[fill] * len(seq) for seq in bucket.get("sequence", [])]
    for key in bucket:
        fill = 1 if key == "loss_mask" or key.endswith("_mask") else 0
        bucket[key].append(result.get(key, [fill] * len(ids)))


def prepare_sequences(keys: Dict, config: PipelineConfig) -> Dict:
    """Shared single-output overflow, packing, and boundary representation."""
    if "sequence" not in keys:
        return keys
    pp = config.preprocessing
    record_mode = "seq_lengths" in config.output.record_keys
    if record_mode:
        required = {"sequence", "loss_mask", "seq_lengths"}
        if not required.issubset(config.output.record_keys):
            raise ValueError(
                "SFT records require sequence, loss_mask, seq_lengths offsets"
            )
        if config.output.position_ids_mode != "doc_reset":
            raise ValueError("SFT records require position_ids_mode=doc_reset")
        keys = dict(keys)
        keys.setdefault("loss_mask", [[1] * len(seq) for seq in keys["sequence"]])
    packed = pack_sequences(
        keys,
        pp.max_seq_len,
        pp.max_packed_len,
        overflow=pp.overflow,
        packing=pp.packing,
        truncation_mode=pp.truncation_mode,
    )
    if record_mode:
        return packed
    lengths = packed.pop("seq_lengths")
    mode = config.output.position_ids_mode
    if mode == "doc_reset":
        packed["position_ids"] = [
            [position for length in segments for position in range(length)]
            for segments in lengths
        ]
    elif mode == "continuous":
        start = 0
        packed["position_ids"] = []
        for segments in lengths:
            end = start + sum(segments)
            packed["position_ids"].append(list(range(start, end)))
            start = end
    return packed
