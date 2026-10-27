"""Pipeline configuration for JSONL preprocessing.

Supports single-sequence (SFT/pretrain) and multi-output (DPO/GRPO)
modes, both driven declaratively through ``input.sections`` or
``input.sources``.
"""

from dataclasses import field
from typing import Dict, List, Optional

from pydantic import field_validator
from pydantic.dataclasses import dataclass

from astrai.config.base import BaseConfig

_PACKING_STRATEGIES = frozenset({"simple", "bfd", "bfd_split"})
_TRUNCATION_MODES = frozenset({"keep_start", "keep_end"})
_POSITION_IDS_MODES = frozenset({"none", "doc_reset", "continuous"})


@dataclass
class InputConfig(BaseConfig):
    """Declarative input mapping.

    Single-output mode (backward-compatible)::

        {"input": {"sections": [{"field": "messages", ...}]}}

    Multi-output mode (DPO / GRPO)::

        {"input": {"sources": {
            "chosen": {"sections": [{"field": "chosen", ...}]},
            "rejected": {"sections": [{"field": "rejected", ...}]},
        }}}

    Args:
        sections (Optional[List[Dict]]): Section list for single-output mode. Defaults to None.
        sources (Optional[Dict[str, Dict]]): Source map for multi-output mode, DPO/GRPO. Defaults to None.
    """

    sections: Optional[List[Dict]] = None
    sources: Optional[Dict[str, Dict]] = None


@dataclass
class ProcessingConfig(BaseConfig):
    """Processing configuration for tokenization and packing.

    Args:
        max_seq_len (int): Maximum sequence length. Defaults to 2048.
        min_chars (int): Minimum number of characters to keep. Defaults to 50.
        max_chars (int): Maximum number of characters to keep. Defaults to 2_000_000.
        max_items (Optional[int]): Maximum number of items to process, None=unlimited. Defaults to None.
        batch_size (int): Number of records tokenized together. Defaults to 256.
        packing_strategy (str): How to pack sequences: 'simple', 'bfd', or 'bfd_split'. Defaults to "simple".
        max_packed_len (int): Maximum length of a packed bin. Defaults to 8192.
        truncation_mode (str): How to truncate over-length sequences: 'keep_start' or 'keep_end'. Defaults to "keep_start".
    """

    max_seq_len: int = 2048
    min_chars: int = 50
    max_chars: int = 2_000_000
    max_items: Optional[int] = None
    batch_size: int = 256
    packing_strategy: str = "simple"
    max_packed_len: int = 8192
    truncation_mode: str = "keep_start"

    @field_validator("packing_strategy")
    def _validate_packing_strategy(cls, v: str) -> str:
        if v not in _PACKING_STRATEGIES:
            raise ValueError(
                f"packing_strategy must be one of {sorted(_PACKING_STRATEGIES)}, got {v!r}"
            )
        return v

    @field_validator("truncation_mode")
    def _validate_truncation_mode(cls, v: str) -> str:
        if v not in _TRUNCATION_MODES:
            raise ValueError(
                f"truncation_mode must be one of {sorted(_TRUNCATION_MODES)}, got {v!r}"
            )
        return v

    @field_validator("max_seq_len", "batch_size", "max_packed_len")
    def _validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be positive, got {v}")
        return v

    @field_validator("min_chars")
    def _validate_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"min_chars must be non-negative, got {v}")
        return v


@dataclass
class OutputConfig(BaseConfig):
    """Output configuration for storage.

    Args:
        domain_key (Optional[str]): Domain key for the output store. Defaults to None.
        storage_format (str): Pipeline output format: 'bin'. Defaults to "bin".
        max_tokens_per_shard (int): Maximum tokens per shard before splitting. Defaults to 100_000_000.
        dtype (Dict[str, str]): Per-key dtype overrides, e.g. {"input_ids": "int32"}. Defaults to {}.
        position_ids_mode (str): Position ids mode: 'none', 'doc_reset', or 'continuous'. Defaults to "doc_reset".
        record_keys (List[str]): Fields stored with per-record offsets. Defaults to an empty list.
    """

    domain_key: Optional[str] = None
    storage_format: str = "bin"
    max_tokens_per_shard: int = 100_000_000
    dtype: Dict[str, str] = field(default_factory=dict)
    position_ids_mode: str = "doc_reset"
    record_keys: List[str] = field(default_factory=list)

    @field_validator("record_keys")
    def _validate_record_keys(cls, value: List[str]) -> List[str]:
        if any(not key.strip() for key in value) or len(set(value)) != len(value):
            raise ValueError("record_keys must contain unique, non-empty field names")
        return value

    @field_validator("storage_format")
    def _validate_storage_format(cls, v: str) -> str:
        if v != "bin":
            raise ValueError("Pipeline output storage_format only supports 'bin'")
        return v

    @field_validator("position_ids_mode")
    def _validate_position_ids_mode(cls, v: str) -> str:
        if v not in _POSITION_IDS_MODES:
            raise ValueError(
                f"position_ids_mode must be one of {sorted(_POSITION_IDS_MODES)}, got {v!r}"
            )
        return v


@dataclass
class PipelineConfig(BaseConfig):
    """Top-level preprocessing pipeline config.

    Args:
        version (int): Config schema version. Defaults to 1.
        input (InputConfig): Input mapping config.
        mask (Dict[str, str]): Per-field mask labels, e.g. {"system": "mask", "assistant": "train"}. Defaults to {}.
        mask_default (str): Default mask label for unlisted fields. Defaults to "mask".
        preprocessing (ProcessingConfig): Processing config.
        output (OutputConfig): Output config.
    """

    version: int = 1
    input: InputConfig = field(default_factory=InputConfig)
    mask: Dict[str, str] = field(default_factory=dict)
    mask_default: str = "mask"
    preprocessing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
