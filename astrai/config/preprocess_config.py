"""Pipeline configuration for JSONL preprocessing.

Supports single-sequence (SFT/pretrain) and multi-output (DPO/GRPO)
modes, both driven declaratively through ``input.sections`` or
``input.sources``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from astrai.config.base import BaseConfig


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
    """

    sections: Optional[List[Dict]] = None
    sources: Optional[Dict[str, Dict]] = None


@dataclass
class ProcessingConfig(BaseConfig):
    """Processing configuration.

    Parameters
    ----------
    max_seq_len : int
        Maximum sequence length (default: 2048).
    min_chars : int
        Minimum number of characters to keep (default: 50).
    max_chars : int
        Maximum number of characters to keep (default: 2_000_000).
    max_items : Optional[int]
        Maximum number of items to process (default: None, unlimited).
    packing_strategy : str
        How to pack sequences into a contiguous stream.

        - ``"simple"``: sequential concatenation (default, backward compatible).
        - ``"bfd"``: best-fit decreasing bin packing, minimises wasted tokens.
        - ``"bfd_split"``: BFD with over-length sequences split into chunks.
    max_packed_len : int
        Maximum length of a packed bin. Sequences longer than this are
        truncated or split depending on ``packing_strategy`` (default: 8192).
    truncation_mode : str
        How to truncate sequences longer than ``max_packed_len``.

        - ``"keep_start"``: keep the first ``max_packed_len`` tokens (default).
        - ``"keep_end"``: keep the last ``max_packed_len`` tokens.
    """

    max_seq_len: int = 2048
    min_chars: int = 50
    max_chars: int = 2_000_000
    max_items: Optional[int] = None
    packing_strategy: str = "simple"
    max_packed_len: int = 8192
    truncation_mode: str = "keep_start"


@dataclass
class OutputConfig(BaseConfig):
    """Output configuration.

    Parameters
    ----------
    domain_key : Optional[str]
        Domain key for the output store (default: None).
    storage_format : str
        Storage format, one of ``"bin"``, ``"jsonl"`` (default: ``"bin"``).
    max_tokens_per_shard : int
        Maximum tokens per shard before splitting (default: 100_000_000).
    dtype : Dict[str, str]
        Per-key dtype overrides, e.g. ``{"input_ids": "int32"}`` (default: {}).
    position_ids_mode : Optional[str]
        How to compute position_ids in packed sequences.

         - ``"none"``: do not generate (default).
        - ``"doc_reset"``: reset to 0 at each document boundary.
        - ``"continuous"``: sequential 0, 1, 2, ... (pretrain, single doc).
    """

    domain_key: Optional[str] = None
    storage_format: str = "bin"
    max_tokens_per_shard: int = 100_000_000
    dtype: Dict[str, str] = field(default_factory=dict)
    position_ids_mode: str = "doc_reset"


@dataclass
class PipelineConfig(BaseConfig):
    version: int = 1
    input: InputConfig = field(default_factory=InputConfig)
    mask: Dict[str, str] = field(default_factory=dict)
    mask_default: str = "mask"
    preprocessing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
