"""Inference engine configuration."""

from astrai.config.base import BaseConfig


class InferenceConfig(BaseConfig):
    """Configuration for inference workspace and execution parameters.

    Centralizes magic constants previously scattered across inference modules.

    Args:
        max_splits (int): Maximum number of splits for split-KV attention (decode partial results). Defaults to 32.
        q_tile_rows (int): Number of rows per Q tile in prefill ragged batching. Defaults to 64.
        prefill_warmup_len (int): Prompt length for prefill warmup (cuBLAS auto-tuning). Defaults to 64.
        default_rep_window (int): Default repetition penalty window size for frequency penalty. Defaults to 64.
        max_recent_tasks (int): Maximum number of recent tasks tracked for aggregate statistics. Defaults to 128.
    """

    max_splits: int = 32
    q_tile_rows: int = 64
    prefill_warmup_len: int = 64
    default_rep_window: int = 64
    max_recent_tasks: int = 128
