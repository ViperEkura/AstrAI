"""Inference engine configuration."""

import json
from pathlib import Path
from typing import Dict, Union

from astrai.config.base import BaseConfig


def load_generation_config(path: Union[str, Path]) -> Dict:
    """Read HF generation defaults, falling back to HF model-config fields."""
    path = Path(path)
    file = path / "generation_config.json"
    if file.is_file():
        return json.loads(file.read_text(encoding="utf-8"))
    file = path / "config.json"
    config = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}
    text_config = config.get("text_config", config)
    fields = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "max_length",
        "max_new_tokens",
        "num_beams",
        "num_return_sequences",
        "repetition_penalty",
        "min_length",
        "min_new_tokens",
        "stop_strings",
    )
    return {name: text_config[name] for name in fields if name in text_config}


def normalize_eos_token_ids(value):
    if value is None:
        return ()
    ids = [value] if type(value) is int else value
    if not isinstance(ids, (list, tuple)) or any(
        type(i) is not int or i < 0 for i in ids
    ):
        raise ValueError("eos_token_id must be a non-negative integer, a list, or None")
    return tuple(dict.fromkeys(ids))


def resolve_generation_parameters(
    defaults: Dict, overrides: Dict, prompt_length: int, max_seq_len: int
) -> Dict:
    """Resolve HF greedy/sampling parameters once, before request admission."""
    config = {**defaults, **overrides}
    for name, supported in (
        ("num_beams", 1),
        ("num_return_sequences", 1),
        ("repetition_penalty", 1.0),
        ("min_length", 0),
        ("min_new_tokens", 0),
    ):
        value = config.get(name, supported)
        if value not in (supported, None):
            raise ValueError(f"Unsupported generation parameter: {name}={value!r}")
    if config.get("stop_strings"):
        raise ValueError(
            "stop_strings are not supported by the tokenized core; use frontend stop_sequences"
        )
    do_sample = config.get("do_sample", False)
    if type(do_sample) is not bool:
        raise ValueError("do_sample must be a boolean")
    temperature = config.get("temperature", 1.0)
    top_p, top_k = config.get("top_p", 1.0), config.get("top_k", 50)
    if do_sample and (not isinstance(temperature, (int, float)) or temperature <= 0):
        raise ValueError("temperature must be positive when do_sample=True")
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    budget = config.get("max_new_tokens")
    if budget is None:
        length = config.get("max_length", 20)
        if type(length) is not int or length < 0:
            raise ValueError("max_length must be a non-negative integer")
        budget = max(0, length - prompt_length) if "max_length" in config else 20
    if type(budget) is not int or budget < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    return {
        "max_tokens": min(budget, max_seq_len - prompt_length),
        "temperature": float(temperature) if do_sample else 0.0,
        "top_p": float(top_p),
        "top_k": top_k,
        "eos_token_ids": normalize_eos_token_ids(config.get("eos_token_id")),
    }


class InferenceConfig(BaseConfig):
    """Configuration for inference workspace and execution parameters.

    Centralizes magic constants previously scattered across inference modules.

    Args:
        max_splits (int): Maximum number of splits for split-KV attention (decode partial results). Defaults to 32.
        q_tile_rows (int): Number of rows per Q tile in prefill ragged batching. Defaults to 64.
        prefill_warmup_len (int): Prompt length for prefill warmup (cuBLAS auto-tuning). Defaults to 64.
        default_rep_window (int): Default repetition penalty window size for frequency penalty. Defaults to 64.
        max_recent_tasks (int): Maximum number of recent tasks tracked for aggregate statistics. Defaults to 128.
        max_num_batched_tokens (int): Token budget per forward for chunked prefill; 0 disables chunking (whole-prompt prefills). Defaults to 0.
    """

    max_splits: int = 32
    q_tile_rows: int = 64
    prefill_warmup_len: int = 64
    default_rep_window: int = 64
    max_recent_tasks: int = 128
    # Chunked-prefill token budget: one forward's prefill windows plus
    # decode steps never exceed this.  None (0) disables chunking.
    max_num_batched_tokens: int = 0
