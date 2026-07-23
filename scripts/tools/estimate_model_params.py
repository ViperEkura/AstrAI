"""Analytically estimate AstrAI autoregressive model parameter counts."""

import argparse
import json
from pathlib import Path


def estimate(config: dict) -> tuple[int, int]:
    dim = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    head_dim = dim // config["num_attention_heads"]
    embedding = config["vocab_size"] * dim
    if config.get("tie_word_embeddings") is not True:
        embedding *= 2

    attention = 2 * dim * dim + 2 * dim * config["num_key_value_heads"] * head_dim
    norms = 2 * dim

    if config.get("ffn_type", "mlp") == "moe":
        expert = 3 * dim * config["intermediate_size"]
        router = dim * config["n_routed_experts"]
        total_ffn = expert * (
            config["n_routed_experts"] + config["n_shared_experts"]
        ) + router
        active_ffn = expert * (
            config["n_activated_experts"] + config["n_shared_experts"]
        ) + router
    else:
        total_ffn = active_ffn = 3 * dim * config["intermediate_size"]

    final_norm = dim
    total = embedding + n_layers * (attention + norms + total_ffn) + final_norm
    active = embedding + n_layers * (attention + norms + active_ffn) + final_norm
    return total, active


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    total, active = estimate(config)
    print(f"total_parameters={total:,} ({total / 1e9:.3f}B)")
    print(f"active_parameters={active:,} ({active / 1e9:.3f}B)")
