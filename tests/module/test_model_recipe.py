import json
from pathlib import Path


def test_12b_mqa_moe_recipe_parameter_count():
    recipe_path = (
        Path(__file__).parents[2]
        / "recipes"
        / "astrai-12b-mqa-moe"
        / "config.json"
    )
    config = json.loads(recipe_path.read_text(encoding="utf-8"))

    dim = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    head_dim = dim // config["num_attention_heads"]
    embedding = 2 * config["vocab_size"] * dim
    attention = (
        2 * dim * dim + 2 * dim * config["num_key_value_heads"] * head_dim
    )
    expert = 3 * dim * config["intermediate_size"]
    router = dim * config["n_routed_experts"]
    total_ffn = expert * (
        config["n_routed_experts"] + config["n_shared_experts"]
    ) + router
    active_ffn = expert * (
        config["n_activated_experts"] + config["n_shared_experts"]
    ) + router
    total = embedding + n_layers * (attention + 2 * dim + total_ffn) + dim
    active = embedding + n_layers * (attention + 2 * dim + active_ffn) + dim

    assert config["num_key_value_heads"] == 1
    assert total == 12_154_702_848
    assert active == 3_170_503_680
