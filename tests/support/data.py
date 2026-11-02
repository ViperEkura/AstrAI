"""Test data builders for preprocessing and dataset scenarios."""

from astrai.config.preprocess_config import (
    InputConfig,
    PipelineConfig,
    ProcessingConfig,
)

TEXT_SECTIONS = [{"field": "text", "action": "train"}]


def make_pipeline_config(sections, *, mask=None, preprocessing=None, sources=None):
    """Build a pipeline config with the common test defaults."""
    return PipelineConfig(
        input=InputConfig(sections=sections, sources=sources),
        mask={} if mask is None else mask,
        mask_default="mask",
        preprocessing=preprocessing or ProcessingConfig(max_seq_len=2048),
    )


def make_dpo_chat_config():
    sources = {
        name: {"sections": [{"field": name, "action": "$role", "template": True}]}
        for name in ("chosen", "rejected")
    }
    return make_pipeline_config(
        None,
        mask={"user": "mask", "assistant": "train"},
        sources=sources,
    )


def make_grpo_config(*, template=True):
    prompt_section = {"field": "prompt", "action": "mask"}
    if template:
        prompt_section["template"] = True
    else:
        prompt_section["add_special_tokens"] = True
    sources = {
        "prompts": {"sections": [prompt_section]},
        "responses": {
            "sections": [{"field": "responses", "action": "train"}],
            "list_field": True,
            "mask_key": "masks",
        },
        "rewards": {"sections": [{"field": "rewards", "action": "value"}]},
    }
    return make_pipeline_config(
        None,
        mask={"user": "mask", "assistant": "train"},
        sources=sources,
    )
