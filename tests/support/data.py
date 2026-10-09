"""Test data builders for preprocessing and dataset scenarios."""

from astrai.config.preprocess_config import (
    InputConfig,
    PipelineConfig,
    ProcessingConfig,
)

CHAT_SECTIONS = [{"field": "messages", "action": "$role", "template": True}]
INSTRUCTION_SECTIONS = [
    {"field": "prompt", "action": "mask", "add_special_tokens": True},
    {"field": "response", "action": "train"},
]
TEXT_SECTIONS = [{"field": "text", "action": "train"}]
GRPO_RESPONSE_SECTIONS = [{"field": "responses", "action": "train"}]


def make_pipeline_config(sections, *, mask=None, preprocessing=None, sources=None):
    """Build a pipeline config with the common test defaults."""
    return PipelineConfig(
        input=InputConfig(sections=sections, sources=sources),
        mask={} if mask is None else mask,
        mask_default="mask",
        preprocessing=preprocessing or ProcessingConfig(max_seq_len=2048),
    )


def make_chat_config():
    return make_pipeline_config(
        CHAT_SECTIONS,
        mask={"system": "mask", "user": "mask", "assistant": "train"},
    )


def make_instruction_config():
    return make_pipeline_config(
        INSTRUCTION_SECTIONS,
        mask={"prompt": "mask", "response": "train"},
    )


def make_text_config():
    return make_pipeline_config(
        TEXT_SECTIONS,
        preprocessing=ProcessingConfig(
            max_seq_len=2048, min_chars=1, max_chars=2_000_000
        ),
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
            "sections": GRPO_RESPONSE_SECTIONS,
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


def make_grpo_no_template_config():
    return make_grpo_config(template=False)
