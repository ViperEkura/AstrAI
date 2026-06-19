import pytest

from astrai.config.preprocess_config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing.builder import (
    MaskBuilderFactory,
    SectionedMaskBuilder,
)
from tests.data.conftest import (
    _CHAT_SECTIONS,
    _INSTRUCTION_SECTIONS,
    _TEXT_SECTIONS,
    make_chat_config,
    make_dpo_chat_config,
    make_grpo_config,
    make_instruction_config,
    make_text_config,
)


def test_chat_simple(chat_tokenizer, builder):
    config = make_chat_config()
    item = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    result = builder.build(item, config, chat_tokenizer)
    assert result is not None
    assert "sequence" in result
    assert "loss_mask" in result
    assert len(result["sequence"]) == len(result["loss_mask"])

    ids = chat_tokenizer.decode(result["sequence"], skip_special_tokens=False)
    assert "system" in ids.lower() or "<|im_start|>system" in ids
    assert "assistant" in ids.lower() or "<|im_start|>assistant" in ids

    total = len(result["sequence"])
    trained = sum(result["loss_mask"])
    assert trained > 0
    assert trained < total


def test_chat_mask_only_assistant(chat_tokenizer, builder):
    config = make_chat_config()
    item = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    result = builder.build(item, config, chat_tokenizer)
    mask = result["loss_mask"]
    ids = result["sequence"]
    assert len(ids) == len(mask)

    trained = [i for i, m in enumerate(mask) if m == 1]
    masked = [i for i, m in enumerate(mask) if m == 0]
    assert len(trained) > 0
    assert len(masked) > 0


@pytest.mark.parametrize(
    "mask_rules,mask_default,expect_nonzero",
    [
        ({"system": "mask", "user": "mask", "assistant": "mask"}, "mask", False),
        ({}, "train", True),
    ],
)
def test_chat_uniform_masking(
    mask_rules, mask_default, expect_nonzero, chat_tokenizer, builder
):
    config = PipelineConfig(
        input=InputConfig(sections=_CHAT_SECTIONS),
        mask=mask_rules,
        mask_default=mask_default,
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )
    item = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }
    result = builder.build(item, config, chat_tokenizer)
    masked_count = sum(result["loss_mask"])
    if expect_nonzero:
        assert masked_count > 0
    else:
        assert masked_count == 0


def test_chat_empty_messages(chat_tokenizer, builder):
    config = make_chat_config()
    assert builder.build({"messages": []}, config, chat_tokenizer) is None
    assert builder.build({}, config, chat_tokenizer) is None


def test_chat_domain_extraction(chat_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_CHAT_SECTIONS),
        mask={"assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
        output=OutputConfig(domain_key="source"),
    )
    item = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        "source": "wiki",
    }
    result = builder.build(item, config, chat_tokenizer)
    assert result["domain"] == "wiki"


def test_chat_truncation(chat_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_CHAT_SECTIONS),
        mask={"assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=10),
    )
    item = {
        "messages": [
            {
                "role": "user",
                "content": "Tell me a very long story about dragons and knights and magic.",
            },
            {"role": "assistant", "content": "Sure! Here is a tale..."},
        ]
    }
    result = builder.build(item, config, chat_tokenizer)
    assert len(result["sequence"]) <= 10
    assert len(result["loss_mask"]) == len(result["sequence"])


def test_instruction_basic(test_tokenizer, builder):
    config = make_instruction_config()
    item = {"prompt": "Translate to French: Hello", "response": "Bonjour"}
    result = builder.build(item, config, test_tokenizer)
    assert result is not None
    assert len(result["sequence"]) == len(result["loss_mask"])


def test_instruction_prompt_masked(test_tokenizer, builder):
    config = make_instruction_config()
    item = {"prompt": "hello", "response": "world"}
    result = builder.build(item, config, test_tokenizer)
    mask = result["loss_mask"]
    ids = result["sequence"]

    prompt_ids = test_tokenizer.encode("hello", add_special_tokens=True)
    p_len = min(len(prompt_ids), len(ids))
    assert all(m == 0 for m in mask[:p_len])
    if p_len < len(ids):
        assert all(m == 1 for m in mask[p_len:])


def test_instruction_train_on_prompt(test_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(
            sections=[
                {"field": "prompt", "action": "train", "add_special_tokens": True},
                {"field": "response", "action": "mask"},
            ]
        ),
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )
    item = {"prompt": "hello", "response": "world"}
    result = builder.build(item, config, test_tokenizer)
    mask = result["loss_mask"]
    ids = result["sequence"]

    prompt_ids = test_tokenizer.encode("hello", add_special_tokens=True)
    p_len = min(len(prompt_ids), len(ids))
    assert all(m == 1 for m in mask[:p_len])


def test_text_basic(test_tokenizer, builder):
    config = make_text_config()
    item = {"text": "Hello world. This is a test document."}
    result = builder.build(item, config, test_tokenizer)
    assert result is not None
    assert "sequence" in result
    assert len(result["sequence"]) > 0
    assert "loss_mask" not in result


def test_text_empty(test_tokenizer, builder):
    config = make_text_config()
    assert builder.build({"text": ""}, config, test_tokenizer) is None
    assert builder.build({"text": "   "}, config, test_tokenizer) is None


def test_text_too_short(test_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_TEXT_SECTIONS),
        preprocessing=ProcessingConfig(min_chars=100),
    )
    assert builder.build({"text": "short"}, config, test_tokenizer) is None


def test_text_truncation(test_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_TEXT_SECTIONS),
        preprocessing=ProcessingConfig(max_seq_len=3, min_chars=1),
    )
    item = {"text": "This is a very long text that should be truncated"}
    result = builder.build(item, config, test_tokenizer)
    assert len(result["sequence"]) <= 3


def test_sectioned_chat(chat_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_CHAT_SECTIONS),
        mask={"system": "mask", "user": "mask", "assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )
    item = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    result = builder.build(item, config, chat_tokenizer)
    assert result is not None
    assert len(result["sequence"]) == len(result["loss_mask"])
    assert sum(result["loss_mask"]) > 0
    assert 0 in result["loss_mask"]


def test_sectioned_instruction(test_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_INSTRUCTION_SECTIONS),
        preprocessing=ProcessingConfig(max_seq_len=2048, min_chars=0),
    )
    item = {"prompt": "Q: Why?", "response": "A: Because."}
    result = builder.build(item, config, test_tokenizer)
    assert result is not None
    mask = result["loss_mask"]
    assert mask[0] == 0
    assert mask[-1] == 1


def test_sectioned_text(test_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_TEXT_SECTIONS),
        preprocessing=ProcessingConfig(max_seq_len=2048, min_chars=1),
    )
    item = {"text": "Hello world, this is a test."}
    result = builder.build(item, config, test_tokenizer)
    assert result is not None
    assert "loss_mask" not in result


def test_sectioned_text_too_short(test_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_TEXT_SECTIONS),
        preprocessing=ProcessingConfig(max_seq_len=2048, min_chars=100),
    )
    assert builder.build({"text": "short"}, config, test_tokenizer) is None


def test_factory_registered():
    names = MaskBuilderFactory.list_registered()
    assert "sectioned" in names


def test_factory_create():
    builder_obj = MaskBuilderFactory.create("sectioned")
    assert isinstance(builder_obj, SectionedMaskBuilder)


def test_dpo_chat_basic(chat_tokenizer, builder):
    config = make_dpo_chat_config()
    item = {
        "chosen": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "rejected": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "5"},
        ],
    }
    result = builder.build(item, config, chat_tokenizer)
    assert result is not None
    assert "chosen" in result
    assert "rejected" in result
    assert "chosen_mask" in result
    assert "rejected_mask" in result
    assert len(result["chosen"]) == len(result["chosen_mask"])
    assert len(result["rejected"]) == len(result["rejected_mask"])
    assert sum(result["chosen_mask"]) > 0
    assert sum(result["rejected_mask"]) > 0


def test_dpo_chosen_only_trained(chat_tokenizer, builder):
    config = make_dpo_chat_config()
    item = {
        "chosen": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        "rejected": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Go away"},
        ],
    }
    result = builder.build(item, config, chat_tokenizer)
    assert 0 in result["chosen_mask"]
    assert 1 in result["chosen_mask"]
    assert 0 in result["rejected_mask"]
    assert 1 in result["rejected_mask"]


def test_dpo_missing_field_is_none(chat_tokenizer, builder):
    config = make_dpo_chat_config()
    assert builder.build({"chosen": [], "rejected": []}, config, chat_tokenizer) is None


def test_grpo_basic(chat_tokenizer, builder):
    config = make_grpo_config()
    item = {
        "prompt": [{"role": "user", "content": "What is 2+2?"}],
        "responses": ["4", "The answer is four", "Four", "2+2=4"],
        "rewards": [1.0, 0.5, 0.8, 0.2],
    }
    result = builder.build(item, config, chat_tokenizer)
    assert result is not None
    assert "prompts" in result
    assert "responses" in result
    assert "masks" in result
    assert "rewards" in result
    assert len(result["responses"]) == len(result["masks"])
    assert result["rewards"] == [1.0, 0.5, 0.8, 0.2]


def test_grpo_response_tokens_all_trained(chat_tokenizer, builder):
    config = make_grpo_config()
    item = {
        "prompt": [{"role": "user", "content": "Q"}],
        "responses": ["A", "B"],
        "rewards": [0.8, 0.2],
    }
    result = builder.build(item, config, chat_tokenizer)
    masks = result["masks"]
    assert all(m == 1 for m in masks)
    assert len(masks) == len(result["responses"])


def test_grpo_single_reward(chat_tokenizer, builder):
    config = make_grpo_config()
    item = {
        "prompt": [{"role": "user", "content": "Q"}],
        "responses": ["A"],
        "rewards": 0.9,
    }
    result = builder.build(item, config, chat_tokenizer)
    assert result["rewards"] == [0.9]
