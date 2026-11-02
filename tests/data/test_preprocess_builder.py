import pytest

from astrai.config.preprocess_config import (
    InputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing.core import prepare_sequences
from tests.support.data import (
    TEXT_SECTIONS,
    make_dpo_chat_config,
    make_grpo_config,
    make_pipeline_config,
)

_CHAT_SECTIONS = [{"field": "messages", "action": "$role", "template": True}]
_INSTRUCTION_SECTIONS = [
    {"field": "prompt", "action": "mask", "add_special_tokens": True},
    {"field": "response", "action": "train"},
]


def _make_chat_config():
    return make_pipeline_config(
        _CHAT_SECTIONS,
        mask={"system": "mask", "user": "mask", "assistant": "train"},
    )


def _make_instruction_config():
    return make_pipeline_config(
        _INSTRUCTION_SECTIONS,
        mask={"prompt": "mask", "response": "train"},
    )


def test_chat_mask_only_assistant(chat_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=_CHAT_SECTIONS), mask={"assistant": "train"}
    )
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    result = builder.build({"messages": messages}, config, chat_tokenizer)
    mask = result["loss_mask"]

    def template_ids(message):
        rendered = chat_tokenizer.apply_chat_template(
            [message], tokenize=False, add_generation_prompt=False
        )
        return chat_tokenizer.encode(rendered, add_special_tokens=False)

    user_len = len(template_ids(messages[0]))
    assistant_len = len(template_ids(messages[1]))
    bos = 1 if chat_tokenizer.bos_token_id is not None else 0

    assert len(mask) == bos + user_len + assistant_len
    assert all(m == 0 for m in mask[: bos + user_len])
    assert all(m == 1 for m in mask[bos + user_len :])


def test_chat_batch_matches_single(chat_tokenizer, builder):
    config = _make_chat_config()
    items = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello."},
            ]
        },
    ]
    batch = builder.build_batch(items, config, chat_tokenizer)
    single = [builder.build(item, config, chat_tokenizer) for item in items]
    assert batch == single


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
    config = _make_chat_config()
    assert builder.build({"messages": []}, config, chat_tokenizer) is None
    assert builder.build({}, config, chat_tokenizer) is None


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
    assert len(result["sequence"]) > 10
    packed = prepare_sequences(
        {k: [v] for k, v in result.items() if k != "domain"}, config
    )
    assert len(packed["sequence"][0]) == 10
    assert len(result["loss_mask"]) == len(result["sequence"])


def test_instruction_batch_matches_single(test_tokenizer, builder):
    config = _make_instruction_config()
    items = [
        {"prompt": "Translate to French: Hello", "response": "Bonjour"},
        {"prompt": "Translate to German: Hello", "response": "Hallo"},
    ]
    assert builder.build_batch(items, config, test_tokenizer) == [
        builder.build(item, config, test_tokenizer) for item in items
    ]


def test_instruction_prompt_masked(test_tokenizer, builder):
    config = _make_instruction_config()
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


def test_text_empty(test_tokenizer, builder):
    config = make_pipeline_config(
        TEXT_SECTIONS, preprocessing=ProcessingConfig(min_chars=1)
    )
    assert builder.build({"text": ""}, config, test_tokenizer) is None
    assert builder.build({"text": "   "}, config, test_tokenizer) is None


def test_text_too_short(test_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=TEXT_SECTIONS),
        preprocessing=ProcessingConfig(min_chars=100),
    )
    assert builder.build({"text": "short"}, config, test_tokenizer) is None


def test_text_truncation(test_tokenizer, builder):
    config = PipelineConfig(
        input=InputConfig(sections=TEXT_SECTIONS),
        preprocessing=ProcessingConfig(max_seq_len=3, min_chars=1),
    )
    item = {"text": "This is a very long text that should be truncated"}
    result = builder.build(item, config, test_tokenizer)
    assert len(result["sequence"]) > 3
    packed = prepare_sequences(
        {k: [v] for k, v in result.items() if k != "domain"}, config
    )
    assert len(packed["sequence"][0]) == 3


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


@pytest.mark.parametrize("missing", ["chosen", "rejected"])
def test_dpo_partial_record_is_none(chat_tokenizer, builder, missing):
    config = make_dpo_chat_config()
    item = {
        "chosen": [{"role": "assistant", "content": "Good"}],
        "rejected": [{"role": "assistant", "content": "Bad"}],
    }
    item.pop(missing)

    assert builder.build(item, config, chat_tokenizer) is None
    assert builder.build_batch([item], config, chat_tokenizer) == [None]


def test_grpo_batch_matches_single(chat_tokenizer, builder):
    config = make_grpo_config()
    items = [
        {
            "prompt": [{"role": "user", "content": "What is 2+2?"}],
            "responses": ["4", "5"],
            "rewards": [1.0, 0.0],
        },
        {
            "prompt": [{"role": "user", "content": "Say hello."}],
            "responses": ["Hello", "Hi"],
            "rewards": [1.0, 0.5],
        },
    ]
    assert builder.build_batch(items, config, chat_tokenizer) == [
        builder.build(item, config, chat_tokenizer) for item in items
    ]


def test_grpo_response_tokens_all_trained(chat_tokenizer, builder):
    config = make_grpo_config()
    item = {
        "prompt": [{"role": "user", "content": "Q"}],
        "responses": ["A", "B"],
        "rewards": [0.8, 0.2],
    }
    result = builder.build(item, config, chat_tokenizer)
    masks = result["masks"]
    # masks is List[List[int]] — each response's mask should be all 1s
    assert len(masks) == 2
    for m in masks:
        assert all(v == 1 for v in m)
        assert len(m) == len(result["responses"][masks.index(m)])


def test_grpo_single_reward(chat_tokenizer, builder):
    config = make_grpo_config()
    item = {
        "prompt": [{"role": "user", "content": "Q"}],
        "responses": ["A"],
        "rewards": 0.9,
    }
    result = builder.build(item, config, chat_tokenizer)
    assert result["rewards"] == [0.9]
