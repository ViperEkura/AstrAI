import pytest

from astrai.config.preprocess_config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing.builder import (
    MaskBuilderFactory,
    MultiOutputMaskBuilder,
    SectionedMaskBuilder,
    SingleOutputMaskBuilder,
)
from tests.data.factories import (
    CHAT_SECTIONS,
    TEXT_SECTIONS,
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

    ids = chat_tokenizer.decode(result["sequence"], skip_special_tokens=False)
    assert "system" in ids.lower() or "<|im_start|>system" in ids
    assert "assistant" in ids.lower() or "<|im_start|>assistant" in ids

    total = len(result["sequence"])
    trained = sum(result["loss_mask"])
    assert trained > 0
    assert trained < total


def test_chat_mask_only_assistant(chat_tokenizer, builder):
    config = make_chat_config()
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
    config = make_chat_config()
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
        input=InputConfig(sections=CHAT_SECTIONS),
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
        input=InputConfig(sections=CHAT_SECTIONS),
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
        input=InputConfig(sections=CHAT_SECTIONS),
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


def test_instruction_batch_matches_single(test_tokenizer, builder):
    config = make_instruction_config()
    items = [
        {"prompt": "Translate to French: Hello", "response": "Bonjour"},
        {"prompt": "Translate to German: Hello", "response": "Hallo"},
    ]
    assert builder.build_batch(items, config, test_tokenizer) == [
        builder.build(item, config, test_tokenizer) for item in items
    ]


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
    assert len(result["sequence"]) > 0
    assert "loss_mask" not in result


def test_text_empty(test_tokenizer, builder):
    config = make_text_config()
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
    assert len(result["sequence"]) <= 3


@pytest.mark.parametrize(
    ("name", "builder_cls"),
    [
        ("single", SingleOutputMaskBuilder),
        ("multi", MultiOutputMaskBuilder),
        ("sectioned", SectionedMaskBuilder),
    ],
)
def test_factory_create(name, builder_cls):
    assert name in MaskBuilderFactory.list_registered()
    assert isinstance(MaskBuilderFactory.create(name), builder_cls)


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

    # responses is List[List[int]] — one per response
    assert len(result["responses"]) == 4
    assert all(isinstance(r, list) for r in result["responses"])
    assert all(isinstance(r[0], int) for r in result["responses"])

    # masks is List[List[int]] — one per response, matching length
    assert len(result["masks"]) == 4
    for i in range(4):
        assert len(result["masks"][i]) == len(result["responses"][i])

    assert result["rewards"] == [1.0, 0.5, 0.8, 0.2]


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


def test_single_builder_matches_facade(chat_tokenizer, builder, single_builder):
    config = make_chat_config()
    item = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    facade_result = builder.build(item, config, chat_tokenizer)
    single_result = single_builder.build(item, config, chat_tokenizer)
    assert single_result == facade_result


def test_single_builder_rejects_multi_config(chat_tokenizer, single_builder):
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
    assert single_builder.build(item, config, chat_tokenizer) is None


def test_multi_builder_matches_facade(chat_tokenizer, builder, multi_builder):
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
    facade_result = builder.build(item, config, chat_tokenizer)
    multi_result = multi_builder.build(item, config, chat_tokenizer)
    assert multi_result == facade_result


def test_multi_builder_rejects_single_config(chat_tokenizer, multi_builder):
    config = make_chat_config()
    item = {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    assert multi_builder.build(item, config, chat_tokenizer) is None
