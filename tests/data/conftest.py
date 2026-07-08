import json
import os
import tempfile

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from astrai.config.preprocess_config import (
    InputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing.builder import (
    MultiOutputMaskBuilder,
    SectionedMaskBuilder,
    SingleOutputMaskBuilder,
)
from astrai.tokenize import AutoTokenizer

_SPECIAL_TOKENS_CONFIG = {
    "bos_token": "<|begin_of_sentence|>",
    "eos_token": "<|end_of_sentence|>",
    "pad_token": "<|_pad_|>",
    "unk_token": "<|_unk_|>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}

_SPECIAL_TOKENS = list(_SPECIAL_TOKENS_CONFIG.values())

_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_CHAT_SECTIONS = [{"field": "messages", "action": "$role", "template": True}]

_INSTRUCTION_SECTIONS = [
    {"field": "prompt", "action": "mask", "add_special_tokens": True},
    {"field": "response", "action": "train"},
]

_TEXT_SECTIONS = [{"field": "text", "action": "train"}]

_GRPO_RESPONSE_SECTIONS = [{"field": "responses", "action": "train"}]


def _build_chat_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        special_tokens=_SPECIAL_TOKENS,
    )
    train_data = [
        "hello world",
        "Hi there!",
        "You are helpful.",
        "What is 2+2?",
        "Tell me a story about dragons and knights.",
        "Sure, here is a tale.",
        "Translate to French: Hello",
        "Bonjour",
        "Artificial Intelligence is a field of computer science.",
        "system",
        "user",
        "assistant",
        "<|im_start|>",
        "<|im_end|>",
        *[chr(i) for i in range(32, 127)],
    ]
    tok.train_from_iterator(train_data, tr)

    auto_tok = AutoTokenizer()
    auto_tok._tokenizer = tok
    auto_tok._special_token_map = {
        "bos_token": "<|begin_of_sentence|>",
        "eos_token": "<|end_of_sentence|>",
        "pad_token": "<|_pad_|>",
        "unk_token": "<|_unk_|>",
    }
    auto_tok.set_chat_template(_CHAT_TEMPLATE)
    return auto_tok


@pytest.fixture(scope="session")
def chat_tokenizer():
    return _build_chat_tokenizer()


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


def make_chat_config():
    return PipelineConfig(
        input=InputConfig(sections=_CHAT_SECTIONS),
        mask={"system": "mask", "user": "mask", "assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )


def make_instruction_config():
    return PipelineConfig(
        input=InputConfig(sections=_INSTRUCTION_SECTIONS),
        mask={"prompt": "mask", "response": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )


def make_text_config():
    return PipelineConfig(
        input=InputConfig(sections=_TEXT_SECTIONS),
        preprocessing=ProcessingConfig(
            max_seq_len=2048, min_chars=1, max_chars=2_000_000
        ),
    )


def make_dpo_chat_config():
    return PipelineConfig(
        input=InputConfig(
            sources={
                "chosen": {
                    "sections": [
                        {"field": "chosen", "action": "$role", "template": True}
                    ]
                },
                "rejected": {
                    "sections": [
                        {"field": "rejected", "action": "$role", "template": True}
                    ]
                },
            }
        ),
        mask={"user": "mask", "assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )


def make_grpo_config():
    return PipelineConfig(
        input=InputConfig(
            sources={
                "prompts": {
                    "sections": [
                        {"field": "prompt", "action": "mask", "template": True}
                    ]
                },
                "responses": {
                    "sections": _GRPO_RESPONSE_SECTIONS,
                    "list_field": True,
                    "mask_key": "masks",
                },
                "rewards": {
                    "sections": [{"field": "rewards", "action": "value"}],
                },
            }
        ),
        mask={"user": "mask", "assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )


def make_grpo_no_template_config():
    return PipelineConfig(
        input=InputConfig(
            sources={
                "prompts": {
                    "sections": [
                        {
                            "field": "prompt",
                            "action": "mask",
                            "add_special_tokens": True,
                        }
                    ]
                },
                "responses": {
                    "sections": _GRPO_RESPONSE_SECTIONS,
                    "list_field": True,
                    "mask_key": "masks",
                },
                "rewards": {
                    "sections": [{"field": "rewards", "action": "value"}],
                },
            }
        ),
        mask={"user": "mask", "assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )


@pytest.fixture
def builder():
    return SectionedMaskBuilder()


@pytest.fixture
def single_builder():
    return SingleOutputMaskBuilder()


@pytest.fixture
def multi_builder():
    return MultiOutputMaskBuilder()


@pytest.fixture
def tokenizer_dir(temp_dir, test_tokenizer):
    d = os.path.join(temp_dir, "tok")
    os.makedirs(d, exist_ok=True)
    test_tokenizer._tokenizer.save(os.path.join(d, "tokenizer.json"))
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump(
            {"special_tokens": {"pad_token": "<|_pad_|>", "unk_token": "<|_unk_|>"}}, f
        )
    return d


@pytest.fixture
def chat_tokenizer_dir(temp_dir, chat_tokenizer):
    d = os.path.join(temp_dir, "tok")
    os.makedirs(d, exist_ok=True)
    chat_tokenizer._tokenizer.save(os.path.join(d, "tokenizer.json"))
    with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
        json.dump(
            {"special_tokens": _SPECIAL_TOKENS_CONFIG, "chat_template": _CHAT_TEMPLATE},
            f,
        )
    return d
