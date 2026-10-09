import json
import os

import pytest

from astrai.preprocessing.builder import (
    MultiOutputMaskBuilder,
    SectionedMaskBuilder,
    SingleOutputMaskBuilder,
)
from tests.support.tokenizers import build_test_tokenizer

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


_CHAT_TOKENIZER_DATA = [
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

_CHAT_TOKENIZER_MAP = {
    "bos_token": "<|begin_of_sentence|>",
    "eos_token": "<|end_of_sentence|>",
    "pad_token": "<|_pad_|>",
    "unk_token": "<|_unk_|>",
}


def _build_chat_tokenizer():
    return build_test_tokenizer(
        vocab_size=512,
        special_tokens=_SPECIAL_TOKENS,
        special_token_map=_CHAT_TOKENIZER_MAP,
        add_prefix_space=False,
        train_data=_CHAT_TOKENIZER_DATA,
        chat_template=_CHAT_TEMPLATE,
    )


@pytest.fixture(scope="session")
def chat_tokenizer():
    return _build_chat_tokenizer()


def _write_tokenizer_dir(dir_path, tokenizer, tokenizer_config):
    """Persist a tokenizer plus ``tokenizer_config.json`` into *dir_path*."""
    tokenizer._tokenizer.save(os.path.join(dir_path, "tokenizer.json"))
    with open(os.path.join(dir_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)


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
    _write_tokenizer_dir(
        d,
        test_tokenizer,
        {"special_tokens": {"pad_token": "<|_pad_|>", "unk_token": "<|_unk_|>"}},
    )
    return d


@pytest.fixture
def chat_tokenizer_dir(temp_dir, chat_tokenizer):
    d = os.path.join(temp_dir, "tok")
    os.makedirs(d, exist_ok=True)
    _write_tokenizer_dir(
        d,
        chat_tokenizer,
        {"special_tokens": _SPECIAL_TOKENS_CONFIG, "chat_template": _CHAT_TEMPLATE},
    )
    return d
