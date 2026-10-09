from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from astrai.tokenize import AutoTokenizer, ChatTemplate

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}SYSTEM: {{ message['content'] }}\n{% endif %}"
    "{% if message['role'] == 'user' %}USER: {{ message['content'] }}\n{% endif %}"
    "{% if message['role'] == 'assistant' %}ASSISTANT: {{ message['content'] }}\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT: {% endif %}"
)


def build_test_tokenizer(
    vocab_size: int = 1000,
    *,
    special_tokens=("<unk>", "<pad>"),
    special_token_map=None,
    add_prefix_space: bool = True,
    train_data=None,
    chat_template: str | None = None,
) -> AutoTokenizer:
    """Build a lightweight BPE ``AutoTokenizer`` for tests.

    ``special_token_map`` defaults to ``{"unk_token", "pad_token"}``
    pointing at the first two special tokens.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=add_prefix_space
    )
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=list(special_tokens),
    )
    tokenizer.train_from_iterator(
        train_data if train_data is not None else [chr(i) for i in range(256)],
        trainer,
    )
    auto_tokenizer = AutoTokenizer()
    auto_tokenizer._tokenizer = tokenizer
    auto_tokenizer._special_token_map = special_token_map or {
        "unk_token": special_tokens[0],
        "pad_token": special_tokens[1],
    }
    if chat_template is not None:
        auto_tokenizer.set_chat_template(chat_template)
    return auto_tokenizer


class FakeTokenizer:
    """Minimal stub tokenizer with optional chat-template support."""

    stop_ids = [2]

    def __init__(self, *, with_chat_template=False):
        if with_chat_template:
            self._chat_template = ChatTemplate.from_string(CHAT_TEMPLATE)
        else:
            self._chat_template = None

    def encode(self, texts, **_):
        if isinstance(texts, str):
            texts = [texts]
        return [[b for b in t.encode("utf-8")] for t in texts]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, list):
            return bytes(b for b in ids if b > 2 or not skip_special_tokens).decode(
                "utf-8", errors="ignore"
            )
        return str(ids)

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, **_
    ):
        if self._chat_template is None:
            raise RuntimeError("Chat template not configured")
        rendered = self._chat_template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
        if tokenize:
            return (
                self.encode(rendered)[0]
                if isinstance(rendered, str)
                else [self.encode(t)[0] for t in rendered]
            )
        return rendered
