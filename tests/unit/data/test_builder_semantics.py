import pytest

from astrai.config.preprocess_config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing.builder import (
    BaseMaskBuilder,
    MultiOutputMaskBuilder,
    SectionRenderer,
    SingleOutputMaskBuilder,
)


class RecordingTokenizer:
    bos_token_id = 1000

    def __init__(self):
        self.calls = []

    def encode(self, texts, *, add_special_tokens):
        self.calls.append((texts, add_special_tokens))

        def tokens(text):
            return ([self.bos_token_id] if add_special_tokens else []) + [
                ord(char) for char in text
            ]

        return tokens(texts) if isinstance(texts, str) else [tokens(t) for t in texts]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is False
        return "".join(f"<{msg['role']}:{msg['content']}>" for msg in messages)


def config_for(*, sections=None, sources=None, **processing):
    return PipelineConfig(
        input=InputConfig(sections=sections, sources=sources),
        preprocessing=ProcessingConfig(**{"min_chars": 0, **processing}),
        output=OutputConfig(domain_key="source"),
    )


def response_source(**section):
    return {
        "list_field": True,
        "sections": [{"field": "responses", "action": "train", **section}],
    }


def assert_scalar_and_batch(builder, items, config, expected):
    scalar, batch = RecordingTokenizer(), RecordingTokenizer()
    assert [builder.build(item, config, scalar) for item in items] == expected
    assert builder.build_batch(items, config, batch) == expected
    return scalar, batch


@pytest.mark.parametrize("special", [None, False, True])
def test_list_special_tokens_match_for_scalar_and_batch(special):
    """Regression: batch used to add BOS by default; scalar ignored explicit True."""
    section = {"field": "responses", "action": "train"}
    if special is not None:
        section["add_special_tokens"] = special
    config = config_for(sources={"responses": response_source(**section)})
    builder = MultiOutputMaskBuilder()
    items = [{"responses": ["a", "bc"], "source": "test"}, {"responses": ["d"]}]
    prefix = [1000] if special else []
    expected = [
        {
            "responses": [prefix + [97], prefix + [98, 99]],
            "responses_mask": [[1] * (len(prefix) + 1), [1] * (len(prefix) + 2)],
            "domain": "test",
        },
        {
            "responses": [prefix + [100]],
            "responses_mask": [[1] * (len(prefix) + 1)],
            "domain": "__default__",
        },
    ]
    scalar_tokenizer, batch_tokenizer = assert_scalar_and_batch(
        builder, items, config, expected
    )
    assert all(isinstance(text, str) for text, _ in scalar_tokenizer.calls)
    assert batch_tokenizer.calls == [(["a", "bc", "d"], bool(special))]


@pytest.mark.parametrize("special", [False, True])
def test_list_templates_do_not_add_encoder_special_tokens(special):
    config = config_for(
        sources={
            "responses": response_source(
                action="$role", template=True, add_special_tokens=special
            )
        },
        min_chars=100,
    )
    config.mask = {"assistant": "train"}
    items = [{"responses": [[{"role": "assistant", "content": "a"}], [], "invalid"]}]
    expected_ids = [ord(char) for char in "<assistant:a>"]
    expected = [
        {
            "responses": [expected_ids],
            "responses_mask": [[1] * len(expected_ids)],
            "domain": "__default__",
        }
    ]
    builder = MultiOutputMaskBuilder()
    tokenizers = assert_scalar_and_batch(builder, items, config, expected)
    assert all(
        not special for tokenizer in tokenizers for _, special in tokenizer.calls
    )


def test_list_elements_skip_empty_values_and_truncate_without_text_filtering():
    config = config_for(
        sources={"responses": response_source()},
        min_chars=10,
        max_chars=2,
        max_seq_len=2,
    )
    items = [{"responses": ["", " ", "a", "long"]}, {"responses": []}]
    expected = [
        {
            "responses": [[97], [108, 111]],
            "responses_mask": [[1], [1, 1]],
            "domain": "__default__",
        },
        None,
    ]
    builder = MultiOutputMaskBuilder()
    assert_scalar_and_batch(builder, items, config, expected)


@pytest.mark.parametrize("first", ["", " ", "a", "too-long-text"])
def test_first_retained_section_gets_special_tokens_after_filtering(first):
    config = config_for(
        sections=[
            {"field": "first", "action": "train"},
            {"field": "second", "action": "train"},
        ],
        min_chars=2,
        max_chars=4,
        max_seq_len=3,
    )
    item = {"first": first, "second": "abcd", "source": 123}
    expected = {"sequence": [1000, 97, 98, 99, 100], "domain": "__default__"}
    builder = SingleOutputMaskBuilder()
    assert_scalar_and_batch(builder, [item], config, [expected])


def test_batch_groups_special_policy_and_preserves_full_mask():
    config = config_for(
        sections=[
            {"field": "prompt", "action": "mask"},
            {"field": "answer", "action": "train"},
        ],
        max_seq_len=4,
    )
    items = [{"prompt": "ab", "answer": "cd"}, {"prompt": "", "answer": "ef"}]
    expected = [
        {
            "sequence": [1000, 97, 98, 99, 100],
            "loss_mask": [0, 0, 0, 1, 1],
            "domain": "__default__",
        },
        {"sequence": [1000, 101, 102], "domain": "__default__"},
    ]
    _, tokenizer = assert_scalar_and_batch(
        SingleOutputMaskBuilder(), items, config, expected
    )
    # Group policies must stay distinct, but group execution order is immaterial.
    assert len(tokenizer.calls) == 2
    assert {special: texts for texts, special in tokenizer.calls} == {
        False: ["cd"],
        True: ["ab", "ef"],
    }


def test_top_level_template_bos_and_empty_records_before_overflow():
    config = config_for(
        sections=[{"field": "messages", "action": "$role", "template": True}],
        max_seq_len=3,
    )
    config.mask = {"assistant": "train"}
    items = [
        {"messages": [{"role": "assistant", "content": "a"}]},
        {"messages": []},
        {},
    ]
    expected = [
        {
            "sequence": [1000] + [ord(c) for c in "<assistant:a>"],
            "loss_mask": [0] + [1] * len("<assistant:a>"),
            "domain": "__default__",
        },
        None,
        None,
    ]
    builder = SingleOutputMaskBuilder()
    assert_scalar_and_batch(builder, items, config, expected)
    config.preprocessing.max_seq_len = 1
    assert builder.build_batch(items, config, RecordingTokenizer()) == expected


def test_multi_requires_all_outputs_and_preserves_explicit_masks_and_values():
    config = config_for(
        sources={
            "chosen": {
                "mask_key": "train_mask",
                "sections": [{"field": "chosen", "action": "train"}],
            },
            "rejected": {"sections": [{"field": "rejected", "action": "mask"}]},
            "scores": {"sections": [{"field": "score", "action": "value"}]},
            "unused": {"sections": []},
        }
    )
    item = {"chosen": "a", "rejected": "b", "score": 0.5, "source": "ranked"}
    items = [item, {**item, "score": None}, {**item, "rejected": ""}]
    expected = [
        {
            "chosen": [1000, 97],
            "train_mask": [1, 1],
            "rejected": [1000, 98],
            "rejected_mask": [0, 0],
            "scores": [0.5],
            "domain": "ranked",
        },
        None,
        None,
    ]
    builder = MultiOutputMaskBuilder()
    assert_scalar_and_batch(builder, items, config, expected)


@pytest.mark.parametrize(
    "invalid", [None, [[1]], [[1], [2], [3]], [1, 2], [[[1]], [[2]]]]
)
def test_batch_tokenizer_shape_and_count_are_checked(invalid):
    class InvalidTokenizer(RecordingTokenizer):
        def encode(self, texts, *, add_special_tokens):
            return invalid

    config = config_for(sections=[{"field": "text", "action": "train"}])
    with pytest.raises(ValueError, match="tokenizer.encode"):
        SingleOutputMaskBuilder().build_batch(
            [{"text": "a"}, {"text": "b"}], config, InvalidTokenizer()
        )


def test_scalar_build_supports_scalar_only_tokenizers():
    class ScalarOnlyTokenizer(RecordingTokenizer):
        def encode(self, text, *, add_special_tokens):
            assert isinstance(text, str)
            return super().encode(text, add_special_tokens=add_special_tokens)

    tokenizer = ScalarOnlyTokenizer()
    config = config_for(sources={"responses": response_source()})
    result = MultiOutputMaskBuilder().build(
        {"responses": ["a", "b"]}, config, tokenizer
    )
    assert result["responses"] == [[97], [98]]
    text_config = config_for(sections=[{"field": "text", "action": "train"}])
    assert SingleOutputMaskBuilder().build({"text": "a"}, text_config, tokenizer)[
        "sequence"
    ] == [1000, 97]


def test_builders_dispatch_to_custom_scalar_and_batch_renderer_methods():
    class CustomRenderer(SectionRenderer):
        def process_sections(
            self, item, sections, config, tokenizer, *, is_top_level=False
        ):
            return [11], [0]

        def process_sections_batch(self, items, sections, config, tokenizer, **kwargs):
            return [([12], [0]) for _ in items]

        def process_list_field(self, item, sections, config, tokenizer):
            return [[21]], [[1]]

        def process_list_field_batch(self, items, sections, config, tokenizer):
            return [([[22]], [[1]]) for _ in items]

    renderer = CustomRenderer()
    section = {"field": "text", "action": "train"}
    single = SingleOutputMaskBuilder(renderer)
    config = config_for(sections=[section])
    assert single.build({}, config, None)["sequence"] == [11]
    assert single.build_batch([{}], config, None)[0]["sequence"] == [12]
    multi = MultiOutputMaskBuilder(renderer)
    config = config_for(
        sources={
            "text": {"sections": [section]},
            "responses": {"sections": [section], "list_field": True},
        }
    )
    assert multi.build({}, config, None) == {
        "text": [11],
        "text_mask": [0],
        "responses": [[21]],
        "responses_mask": [[1]],
        "domain": "__default__",
    }
    assert multi.build_batch([{}], config, None) == [
        {
            "text": [12],
            "text_mask": [0],
            "responses": [[22]],
            "responses_mask": [[1]],
            "domain": "__default__",
        }
    ]


def test_list_batch_keeps_custom_section_renderer_dispatch():
    class CustomRenderer(SectionRenderer):
        def process_sections_batch(self, items, sections, config, tokenizer, **kwargs):
            assert kwargs == {"is_top_level": False, "filter_text": False}
            assert sections[0]["add_special_tokens"] is False
            return [([77], [1]) for _ in items]

    renderer = CustomRenderer()
    result = renderer.process_list_field_batch(
        [{"responses": ["a", "b"]}],
        [{"field": "responses", "action": "train"}],
        None,
        None,
    )
    assert result == [([[77], [77]], [[1], [1]])]


def test_empty_batch_and_base_builder_fallback_do_not_recurse():
    class ScalarBuilder(BaseMaskBuilder):
        def build(self, item, config, tokenizer):
            return {"value": item["value"] + 1}

    assert ScalarBuilder().build_batch([{"value": 1}, {"value": 2}], None, None) == [
        {"value": 2},
        {"value": 3},
    ]
    tokenizer = RecordingTokenizer()
    config = config_for(sections=[{"field": "text", "action": "train"}])
    assert SingleOutputMaskBuilder().build_batch([], config, tokenizer) == []
    assert MultiOutputMaskBuilder().build_batch([], config, tokenizer) == []
    assert tokenizer.calls == []
