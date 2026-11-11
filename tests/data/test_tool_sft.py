"""Tool SFT keeps full-conversation tokens and assistant-only supervision."""

import json
import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import decoders

from astrai.config.preprocess_config import InputConfig, PipelineConfig
from astrai.dataset import DatasetFactory, sft_collate_fn
from astrai.inference.network.app import ChatCompletionRequest
from astrai.inference.network.openai import OpenAIResponseBuilder
from astrai.inference.network.tool_parser import SimpleJsonToolParser
from astrai.preprocessing.builder import SingleOutputMaskBuilder
from astrai.preprocessing.pipeline import Pipeline
from astrai.tokenize import AutoTokenizer, ChatTemplate
from tests.support.tokenizers import build_test_tokenizer

EXAMPLE = Path(__file__).resolve().parents[2] / "docs" / "examples" / "tool_sft"
START, END = "<｜im▁start｜>", "<｜im▁end｜>"


@pytest.fixture
def record():
    return json.loads(EXAMPLE.joinpath("data.jsonl").read_text().splitlines()[0])


@pytest.fixture
def tokenizer():
    tokenizer = build_test_tokenizer(
        special_tokens=("<unk>", "<pad>", START, END),
        add_prefix_space=False,
        train_data=[chr(i) for i in range(256)]
        + ["北京现在天气怎么样？晴，气温 18°C。上海下雨谢谢不客气你好"],
        chat_template=EXAMPLE.joinpath("chat_template.jinja").read_text(),
    )
    tokenizer._tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def encode(tokenizer, record):
    return tokenizer.apply_chat_template(
        record["messages"],
        tools=record.get("tools", []),
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
    )


def test_full_conversation_mask_and_parser(tokenizer, record):
    original = deepcopy(record)
    encoded = encode(tokenizer, record)
    prompt = tokenizer.apply_chat_template(
        record["messages"],
        tools=record["tools"],
        add_generation_prompt=False,
        tokenize=False,
    )
    assert record == original
    assert prompt.count("Available tools:") == 1
    assert encoded["input_ids"] == tokenizer.encode(prompt, add_special_tokens=False)
    supervised = tokenizer.decode(
        [i for i, m in zip(encoded["input_ids"], encoded["assistant_masks"]) if m],
        skip_special_tokens=False,
    )
    call = json.dumps(
        {"name": "get_weather", "arguments": {"city": "北京"}}, ensure_ascii=False
    )
    assert supervised == call + END + "北京现在晴，气温 18°C。" + END
    parsed = SimpleJsonToolParser().parse_complete(call)
    assert parsed["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "city": "北京"
    }
    # A returned call can be sent back unchanged in the next API request.
    replay = deepcopy(record)
    replay["messages"][1] = {"role": "assistant", **parsed}
    replay["messages"][2]["tool_call_id"] = parsed["tool_calls"][0]["id"]
    assert encode(tokenizer, replay) == encoded
    # Object and API JSON-string arguments have identical training tokens/masks.
    record["messages"][1]["tool_calls"][0]["function"]["arguments"] = {"city": "北京"}
    assert encode(tokenizer, record) == encoded


def test_parallel_calls_and_multiple_rounds(tokenizer, record):
    second = deepcopy(record["messages"][1]["tool_calls"][0])
    second["id"] = "call_2"
    second["function"]["arguments"] = {"city": "上海"}
    record["messages"][1]["tool_calls"].append(second)
    record["messages"].insert(
        3, {"role": "tool", "tool_call_id": "call_2", "content": "上海下雨"}
    )
    record["messages"].extend(
        [
            {"role": "user", "content": "谢谢"},
            {"role": "assistant", "content": "不客气"},
        ]
    )
    output = encode(tokenizer, record)
    text = tokenizer.decode(
        [i for i, m in zip(output["input_ids"], output["assistant_masks"]) if m],
        skip_special_tokens=False,
    )
    assert text.count(END) == 3
    assert '"city": "上海"' in text
    assert "上海下雨" not in text
    assert text.endswith("不客气" + END)
    rendered = tokenizer.apply_chat_template(record["messages"], tokenize=False)
    assert (
        '"name": "get_weather", "arguments": {"city": "上海"}, "result": "上海下雨"'
        in rendered
    )


def test_batch_and_template_save_reload(tokenizer, record, tmp_path):
    plain = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "Hello!"},
    ]
    conversations = [record["messages"], plain]
    kwargs = dict(add_generation_prompt=False, return_assistant_tokens_mask=True)
    batch = tokenizer.apply_chat_template(conversations, **kwargs)
    singles = [
        tokenizer.apply_chat_template(messages, **kwargs) for messages in conversations
    ]
    assert batch == {key: [s[key] for s in singles] for key in batch}
    tokenizer.save_pretrained(tmp_path)
    loaded = AutoTokenizer(tmp_path)
    assert loaded.apply_chat_template(conversations, **kwargs) == batch
    assert (
        pickle.loads(pickle.dumps(tokenizer)).apply_chat_template(
            conversations, **kwargs
        )
        == batch
    )


def test_pipeline_packing_preserves_supervision(tokenizer, record, tmp_path):
    config = PipelineConfig.from_file(str(EXAMPLE / "pipeline.json"))
    records = [
        record,
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        },
    ]
    builder = SingleOutputMaskBuilder()
    expected = [builder.build(item, config, tokenizer) for item in records]
    assert builder.build_batch(records, config, tokenizer) == expected
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    source = tmp_path / "data.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in records))
    output = tmp_path / "out"
    Pipeline(config, [str(source)], str(output), str(tmp_path / "tokenizer")).run()
    dataset = DatasetFactory.load("sft", str(output), window_size=8192)
    assert len(dataset) == 1
    sample = dataset[0]
    ids = expected[0]["sequence"] + expected[1]["sequence"]
    mask = expected[0]["loss_mask"] + expected[1]["loss_mask"]
    assert sample["input_ids"].tolist() == ids
    assert sample["loss_mask"].tolist() == mask
    assert sample["seq_lengths"].tolist() == [len(e["sequence"]) for e in expected]
    batch = sft_collate_fn([sample])
    labels = batch["target_ids"][0][batch["loss_mask"][0].bool()].tolist()
    assert labels == [i for i, m in zip(ids, mask) if m]


def test_serving_keeps_tool_history(tokenizer, record):
    request = ChatCompletionRequest(**record)
    prompt, _, _ = OpenAIResponseBuilder().prepare(
        request, SimpleNamespace(tokenizer=tokenizer)
    )
    expected = tokenizer.apply_chat_template(
        record["messages"], tools=record["tools"], tokenize=False
    )
    assert prompt == expected
    assert prompt.endswith(START + "assistant\n")


@pytest.mark.parametrize("arguments", ["{bad", "[]", [], None, "null"])
def test_reject_invalid_tool_arguments(tokenizer, record, arguments):
    record["messages"][1]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(ValueError, match="Tool arguments"):
        encode(tokenizer, record)


@pytest.mark.parametrize(
    "template",
    [
        "{% set x %}{% generation %}x{% endgeneration %}{% endset %}{{ x }}",
        "{% filter upper %}{% generation %}x{% endgeneration %}{% endfilter %}",
        "{% macro x() %}{% generation %}x{% endgeneration %}{% endmacro %}{{ x() }}",
        "{% generation %}{% generation %}x{% endgeneration %}{% endgeneration %}",
        "{% for x in messages recursive %}{% generation %}x{% endgeneration %}{% endfor %}",
    ],
)
def test_reject_buffered_generation_blocks(template):
    with pytest.raises(ValueError, match="buffered constructs"):
        ChatTemplate.from_string(template).render_with_spans([])


def test_spans_at_start_and_tracker_reset():
    template = ChatTemplate.from_string(
        "{% generation %}{{ messages[0]['content'] }}{% endgeneration %}tail"
    )
    with pytest.raises(Exception):
        template.render_with_spans([])
    assert template.render_with_spans([{"content": "你好"}]) == ("你好tail", [(0, 2)])
    assert template.render_with_spans([{"content": "ok"}]) == ("oktail", [(0, 2)])
    assert template.render([{"content": "ok"}]) == "oktail"


def test_reject_token_crossing_mask_boundary():
    tokenizer = build_test_tokenizer(
        add_prefix_space=False,
        train_data=["foobar"] * 20,
        chat_template="foo{% generation %}bar{% endgeneration %}",
    )
    assert len(tokenizer.encode("foobar", add_special_tokens=False)) == 1
    with pytest.raises(ValueError, match="crosses a generation boundary"):
        tokenizer.apply_chat_template([], return_assistant_tokens_mask=True)
    tokenizer.set_chat_template(
        "{% generation %}foo{% endgeneration %}{% generation %}bar{% endgeneration %}"
    )
    assert tokenizer.apply_chat_template([], return_assistant_tokens_mask=True)[
        "assistant_masks"
    ] == [1]


def test_no_generation_supervision_fails(tokenizer, record):
    tokenizer.set_chat_template("{{ messages | tojson }}")
    config = PipelineConfig.from_file(str(EXAMPLE / "pipeline.json"))
    with pytest.raises(ValueError, match="no supervised tokens"):
        SingleOutputMaskBuilder().build(record, config, tokenizer)
    with pytest.raises(ValueError, match="tokenize=True"):
        tokenizer.apply_chat_template(
            record["messages"], tokenize=False, return_assistant_tokens_mask=True
        )


@pytest.mark.parametrize(
    "sections,sources",
    [
        (None, None),
        ([{"field": "messages", "action": "train", "template": True}], None),
        ([{"field": "messages", "action": "$role", "template": True}], {"chosen": {}}),
    ],
)
def test_invalid_mask_configuration(sections, sources):
    with pytest.raises(ValueError, match="chat_template_mask requires"):
        InputConfig(chat_template_mask=True, sections=sections, sources=sources)
