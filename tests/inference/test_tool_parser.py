import json

"""Public tool parsing contracts, including incremental byte boundaries."""

import pytest

from astrai.inference.network.tool_parser import SimpleJsonToolParser


@pytest.mark.parametrize(
    "source,expected",
    [
        ('{"city": "Beijing"}', '{"city": "Beijing"}'),
        ('{"a": {"b": {"c": {"d": 4}}}}', '{"a": {"b": {"c": {"d": 4}}}}'),
        (
            '{"active": true, "count": 0, "nick": null}',
            '{"active": true, "count": 0, "nick": null}',
        ),
        ('{"items": [1, 2, 3]}', '{"items": [1, 2, 3]}'),
        ('{"rows": [{"id": 1}, {"id": 2}]}', '{"rows": [{"id": 1}, {"id": 2}]}'),
        ('{"text": "你好，世界"}', '{"text": "你好，世界"}'),
        (r'{"text": "he said \"hello\""}', '{"text": "he said \\"hello\\""}'),
        (
            '{"code": "function(x) { return x + 1; }"}',
            '{"code": "function(x) { return x + 1; }"}',
        ),
        ('"just a string"', "just a string"),
        ("{}", "{}"),
    ],
    ids=[
        "object",
        "nested",
        "literals",
        "array",
        "object-array",
        "unicode",
        "escaped",
        "braces",
        "string",
        "empty",
    ],
)
def test_complete_arguments_preserve_source(source, expected):
    result = SimpleJsonToolParser().parse_complete(
        '{"name": "f", "arguments": ' + source + "}"
    )
    assert result["content"] is None
    assert result["tool_calls"][0]["function"] == {"name": "f", "arguments": expected}


@pytest.mark.parametrize("prefix", ["", "Some text. "])
def test_complete_multiple_calls_preserve_content_order_and_ids(prefix):
    body = prefix + '{"name": "f1", "arguments": {"a": 1}}{"name": "f2"}'
    result = SimpleJsonToolParser().parse_complete(body)
    assert result["content"] == (prefix.strip() or None)
    assert [call["function"] for call in result["tool_calls"]] == [
        {"name": "f1", "arguments": '{"a": 1}'},
        {"name": "f2", "arguments": ""},
    ]
    identifiers = [call["id"] for call in result["tool_calls"]]
    assert len(set(identifiers)) == 2
    assert all(identifier.startswith("call_") for identifier in identifiers)


@pytest.mark.parametrize(
    "body",
    ["Hello world", '{"not_a_tool": true}', '{"nam', '{"name": "f", "arguments": {"x"'],
)
def test_complete_ignores_plain_or_unfinished_calls(body):
    assert SimpleJsonToolParser().parse_complete(body) is None


@pytest.mark.parametrize(
    "body,arguments",
    [
        ('{"name": "f", "arguments": {"city"', '{"city"'),
        ('{"name": "f", "arguments": {"a": {"b": ', '{"a": {"b":'),
        ('{"name": "f", "arguments": {"items": [1, 2, ', '{"items": [1, 2,'),
    ],
)
def test_partial_calls_emit_available_name_and_arguments(body, arguments):
    parser = SimpleJsonToolParser()
    deltas = parser.feed(body)
    functions = [call["function"] for delta in deltas for call in delta["tool_calls"]]
    assert [function["name"] for function in functions if "name" in function] == ["f"]
    assert "".join(function.get("arguments", "") for function in functions) == arguments


@pytest.mark.parametrize(
    "body,arguments",
    [
        (
            '{"name": "deep", "arguments": {"a": {"b": {"c": 42}}}}',
            '{"a": {"b": {"c": 42}}}',
        ),
        (
            '{"name": "translate", "arguments": {"text": "你好，世界"}}',
            '{"text": "你好，世界"}',
        ),
        ('{"name": "add", "arguments": {"items": [1, 2, 3]}}', '{"items": [1, 2, 3]}'),
        (
            r'{"name":"echo","arguments":{"text":"a{b}c \"quoted\""}}',
            '{"text":"a{b}c \\"quoted\\""}',
        ),
        ('{"name":"ping","arguments":{}}', "{}"),
        (
            '{"name": "get_weather","arguments": {"city":"Beijing","unit":"c"}}',
            '{"city":"Beijing","unit":"c"}',
        ),
    ],
)
def test_streamed_arguments_match_complete_source(body, arguments):
    parser = SimpleJsonToolParser()
    names, chunks = _simulate_streaming(parser, body)
    expected_name = json.loads(body)["name"]
    completed = parser.parse_complete(body)["tool_calls"][0]["function"]
    assert names == [expected_name]
    assert "".join(chunks) == arguments
    assert completed == {"name": expected_name, "arguments": arguments}
    assert json.loads("".join(chunks)) == json.loads(body)["arguments"]


def test_feed_incremental_text():
    parser = SimpleJsonToolParser()
    assert parser.feed("He") == [{"content": "He"}]
    assert parser.feed("Hello") == [{"content": "llo"}]


def test_feed_no_content_when_no_new_text():
    parser = SimpleJsonToolParser()
    parser.feed("Hello")
    assert parser.feed("Hello") == []


def test_feed_content_after_tool_call_is_not_emitted():
    parser = SimpleJsonToolParser()
    deltas = parser.feed('{"name": "f", "arguments": {}} trailing text')
    assert parser.has_tool_calls
    assert not any("trailing" in d.get("content", "") for d in deltas)


def _simulate_streaming(parser, text):
    all_delta_names = []
    all_args_chunks = []
    for i in range(1, len(text) + 1):
        deltas = parser.feed(text[:i])
        for d in deltas:
            if "tool_calls" in d:
                for tc in d["tool_calls"]:
                    fn = tc.get("function", {})
                    if "name" in fn:
                        all_delta_names.append(fn["name"])
                    if "arguments" in fn and fn["arguments"]:
                        all_args_chunks.append(fn["arguments"])
    return all_delta_names, all_args_chunks


def test_streaming_token_by_token_text_then_tool():
    parser = SimpleJsonToolParser()
    parts = [
        "I'll ",
        "check ",
        "that. ",
        '{"',
        'name": "search", ',
        '"arguments": {"q": "hello"}}',
    ]
    body = ""
    content_chunks = []
    tool_names = []
    for part in parts:
        body += part
        deltas = parser.feed(body)
        for d in deltas:
            if "content" in d:
                content_chunks.append(d["content"])
            if "tool_calls" in d:
                for tc in d["tool_calls"]:
                    fn = tc.get("function", {})
                    if "name" in fn:
                        tool_names.append(fn["name"])
    full_content = "".join(content_chunks)
    assert "I'll check that." in full_content
    assert "search" in tool_names


def test_streaming_args_diff_only_emits_new_bytes():
    parser = SimpleJsonToolParser()
    step1 = parser.feed('{"name": "f", "arguments": {"city": "Bei')
    step2 = parser.feed('{"name": "f", "arguments": {"city": "Beijing"}}')
    all_args = []
    for step in (step1, step2):
        for d in step:
            if "tool_calls" in d:
                for tc in d["tool_calls"]:
                    fn = tc.get("function", {})
                    if "arguments" in fn and fn["arguments"]:
                        all_args.append(fn["arguments"])
    joined = "".join(all_args)
    assert "city" in joined
    assert "Beijing" in joined
    assert joined.startswith('{"city":')
    assert all_args[0] != all_args[1]


def test_streaming_distinct_tool_call_ids():
    parser = SimpleJsonToolParser()
    text = '{"name": "f1", "arguments": {"a": 1}}{"name": "f2", "arguments": {"b": 2}}'
    calls = {}
    for i in range(1, len(text) + 1):
        for delta in parser.feed(text[:i]):
            for call in delta.get("tool_calls", []):
                actual = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                if "id" in call:
                    actual["id"] = call["id"]
                for field, value in call.get("function", {}).items():
                    actual[field] += value
    assert list(calls) == [0, 1]
    assert [call["name"] for call in calls.values()] == ["f1", "f2"]
    assert [call["arguments"] for call in calls.values()] == ['{"a": 1}', '{"b": 2}']
    identifiers = [call["id"] for call in calls.values()]
    assert len(set(identifiers)) == 2
    assert all(identifier.startswith("call_") for identifier in identifiers)


def test_feed_then_parse_complete_same_instance():
    parser = SimpleJsonToolParser()
    parser.feed('{"name": "get_weather", "arguments": {"city": "Beijing"}}')
    result = parser.parse_complete(
        '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    )
    assert result is not None
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert parser.has_tool_calls


def test_streaming_partial_name_prefix_never_leaks_into_content():
    parser = SimpleJsonToolParser()
    parts = ["Hello ", '{"', '{"n', '{"na', '{"name"']
    emitted = []
    body = ""
    for part in parts:
        body += part
        for d in parser.feed(body):
            if "content" in d:
                emitted.append(d["content"])
    assert "".join(emitted) == "Hello "


def test_finalize_flushes_withheld_plain_json_content():
    parser = SimpleJsonToolParser()
    text = 'Answer: {"price": 1}'
    deltas = parser.feed(text)
    streamed = "".join(d["content"] for d in deltas if "content" in d)
    flushed = parser.finalize(text)
    joined = streamed + "".join(d["content"] for d in flushed if "content" in d)
    assert joined == text
    assert not parser.has_tool_calls
    assert parser.finalize(text) == []
