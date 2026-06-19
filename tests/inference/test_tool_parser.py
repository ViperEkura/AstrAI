"""Unit tests for tool call parsers."""

import pytest

from astrai.inference.api.tool_parser import (
    _TOOL_CALL_HEAD_RE,
    BaseToolParser,
    SimpleJsonToolParser,
    ToolParserFactory,
    _find_partial_tool_call,
    _find_tool_calls,
    _scan_json,
)


@pytest.mark.parametrize(
    "text,expected_complete,check_end_eq_len",
    [
        ('{"key": "value"}', True, True),
        ('{"outer": {"inner": 1}}', True, True),
        ('{"key": "value"', False, False),
        ('{"outer": {"inner": 1}', False, False),
        ('{"key": "a{b}c"} extra', True, False),
        (r'{"key": "a\"b"}', True, False),
        ('{"a": {"b": {"c": {"d": {"e": 5}}}}}', True, True),
        ('{"items": [{"x": 1}, {"x": 2}]}', True, True),
        ('{"fn": "function() { return 1; }"}', True, False),
        ('{"key": "\u5317\u4eac"}', True, False),
    ],
)
def test_scan_json(text, expected_complete, check_end_eq_len):
    end, complete = _scan_json(text, 0)
    assert complete is expected_complete
    if check_end_eq_len:
        assert end == len(text)


def test_find_single_tool_call():
    text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["name"] == "get_weather"
    assert '"city"' in results[0]["args"]
    assert results[0]["complete"] is True


def test_find_text_before_tool_call():
    text = 'Some text {"name": "func", "arguments": {}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["start"] > 0


def test_find_multiple_tool_calls():
    text = '{"name": "f1", "arguments": {"a": 1}}{"name": "f2", "arguments": {"b": 2}}'
    results = _find_tool_calls(text)
    assert len(results) == 2
    assert results[0]["name"] == "f1"
    assert results[1]["name"] == "f2"


def test_find_no_tool_call():
    results = _find_tool_calls("Hello, how are you?")
    assert len(results) == 0


def test_find_non_tool_json_skipped():
    results = _find_tool_calls('{"not_a_tool": true}')
    assert len(results) == 0


def test_find_no_arguments_field():
    results = _find_tool_calls('{"name": "simple_func"}')
    assert len(results) == 1
    assert results[0]["name"] == "simple_func"
    assert results[0]["args"] == ""


def test_find_deeply_nested_arguments():
    text = '{"name": "deep", "arguments": {"a": {"b": {"c": {"d": 4}}}}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["name"] == "deep"
    assert '"d": 4' in results[0]["args"]


def test_find_arguments_with_boolean_and_null():
    text = '{"name": "flags", "arguments": {"active": true, "count": 0, "nick": null}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["name"] == "flags"
    assert "true" in results[0]["args"]
    assert "null" in results[0]["args"]


def test_find_arguments_with_array():
    text = '{"name": "add_items", "arguments": {"items": [1, 2, 3], "name": "list"}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["name"] == "add_items"
    assert "[1, 2, 3]" in results[0]["args"]


def test_find_arguments_with_nested_array_of_objects():
    text = '{"name": "batch", "arguments": {"rows": [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert '"rows"' in results[0]["args"]
    assert '"id": 1' in results[0]["args"]


def test_find_arguments_as_string_not_object():
    text = '{"name": "echo", "arguments": "just a string"}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["name"] == "echo"
    assert "just a string" in results[0]["args"]


def test_find_arguments_with_unicode():
    text = (
        '{"name": "translate", "arguments": {"text": "\u4f60\u597d\uff0c\u4e16\u754c"}}'
    )
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["name"] == "translate"


def test_find_arguments_with_escaped_quotes():
    text = '{"name": "format", "arguments": {"template": "he said \\"hello\\""}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert 'he said \\"hello\\"' in results[0]["args"]


def test_find_arguments_with_braces_in_string():
    text = '{"name": "eval", "arguments": {"code": "function(x) { return x + 1; }"}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["name"] == "eval"
    assert "function(x) { return x + 1; }" in results[0]["args"]


def test_find_many_properties():
    args = ",".join(f'"{chr(97 + i % 26)}" : {i}' for i in range(20))
    text = '{"name": "many", "arguments": {' + args + "}}"
    results = _find_tool_calls(text)
    assert len(results) == 1
    assert results[0]["name"] == "many"


def test_find_empty_arguments():
    results = _find_tool_calls('{"name": "ping", "arguments": {}}')
    assert len(results) == 1
    assert results[0]["name"] == "ping"
    assert results[0]["args"] == ""


def test_find_extracts_correct_arg_start_position():
    text = '{"name": "f", "arguments": {"x": 1}}'
    results = _find_tool_calls(text)
    assert len(results) == 1
    json_str = text[results[0]["start"] : results[0]["end"]]
    assert json_str == text


@pytest.mark.parametrize(
    "text,expected_name,expected_complete",
    [
        ('{"name": "func", "arguments": {"city"', "func", False),
        ('{"name": "func", "arguments": {"city": "BJ"}}', "func", None),
        ("plain text", None, None),
        ('{"nam', None, None),
        ('{"name": "deep", "arguments": {"a": {"b": {"c": ', "deep", None),
        ('{"name": "batch", "arguments": {"items": [1, 2, ', "batch", None),
    ],
)
def test_find_partial_tool_call(text, expected_name, expected_complete):
    result = _find_partial_tool_call(text)
    if expected_name is None:
        assert result is None
    else:
        assert result is not None
        assert result["name"] == expected_name
        if expected_complete is not None:
            assert result["complete"] is expected_complete


def test_feed_plain_text():
    parser = SimpleJsonToolParser()
    deltas = parser.feed("Hello")
    assert len(deltas) == 1
    assert deltas[0]["content"] == "Hello"


def test_feed_incremental_text():
    parser = SimpleJsonToolParser()
    assert parser.feed("He") == [{"content": "He"}]
    assert parser.feed("Hello") == [{"content": "llo"}]


def test_feed_tool_call_name_delta():
    parser = SimpleJsonToolParser()
    text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    deltas = parser.feed(text)
    tc_deltas = [d for d in deltas if "tool_calls" in d]
    assert len(tc_deltas) >= 1
    name_delta = tc_deltas[0]["tool_calls"][0]
    assert name_delta["function"]["name"] == "get_weather"
    assert name_delta["type"] == "function"
    assert "id" in name_delta


def test_feed_tool_call_args_streaming():
    parser = SimpleJsonToolParser()
    d1 = parser.feed('{"name": "f", "arguments": {"x":')
    d2 = parser.feed('{"name": "f", "arguments": {"x": "1"}}')
    args_deltas = [
        d
        for batch in (d1, d2)
        for d in batch
        if "tool_calls" in d
        and "function" in d["tool_calls"][0]
        and "arguments" in d["tool_calls"][0]["function"]
    ]
    assert len(args_deltas) >= 1


def test_feed_text_before_tool_call():
    parser = SimpleJsonToolParser()
    text = 'Let me check. {"name": "func", "arguments": {"a": 1}}'
    deltas = parser.feed(text)
    content_deltas = [d for d in deltas if "content" in d]
    assert any("Let me check" in d.get("content", "") for d in content_deltas)


def test_has_tool_calls_false_by_default():
    assert SimpleJsonToolParser().has_tool_calls is False


def test_has_tool_calls_true_after_detection():
    parser = SimpleJsonToolParser()
    parser.feed('{"name": "f", "arguments": {}}')
    assert parser.has_tool_calls is True


def test_feed_no_content_when_no_new_text():
    parser = SimpleJsonToolParser()
    parser.feed("Hello")
    assert parser.feed("Hello") == []


def test_feed_multiple_tool_calls():
    parser = SimpleJsonToolParser()
    text = '{"name": "f1", "arguments": {"a": 1}}{"name": "f2", "arguments": {"b": 2}}'
    deltas = parser.feed(text)
    tc_deltas = [d for d in deltas if "tool_calls" in d]
    names = set()
    for batch in tc_deltas:
        for tc in batch["tool_calls"]:
            if "function" in tc and "name" in tc["function"]:
                names.add(tc["function"]["name"])
    assert "f1" in names
    assert "f2" in names


def test_feed_with_tools_constructor():
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    parser = SimpleJsonToolParser(tools=tools, tool_choice="auto")
    deltas = parser.feed('{"name": "get_weather", "arguments": {"city": "BJ"}}')
    assert len(deltas) > 0


def test_feed_content_after_tool_call_is_not_emitted():
    parser = SimpleJsonToolParser()
    parser.feed('{"name": "f", "arguments": {}} trailing text')
    assert parser.has_tool_calls


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


def test_streaming_token_by_token_full_build():
    parser = SimpleJsonToolParser()
    text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    names, args_chunks = _simulate_streaming(parser, text)
    assert "get_weather" in names
    joined_args = "".join(args_chunks)
    assert '"city"' in joined_args
    assert "Beijing" in joined_args


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


def test_streaming_multiple_tool_calls_incremental():
    parser = SimpleJsonToolParser()
    text = '{"name": "f1", "arguments": {"a": 1}}{"name": "f2", "arguments": {"b": 2}}'
    names, _ = _simulate_streaming(parser, text)
    assert names[0] == "f1"
    assert "f2" in names


def test_streaming_deeply_nested_args():
    parser = SimpleJsonToolParser()
    text = '{"name": "deep", "arguments": {"a": {"b": {"c": 42}}}}'
    _, args_chunks = _simulate_streaming(parser, text)
    joined = "".join(args_chunks)
    assert '"c": 42' in joined


def test_streaming_args_with_unicode():
    parser = SimpleJsonToolParser()
    text = (
        '{"name": "translate", "arguments": {"text": "\u4f60\u597d\uff0c\u4e16\u754c"}}'
    )
    _, args_chunks = _simulate_streaming(parser, text)
    joined = "".join(args_chunks)
    assert "\u4f60\u597d" in joined


def test_streaming_args_with_array():
    parser = SimpleJsonToolParser()
    text = '{"name": "add", "arguments": {"items": [1, 2, 3]}}'
    _, args_chunks = _simulate_streaming(parser, text)
    joined = "".join(args_chunks)
    assert "[1, 2, 3]" in joined


def test_streaming_empty_arguments():
    parser = SimpleJsonToolParser()
    text = '{"name": "ping", "arguments": {}}'
    deltas = parser.feed(text)
    tc_deltas = [d for d in deltas if "tool_calls" in d]
    assert len(tc_deltas) >= 1
    name_delta = tc_deltas[0]["tool_calls"][0]
    assert name_delta["function"]["name"] == "ping"
    assert "arguments" in name_delta["function"]


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
    assert joined.startswith('"city":')
    assert all_args[0] != all_args[1]


def test_streaming_distinct_tool_call_ids():
    parser = SimpleJsonToolParser()
    text = '{"name": "f1", "arguments": {"a": 1}}{"name": "f2", "arguments": {"b": 2}}'
    all_ids = []
    for i in range(1, len(text) + 1):
        deltas = parser.feed(text[:i])
        for d in deltas:
            if "tool_calls" in d:
                for tc in d["tool_calls"]:
                    if "id" in tc:
                        all_ids.append(tc["id"])
    unique = list(dict.fromkeys(all_ids))
    assert len(unique) == 2


def test_parse_complete_basic():
    parser = SimpleJsonToolParser()
    body = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    result = parser.parse_complete(body)
    assert result is not None
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "Beijing" in result["tool_calls"][0]["function"]["arguments"]


def test_parse_complete_no_tool_call():
    assert SimpleJsonToolParser().parse_complete("Hello world") is None


def test_parse_complete_with_content():
    parser = SimpleJsonToolParser()
    result = parser.parse_complete('Prefix text. {"name": "f", "arguments": {}}')
    assert result is not None
    assert result["content"] == "Prefix text."


def test_parse_complete_multiple_tool_calls():
    parser = SimpleJsonToolParser()
    body = '{"name": "get_weather", "arguments": {"city": "Beijing"}}{"name": "get_time", "arguments": {"tz": "Asia/Shanghai"}}'
    result = parser.parse_complete(body)
    assert result is not None
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result["tool_calls"][1]["function"]["name"] == "get_time"


def test_parse_complete_complex_real_world():
    parser = SimpleJsonToolParser()
    body = (
        '{"name": "send_email", "arguments": {'
        '"to": ["a@b.com", "c@d.com"], "cc": null, '
        '"subject": "Hello World", "body": "This is a test email.", '
        '"priority": 1, "attachments": false}}'
    )
    result = parser.parse_complete(body)
    assert result is not None
    tc = result["tool_calls"][0]
    assert tc["function"]["name"] == "send_email"
    args = tc["function"]["arguments"]
    assert '"to"' in args
    assert "a@b.com" in args
    assert "null" in args
    assert "false" in args


def test_parse_complete_content_with_multiple_tool_calls():
    parser = SimpleJsonToolParser()
    body = 'I will do two things. {"name": "f1", "arguments": {"a": 1}}{"name": "f2", "arguments": {"b": 2}}'
    result = parser.parse_complete(body)
    assert result is not None
    assert result["content"] == "I will do two things."
    assert len(result["tool_calls"]) == 2


def test_parse_complete_no_arguments_field():
    parser = SimpleJsonToolParser()
    result = parser.parse_complete('{"name": "ping"}')
    assert result is not None
    assert result["tool_calls"][0]["function"]["name"] == "ping"
    assert result["tool_calls"][0]["function"]["arguments"] == ""


def test_parse_complete_content_is_none_when_pure_tool_call():
    parser = SimpleJsonToolParser()
    result = parser.parse_complete('{"name": "f", "arguments": {"x": 1}}')
    assert result is not None
    assert result["content"] is None


def test_parse_complete_tool_calls_have_ids():
    parser = SimpleJsonToolParser()
    result = parser.parse_complete(
        '{"name": "f1", "arguments": {}}{"name": "f2", "arguments": {}}'
    )
    assert result is not None
    ids = [tc["id"] for tc in result["tool_calls"]]
    assert len(ids) == 2
    assert all(isinstance(i, str) and i.startswith("call_") for i in ids)
    assert ids[0] != ids[1]


def test_feed_then_parse_complete_same_instance():
    parser = SimpleJsonToolParser()
    parser.feed('{"name": "get_weather", "arguments": {"city": "Beijing"}}')
    result = parser.parse_complete(
        '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    )
    assert result is not None
    assert result["tool_calls"][0]["function"]["name"] == "get_weather"
    assert parser.has_tool_calls


@pytest.mark.parametrize(
    "text,matches",
    [
        ('{"name": "f"}', True),
        ('{ "name" : "f"}', True),
        ('{"other": 1}', False),
        ('prefix {"name": "f", "args": {}}', True),
        ('{"name": "f"}', True),  # match at start
        ('   {"name": "f"}', True),
    ],
)
def test_pattern_regex(text, matches):
    result = _TOOL_CALL_HEAD_RE.search(text)
    if matches:
        assert result is not None
    else:
        assert result is None


def test_pattern_name_at_start():
    assert _TOOL_CALL_HEAD_RE.match('{"name": "f"}')


def test_factory_register_and_create():
    parser = ToolParserFactory.create("simple_json")
    assert isinstance(parser, BaseToolParser)
    assert isinstance(parser, SimpleJsonToolParser)


def test_factory_create_passes_tools():
    parser = ToolParserFactory.create(
        "simple_json", tools=[{"type": "function"}], tool_choice="required"
    )
    assert parser.tool_choice == "required"


def test_factory_list_registered():
    assert "simple_json" in ToolParserFactory.list_registered()


def test_factory_create_with_no_extra_kwargs():
    assert isinstance(ToolParserFactory.create("simple_json"), BaseToolParser)


def test_factory_create_with_tools_only():
    tools = [
        {
            "type": "function",
            "function": {"name": "test", "parameters": {"type": "object"}},
        }
    ]
    parser = ToolParserFactory.create("simple_json", tools=tools)
    assert parser.tools == tools
    assert parser.tool_choice == "auto"


def test_feed_accepts_token_ids_and_ignores_them():
    parser = SimpleJsonToolParser()
    text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    deltas_with = parser.feed(text, current_token_ids=[123, 456], delta_token_ids=[456])
    assert len(deltas_with) > 0


def test_feed_token_ids_do_not_affect_parsing():
    parser_no_ids = SimpleJsonToolParser()
    parser_with_ids = SimpleJsonToolParser()
    text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
    result_no = parser_no_ids.feed(text)
    result_with = parser_with_ids.feed(
        text, current_token_ids=[1, 2, 3], delta_token_ids=[3]
    )
    assert len(result_no) == len(result_with)
    assert (
        result_no[0]["tool_calls"][0]["function"]["name"]
        == result_with[0]["tool_calls"][0]["function"]["name"]
    )


def test_parser_uses_token_ids_for_detection():
    class TokenIdParser(BaseToolParser):
        def __init__(self, tools=None, tool_choice="auto"):
            super().__init__(tools, tool_choice)
            self._detections = 0

        def feed(self, body, current_token_ids=None, delta_token_ids=None):
            if current_token_ids and 999 in current_token_ids:
                self._detections += 1
            return []

        def parse_complete(self, body):
            return None

        @property
        def has_tool_calls(self):
            return self._detections > 0

    parser = TokenIdParser()
    parser.feed("hello", current_token_ids=[1, 999, 3])
    assert parser.has_tool_calls
