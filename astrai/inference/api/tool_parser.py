"""Tool call parsers for extracting structured tool calls from model output.

Patterned after vLLM's ToolParser abstraction. Each parser knows how to
detect and incrementally extract tool calls from raw generated text.

Subclasses may optionally consume ``token_ids`` for token-level parsing
(e.g. Harmony / VLM-style parsers).
"""

import json
import re
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from astrai.factory import BaseFactory


class BaseToolParser(ABC):
    """Abstract tool call parser — one instance per request.

    Maintains streaming state internally so that each call to :meth:`feed`
    can diff against previously emitted content.

    Parameters
    ----------
    tools : list of dict, optional
        Tool definitions from the request.
    tool_choice : str
        ``"auto"`` / ``"required"`` / ``"none"`` or a named tool choice
        dict.
    """

    def __init__(self, tools: Optional[List[Dict]] = None, tool_choice: str = "auto"):
        self.tools = tools or []
        self.tool_choice = tool_choice

    @abstractmethod
    def feed(
        self,
        body: str,
        current_token_ids: Optional[List[int]] = None,
        delta_token_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Feed the *full* accumulated text each step.

        Returns a list of delta dicts to emit. Each delta is one of:

        - ``{"content": "text"}``      — plain text delta
        - ``{"tool_calls": [...]}``    — tool-call delta (OpenAI format)

        Returns an empty list when nothing new should be emitted.

        Parameters
        ----------
        body : str
            The complete accumulated generated text so far.
        current_token_ids : list of int, optional
            All token IDs decoded into *body* (cumulative).
        delta_token_ids : list of int, optional
            Only the token IDs for this chunk.
        """

    @abstractmethod
    def parse_complete(self, body: str) -> Optional[Dict]:
        """Parse the *complete* generated text after generation ends.

        Returns ``None`` when no tool calls were found, otherwise a dict
        with ``content`` (str or None) and ``tool_calls`` (list of dicts).
        """

    @property
    @abstractmethod
    def has_tool_calls(self) -> bool:
        """True if the parser detected at least one tool call in the stream."""


class ToolParserFactory(BaseFactory["BaseToolParser"]):
    pass


_TOOL_CALL_HEAD_RE = re.compile(r'\{\s*"name"\s*:')


def _scan_json(text: str, start: int = 0):
    """Scan for a complete JSON object starting at *start*.

    Returns ``(end, complete)`` where *end* is one-past the closing
    brace (or ``len(text)`` if unclosed), and *complete* is a bool.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1, True
    return len(text), False


def _parse_tool_call_json(json_str: str, complete: bool):
    """Extract *name* and *arguments* from a tool-call JSON string.

    Returns ``(name, args, valid)``.
    """
    if complete:
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            return None, "", False
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            return None, "", False
        args = obj.get("arguments")
        if isinstance(args, dict):
            if not args:
                args = ""
            else:
                args = json.dumps(args, ensure_ascii=False)
                args = args[1:-1].rstrip()
        elif isinstance(args, list):
            args = json.dumps(args, ensure_ascii=False) if args else ""
        elif isinstance(args, str):
            pass
        else:
            args = str(args) if args is not None else ""
        return name, args, True

    name_match = re.search(r'"name"\s*:\s*"([^"]*)"', json_str)
    if not name_match:
        return None, "", False
    name = name_match.group(1)

    args_match = re.search(r'"arguments"\s*:\s*(.*)', json_str, re.DOTALL)
    if not args_match:
        return name, "", True

    raw = args_match.group(1).rstrip()
    if raw.startswith("{"):
        inner = raw[1:].rstrip()
        if inner.endswith("}"):
            inner = inner[:-1].rstrip()
        raw = inner
    return name, raw, True


def _find_tool_calls(text: str, start_pos: int = 0):
    """Find all complete ``{...}`` tool-call objects in *text*.

    Returns a list of dicts with keys *start*, *end*, *name*, *args*,
    *complete*.
    """
    results = []
    pos = start_pos

    while True:
        brace = text.find("{", pos)
        if brace == -1:
            break

        end, complete = _scan_json(text, brace)
        if not complete:
            break

        json_str = text[brace:end]

        name, args, valid = _parse_tool_call_json(json_str, complete=True)
        if not valid or name is None:
            pos = end
            continue

        results.append(
            {
                "start": brace,
                "end": end,
                "name": name,
                "args": args,
                "complete": True,
            }
        )
        pos = end

    return results


def _find_partial_tool_call(text: str, start_pos: int = 0):
    """Find one incomplete (still-generating) tool-call JSON object."""
    brace = text.find("{", start_pos)
    if brace == -1:
        return None

    json_str = text[brace:]
    if '"name"' not in json_str:
        return None

    name, args, valid = _parse_tool_call_json(json_str, complete=False)
    if not valid or name is None:
        return None

    return {
        "start": brace,
        "name": name,
        "args": args,
        "complete": False,
    }


@ToolParserFactory.register("simple_json")
class SimpleJsonToolParser(BaseToolParser):
    """Parser for models that output tool calls as plain JSON objects.

    Detects ``{"name": "<func>", "arguments": {...}}`` anywhere in the
    generated text.  Handles single and (non-overlapping) multiple tool
    calls.  Text preceding the first tool call is emitted as plain
    ``content`` deltas.
    """

    def __init__(self, tools=None, tool_choice="auto"):
        super().__init__(tools, tool_choice)
        self._emitted_content_len = 0
        self._tc_state: List[Dict] = []
        self._has_tool_calls = False

    # -------------------------------------------------------------- feed

    def feed(
        self,
        body: str,
        current_token_ids: Optional[List[int]] = None,
        delta_token_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        deltas: List[Dict] = []

        completed = _find_tool_calls(body)

        if not completed:
            partial = _find_partial_tool_call(body)
            if not partial:
                return self._emit_plain_content(body, deltas)
            all_tcs = [partial]
        else:
            all_tcs = completed
            partial = _find_partial_tool_call(body, completed[-1]["end"])
            if partial:
                all_tcs = completed + [partial]

        first_start = all_tcs[0]["start"]
        if first_start > self._emitted_content_len:
            content = body[self._emitted_content_len : first_start]
            self._emitted_content_len = first_start
            if content:
                deltas.append({"content": content})

        for i, tc in enumerate(all_tcs):
            if i >= len(self._tc_state):
                self._tc_state.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "name_emitted": False,
                        "args_emitted_len": 0,
                    }
                )
                self._has_tool_calls = True
            st = self._tc_state[i]

            if not st["name_emitted"]:
                st["name_emitted"] = True
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": st["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": ""},
                            }
                        ]
                    }
                )

            new_args = tc["args"]
            if len(new_args) > st["args_emitted_len"]:
                diff = new_args[st["args_emitted_len"] :]
                st["args_emitted_len"] = len(new_args)
                deltas.append(
                    {
                        "tool_calls": [
                            {
                                "index": i,
                                "function": {"arguments": diff},
                            }
                        ]
                    }
                )

        return deltas

    def _emit_plain_content(self, body: str, deltas: List[Dict]) -> List[Dict]:
        new_content = body[self._emitted_content_len :]
        if new_content:
            self._emitted_content_len = len(body)
            deltas.append({"content": new_content})
        return deltas

    # -------------------------------------------------------- complete

    def parse_complete(self, body: str) -> Optional[Dict]:
        completed = _find_tool_calls(body)
        if not completed:
            return None

        content = body[: completed[0]["start"]].strip() or None
        tool_calls = []
        for i, tc in enumerate(completed):
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["args"],
                    },
                }
            )
        return {"content": content, "tool_calls": tool_calls}

    @property
    def has_tool_calls(self) -> bool:
        return self._has_tool_calls
