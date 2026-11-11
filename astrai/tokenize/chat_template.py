import json
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from functools import cached_property
from typing import Any, Dict, List, Optional

from jinja2 import Template, TemplateError, nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

MessageType = Dict[str, Any]


def _raise_exception(message):
    raise TemplateError(message)


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _strftime_now(format):
    return datetime.now().strftime(format)


_generation_tracker = ContextVar("generation_tracker", default=None)


class GenerationExtension(Extension):
    """Record character spans without inserting markers into rendered text."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("capture"), [], [], body).set_lineno(
            lineno
        )

    def capture(self, caller):
        tracker = _generation_tracker.get()
        if tracker is None:
            return caller()
        if tracker["inside"]:
            raise ValueError("Nested generation blocks are not supported")
        tracker["inside"] = True
        try:
            text = caller()
        finally:
            tracker["inside"] = False
        start = tracker["position"]
        tracker["spans"].append((start, start + len(text)))
        return text


def _validate_generation_blocks(node, parents=()):
    """Generation spans require blocks that stream directly to the root."""
    is_generation = (
        isinstance(node, nodes.CallBlock)
        and isinstance(node.call.node, nodes.ExtensionAttribute)
        and node.call.node.identifier == GenerationExtension.identifier
    )
    if is_generation and any(
        not isinstance(parent, (nodes.Template, nodes.For, nodes.If, nodes.Scope))
        or (isinstance(parent, nodes.For) and parent.recursive)
        for parent in parents
    ):
        raise ValueError(
            "Generation blocks cannot be nested or placed in buffered constructs "
            "(macros, set/filter/call blocks, or recursive loops)"
        )
    for child in node.iter_child_nodes():
        _validate_generation_blocks(child, parents + (node,))


def normalize_tool_messages(messages):
    """Accept API JSON-string arguments and template-native objects without mutation."""
    result = deepcopy(messages)
    for message in result:
        calls = message.get("tool_calls")
        if calls is None:
            continue
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        for call in calls:
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                raise ValueError("Only function tool calls are supported")
            function = call.get("function")
            if (
                not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"]
            ):
                raise ValueError("Tool calls require a function name")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Tool arguments must contain valid JSON"
                    ) from error
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            function["arguments"] = arguments
    return result


class ChatTemplate:
    """A chat template with Jinja2 rendering support.

    Attributes:
        name: Unique identifier for the template.
        template_str: Jinja2 template string.
        description: Optional description.
        default_variables: Optional dictionary of default variable values.
        special_tokens: Optional dictionary mapping token names to their string values.
    """

    def __init__(
        self,
        name: str = "",
        template_str: str = "",
        description: str = "",
        default_variables: Optional[Dict[str, Any]] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.template_str = template_str
        self.description = description
        self.default_variables = default_variables or {}
        self.special_tokens = special_tokens or {}

    @cached_property
    def _compiled(self) -> Template:
        """Lazy-compiled Jinja2 template, cached on first access.

        The compiled :class:`~jinja2.Template` holds a dynamically-generated
        ``root`` render function whose ``__module__`` is ``None``; under
        ``pickle`` it falls back to ``__main__`` and breaks ``spawn``-based
        multiprocessing.  :meth:`__getstate__` drops the cached template so
        that pickle serialises only ``template_str``; each worker rebuilds
        the cache on first render.
        """
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols", GenerationExtension],
        )
        environment.filters["tojson"] = _tojson
        environment.globals.update(
            raise_exception=_raise_exception, strftime_now=_strftime_now
        )
        parsed = environment.parse(self.template_str)
        _validate_generation_blocks(parsed)
        return environment.from_string(parsed)

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude the cached Jinja2 template from pickling.

        ``Template.root_render_func`` is a dynamically generated closure
        that cannot be pickled by reference.  Dropping ``_compiled`` here
        lets :class:`cached_property` rebuild it on first access after
        unpickle.
        """
        state = self.__dict__.copy()
        state.pop("_compiled", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    @classmethod
    def from_string(
        cls,
        template_str: str,
        description: str = "",
        default_variables: Optional[Dict[str, Any]] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> "ChatTemplate":
        """Create a ChatTemplate instance directly from a template string."""
        return cls(
            name="",
            template_str=template_str,
            description=description,
            default_variables=default_variables,
            special_tokens=special_tokens,
        )

    def render(
        self,
        messages: List[MessageType],
        system_prompt: Optional[str] = None,
        **extra_variables: Any,
    ) -> str:
        """Render the template with given messages and variables.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional system prompt string.
            **extra_variables: Additional variables to pass to the template.
                These override default_variables and special_tokens.

        Returns:
            Rendered prompt string.
        """
        # Merge default variables, special tokens, and extra variables
        variables = {**self.default_variables, **self.special_tokens, **extra_variables}
        variables["messages"] = normalize_tool_messages(messages)
        if system_prompt is not None:
            variables["system_prompt"] = system_prompt

        return self._compiled.render(**variables)

    def render_with_spans(self, messages, **extra_variables):
        """Render once and track template-declared generation regions."""
        variables = {**self.default_variables, **self.special_tokens, **extra_variables}
        variables["messages"] = normalize_tool_messages(messages)
        tracker = {"position": 0, "spans": [], "inside": False}
        token = _generation_tracker.set(tracker)
        parts = []
        try:
            for part in self._compiled.generate(**variables):
                parts.append(part)
                tracker["position"] += len(part)
        finally:
            _generation_tracker.reset(token)
        return "".join(parts), tracker["spans"]
