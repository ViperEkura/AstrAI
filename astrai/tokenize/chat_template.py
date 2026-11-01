import json
from datetime import datetime
from functools import cached_property
from typing import Any, Dict, List, Optional

from jinja2 import Template, TemplateError
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
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        environment.filters["tojson"] = _tojson
        environment.globals.update(
            raise_exception=_raise_exception, strftime_now=_strftime_now
        )
        return environment.from_string(self.template_str)

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
        variables["messages"] = messages
        if system_prompt is not None:
            variables["system_prompt"] = system_prompt

        return self._compiled.render(**variables)
