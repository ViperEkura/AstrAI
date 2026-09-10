"""Generate grouped click CLIs from pydantic config fields.

A config class alone does not make a CLI: some options need defaults that
differ from the config defaults (e.g. ``num_workers``), choices come from
factory registries or frozenset validators, and values merge across three
layers (option defaults -> YAML -> explicit CLI flags). ``OptSpec`` records
those overrides in a declarative table; types and defaults are inferred
from the backing config field wherever the spec leaves them ``AUTO``.
"""

import dataclasses
import re
import types
import typing as t
from collections import OrderedDict
from collections.abc import Sequence

import click
import yaml


class GroupedOption(click.Option):
    """A ``click.Option`` that carries a ``group`` label for help output."""

    def __init__(self, *args, group: str = "Options", **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group


class GroupedCommand(click.Command):
    """A ``click.Command`` that renders options grouped by their ``group``."""

    def format_options(self, ctx, formatter):
        groups: OrderedDict[str, list] = OrderedDict()
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is None:
                continue
            group = getattr(param, "group", "Options")
            groups.setdefault(group, []).append(record)
        for group_name, records in groups.items():
            with formatter.section(group_name):
                formatter.write_dl(records)


def opt(*param_decls, group: str, **kwargs):
    """Shorthand for ``click.option`` that tags the option with a group."""
    kwargs.setdefault("cls", GroupedOption)
    kwargs["group"] = group
    return click.option(*param_decls, **kwargs)


class _Auto:
    def __repr__(self) -> str:
        return "AUTO"


AUTO = _Auto()

_YAML_FLOAT_PATTERN = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)

DEFAULT_YAML_SECTIONS = ("model", "data", "parallel", "training", "ckpt", "log")


def _enable_yaml12_floats() -> None:
    """PyYAML implements YAML 1.1, where ``2e-5`` parses as a string; switch its
    float resolver to the YAML 1.2 core schema so scientific notation works."""
    yaml.SafeLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float", _YAML_FLOAT_PATTERN, list("-+0123456789.")
    )


def merge_yaml_into_kwargs(
    config_path: str,
    passed_kwargs: dict,
    explicit_keys: set[str] | None = None,
    sections: Sequence[str] = DEFAULT_YAML_SECTIONS,
    allowed_keys: Sequence[str] | None = None,
) -> dict:
    """Merge option defaults, YAML values, then explicit CLI values.

    ``sections`` selects which top-level YAML mappings feed the flat kwargs
    namespace. ``allowed_keys`` optionally restricts the accepted keys across
    those sections: unknown keys are warned about once and dropped.
    """
    _enable_yaml12_floats()

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise click.UsageError(f"config must be a mapping: {config_path}")

    merged = dict(passed_kwargs)
    seen: set[str] = set()
    for section in sections:
        values = cfg.get(section) or {}
        if not isinstance(values, dict):
            raise click.UsageError(f"top-level {section} section must be a mapping")
        if allowed_keys is None:
            merged.update(values)
            seen.update(values)
        else:
            merged.update({k: v for k, v in values.items() if k in allowed_keys})
            seen.update(values)

    if allowed_keys is not None:
        unknown = sorted(seen - set(allowed_keys))
        if unknown:
            click.echo(
                f"Warning: ignoring unknown config keys: {', '.join(unknown)}",
                err=True,
            )

    if explicit_keys is None:
        explicit_keys = set(passed_kwargs)
    for key in explicit_keys:
        if key in passed_kwargs:
            merged[key] = passed_kwargs[key]

    return merged


@dataclasses.dataclass(frozen=True)
class OptSpec:
    """One CLI option, optionally backed by a field of a config class.

    ``type``/``default`` left as ``AUTO`` are inferred from the backing
    config field (bool becomes a ``--x/--no-x`` pair, lists become
    repeatable options). Standalone specs for CLI-only options must carry
    ``type`` or ``default`` explicitly. ``choices`` overrides the inferred
    type with ``click.Choice``.
    """

    name: str
    group: str
    type: t.Any = AUTO
    default: t.Any = AUTO
    help: str | None = None
    choices: Sequence[str] | None = None
    multiple: bool = False
    is_flag: bool = False
    required: bool = False
    param_decls: tuple[str, ...] = ()


def _resolve_hints(config_cls: type) -> dict[str, t.Any]:
    """Resolve field annotations with ``typing.get_type_hints``.

    Raw ``Field.type`` stays a string under PEP 563 (``from __future__
    import annotations``) and never mentions ``types.UnionType``; resolved
    hints cover old-style ``Optional[X]``/``Union[X, None]``, new-style
    ``X | None``, and stringified forward references alike.
    """
    try:
        return t.get_type_hints(config_cls)
    except Exception:
        return {}


def _unwrap_optional(annotation: t.Any) -> t.Any | None:
    """Return the single non-None member of an optional annotation."""
    if annotation is None or isinstance(annotation, str):
        return None
    origin = t.get_origin(annotation)
    if origin is not t.Union and origin is not types.UnionType:
        return None
    args = [a for a in t.get_args(annotation) if a is not type(None)]
    if len(args) == 1:
        return args[0]
    return None


def _annotation(
    spec: OptSpec,
    field: dataclasses.Field | None,
    hints: dict[str, t.Any] | None = None,
) -> t.Any:
    if field is not None:
        if hints and spec.name in hints:
            return hints[spec.name]
        return field.type
    if spec.type is not AUTO:
        return spec.type
    return None


def _click_type(
    spec: OptSpec,
    field: dataclasses.Field | None,
    hints: dict[str, t.Any] | None = None,
    annotation: t.Any = None,
) -> t.Any:
    if spec.choices is not None:
        return click.Choice(list(spec.choices))
    if annotation is None:
        if spec.type is not AUTO and spec.type is not bool:
            return spec.type
        annotation = _annotation(spec, field, hints)
    inner = _unwrap_optional(annotation)
    if inner is not None:
        return _click_type(spec, field, hints, annotation=inner)
    origin = t.get_origin(annotation)
    if annotation is bool or spec.type is bool:
        return click.BOOL
    if annotation is int:
        return click.INT
    if annotation is float:
        return click.FLOAT
    if annotation is str:
        return click.STRING
    if origin in (list, tuple) or annotation in (list, tuple):
        return click.STRING
    raise TypeError(
        f"cannot infer a click type for option {spec.name!r}; set OptSpec.type"
    )


def _resolve_default(spec: OptSpec, field: dataclasses.Field | None) -> t.Any:
    if spec.default is not AUTO:
        return spec.default
    if field is not None:
        if field.default is not dataclasses.MISSING:
            return field.default
        if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            return field.default_factory()  # type: ignore[misc]
    return None


def _is_flag_pair(
    spec: OptSpec,
    field: dataclasses.Field | None,
    hints: dict[str, t.Any] | None = None,
) -> bool:
    if spec.is_flag:
        return False
    annotation = _annotation(spec, field, hints)
    return annotation is bool


def option_from_spec(
    spec: OptSpec,
    fields_by_name: dict[str, dataclasses.Field] | None = None,
    hints: dict[str, t.Any] | None = None,
) -> t.Callable:
    """Build a grouped ``click.option`` decorator from one spec."""
    fields_by_name = fields_by_name or {}
    hints = hints or {}
    field = fields_by_name.get(spec.name)

    kwargs: dict[str, t.Any] = {
        "cls": GroupedOption,
        "group": spec.group,
    }
    if spec.help is not None:
        kwargs["help"] = spec.help
    if spec.required:
        kwargs["required"] = True

    if spec.is_flag:
        decls = spec.param_decls or (f"--{spec.name}",)
        kwargs["is_flag"] = True
        kwargs["default"] = _resolve_default(spec, field)
    elif _is_flag_pair(spec, field, hints):
        if spec.param_decls:
            raise ValueError(
                f"flag pair {spec.name!r} does not support custom param_decls"
            )
        decls = (f"--{spec.name}/--no-{spec.name}",)
        kwargs["type"] = click.BOOL
        kwargs["default"] = _resolve_default(spec, field)
    else:
        decls = spec.param_decls or (f"--{spec.name}",)
        kwargs["type"] = _click_type(spec, field, hints)
        kwargs["default"] = _resolve_default(spec, field)
        multiple = spec.multiple
        if not multiple and field is not None:
            annotation = _annotation(spec, field, hints)
            origin = t.get_origin(annotation)
            multiple = origin in (list, tuple) or annotation in (list, tuple)
        if multiple:
            kwargs["multiple"] = True
            if isinstance(kwargs["default"], list):
                kwargs["default"] = tuple(kwargs["default"])
    return opt(*decls, **kwargs)


def apply_specs(
    specs: Sequence[OptSpec],
    config_cls: type | None = None,
) -> t.Callable:
    """Apply a table of specs as click options, in table order.

    Options render top-to-bottom in the command's help output following the
    table order (click reverses decorator application, so specs are applied
    reversed). ``config_cls`` supplies type/default inference for specs whose
    name matches one of its fields.
    """
    fields_by_name: dict[str, dataclasses.Field] = {}
    hints: dict[str, t.Any] = {}
    if config_cls is not None:
        fields_by_name = {f.name: f for f in dataclasses.fields(config_cls)}
        hints = _resolve_hints(config_cls)

    def decorator(func):
        for spec in reversed(specs):
            func = option_from_spec(spec, fields_by_name, hints)(func)
        return func

    return decorator
