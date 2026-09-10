from pathlib import Path

import click
import torch
from click.core import ParameterSource

from astrai.config.cli import (
    GroupedCommand,
    OptSpec,
    apply_specs,
    merge_yaml_into_kwargs,
)
from astrai.inference import run_server

_DTYPES = ["bfloat16", "float16", "float32"]
_SERVER_KEYS = (
    "host",
    "port",
    "reload",
    "param_path",
    "device",
    "dtype",
    "max_batch_size",
    "max_seq_len",
)


def _merge_yaml_into_kwargs(
    config_path: str,
    passed_kwargs: dict,
    explicit_keys: set[str] | None = None,
) -> dict:
    """Merge Click defaults, YAML server values, then explicit CLI values."""
    return merge_yaml_into_kwargs(
        config_path,
        passed_kwargs,
        explicit_keys,
        sections=("server",),
        allowed_keys=_SERVER_KEYS,
    )


_SPECS = [
    OptSpec(
        "config_path",
        "Server",
        type=click.Path(exists=True, dir_okay=False),
        param_decls=("--config", "-c", "config_path"),
        help="Serving YAML config. CLI flags override YAML values.",
    ),
    OptSpec("host", "Server", type=str, default="0.0.0.0", help="Host address."),
    OptSpec("port", "Server", type=int, default=8000, help="Port number."),
    OptSpec(
        "reload",
        "Server",
        is_flag=True,
        default=False,
        help="Enable auto-reload for development.",
    ),
    OptSpec(
        "param_path",
        "Model",
        type=click.Path(exists=True),
        default=None,
        help="Path to model parameters.",
    ),
    OptSpec(
        "device", "Model", type=str, default="cuda", help="Device to load model on."
    ),
    OptSpec(
        "dtype",
        "Model",
        choices=_DTYPES,
        default="bfloat16",
        help="Data type for model weights.",
    ),
    OptSpec(
        "max_batch_size",
        "Performance",
        type=int,
        default=16,
        help="Maximum batch size for continuous batching.",
    ),
    OptSpec(
        "max_seq_len",
        "Performance",
        type=int,
        default=None,
        help="Maximum sequence length (KV cache size + prompt truncation). "
        "Uses model config if not set.",
    ),
]


def _as_int(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise click.UsageError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise click.UsageError(f"{name} must be an integer, got {value!r}") from None


def _resolve_server_config(
    config_path: str,
    passed_kwargs: dict,
    explicit_keys: set[str] | None = None,
) -> dict:
    """Merge YAML values, then coerce and validate the resolved settings.

    ``explicit_keys`` are CLI flags that win over YAML; when None, YAML values
    win over Click defaults.
    """
    merged = _merge_yaml_into_kwargs(config_path, passed_kwargs, explicit_keys or set())
    resolved = dict(merged)
    resolved["port"] = _as_int(resolved["port"], "server.port") or 8000
    resolved["max_batch_size"] = (
        _as_int(resolved["max_batch_size"], "server.max_batch_size") or 16
    )
    resolved["max_seq_len"] = _as_int(resolved["max_seq_len"], "server.max_seq_len")
    resolved["reload"] = bool(resolved["reload"])
    if resolved["dtype"] not in _DTYPES:
        raise click.UsageError(
            f"server.dtype must be one of {', '.join(_DTYPES)}, got {resolved['dtype']!r}"
        )
    return resolved


@click.command(
    name="serve",
    cls=GroupedCommand,
    help="Launch inference server (OpenAI-compatible API).",
)
@apply_specs(_SPECS)
@click.pass_context
def server_command(ctx, config_path, **kwargs):
    """Launch inference server (OpenAI-compatible API)."""
    if config_path:
        explicit_keys = {
            key
            for key in kwargs
            if ctx.get_parameter_source(key) is ParameterSource.COMMANDLINE
        }
        kwargs = _resolve_server_config(config_path, kwargs, explicit_keys)
        click.echo(f"Config: {config_path}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    project_root = Path(__file__).parent.parent.parent
    kwargs["param_path"] = kwargs.get("param_path") or str(project_root / "params")

    click.echo(f"Starting server on http://{kwargs['host']}:{kwargs['port']}")
    click.echo(
        f"Model: {kwargs['param_path']}  |  "
        f"Device: {kwargs['device']}  |  Dtype: {kwargs['dtype']}"
    )
    run_server(
        host=kwargs["host"],
        port=kwargs["port"],
        reload=kwargs["reload"],
        device=kwargs["device"],
        dtype=dtype_map[kwargs["dtype"]],
        param_path=Path(kwargs["param_path"]),
        max_batch_size=kwargs["max_batch_size"],
        max_seq_len=kwargs["max_seq_len"],
    )


if __name__ == "__main__":
    server_command()
