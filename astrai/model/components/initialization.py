"""Thread-local control of initialization during complete checkpoint loads."""

from contextlib import contextmanager
from contextvars import ContextVar

_skip_parameter_init = ContextVar("astrai_skip_parameter_init", default=False)


@contextmanager
def skip_parameter_init(enabled: bool):
    token = _skip_parameter_init.set(enabled)
    try:
        yield
    finally:
        _skip_parameter_init.reset(token)


def should_initialize() -> bool:
    return not _skip_parameter_init.get()
