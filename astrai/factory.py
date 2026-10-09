"""Base factory with decorator-based registration and kwarg-filtered instantiation."""

import inspect
import sys
import warnings
from abc import ABC
from typing import (
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

T = TypeVar("T")


def _resolve_base_type(
    arg: Union[Type, str, ForwardRef], factory_cls: type
) -> Optional[Type]:
    """Resolve the generic type-arg T to a concrete class.

    - Concrete class (``BaseFactory[MyBase]``): returned directly.
    - Forward reference (``BaseFactory["MyBase"]``): ``Base["X"]``
      produces a ``ForwardRef("X")`` at class-creation time.  We
      extract the name and evaluate it in the factory module's
      global namespace — the same mechanism ``typing.get_type_hints``
      uses internally.
    """
    if isinstance(arg, type):
        return arg

    if isinstance(arg, str):
        name = arg
    elif isinstance(arg, ForwardRef):
        name = arg.__forward_arg__
    else:
        return None

    mod = sys.modules.get(factory_cls.__module__)
    if mod is None:
        return None
    try:
        return eval(name, vars(mod))  # noqa: S307
    except NameError:
        return None


def _validate_component(component_cls: Type, base: Optional[Type]) -> None:
    """Validate that *component_cls* inherits from *base*.

    No-op when *base* is ``None`` (e.g. forward-ref resolution failed).
    """
    if base is not None and not issubclass(component_cls, base):
        raise TypeError(f"{component_cls.__name__} must inherit from {base.__name__}")


class BaseFactory(ABC, Generic[T]):
    """Generic factory with decorator-based registration.

    Create a factory by subclassing with the desired base type::

        class MyFactory(BaseFactory[MyBase]):
            pass

    Register components with the ``register`` decorator::

        @MyFactory.register("custom")
        class CustomComponent(MyBase):
            ...

        obj = MyFactory.create("custom", *args, **kwargs)

    ``create()`` filters kwargs to match the component's ``__init__``
    signature so components don't need ``**kwargs`` just to absorb
    unrelated parameters.
    """

    _entries: Dict[str, Type[T]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for orig_base in getattr(cls, "__orig_bases__", ()):
            if get_origin(orig_base) is BaseFactory:
                (arg,) = get_args(orig_base)
                cls._entries = {}
                cls._component_base = _resolve_base_type(arg, cls)
                return

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a component class.

        Validates that the decorated class inherits from the generic
        type parameter ``T`` declared on the factory.
        """

        def decorator(component_cls: Type[T]) -> Type[T]:
            _validate_component(component_cls, cls._component_base)
            if name in cls._entries:
                raise ValueError(f"Component '{name}' is already registered")
            cls._entries[name] = component_cls
            return component_cls

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> T:
        """Create a component instance by name, filtering kwargs to match
        the component's ``__init__`` signature.
        """
        component_cls = cls._entries.get(name)
        if component_cls is None:
            raise ValueError(
                f"Unknown component: '{name}'. Supported types: {sorted(cls._entries)}"
            )
        sig = inspect.signature(component_cls.__init__)
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_kwargs:
            valid = {
                p.name
                for p in sig.parameters.values()
                if p.name != "self" and p.kind != inspect.Parameter.VAR_KEYWORD
            }
            discarded = sorted(set(kwargs) - valid)
            if discarded:
                warnings.warn(
                    f"{cls.__name__}.create({name!r}) ignored arguments {discarded}; "
                    "this will raise TypeError in the next minor release",
                    DeprecationWarning,
                    stacklevel=2,
                )
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return component_cls(*args, **kwargs)

    @classmethod
    def create_checked(cls, name: str, *args, **kwargs) -> T:
        """Instantiate with Python's normal signature validation."""
        component_cls = cls.get_component_class(name)
        inspect.signature(component_cls).bind(*args, **kwargs)
        return component_cls(*args, **kwargs)

    @classmethod
    def create_from_config(cls, name: str, config: dict, **explicit) -> T:
        """Map a broad model config to one registered component's parameters."""
        component_cls = cls.get_component_class(name)
        sig = inspect.signature(component_cls)
        valid = set(sig.parameters)
        kwargs = {key: value for key, value in config.items() if key in valid}
        kwargs.update(explicit)
        return cls.create_checked(name, **kwargs)

    @classmethod
    def get_component_class(cls, name: str) -> Type[T]:
        """Get the registered component class without instantiating it."""
        entry = cls._entries.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown component: '{name}'. Supported types: {sorted(cls._entries)}"
            )
        return entry

    @classmethod
    def list_registered(cls) -> List[str]:
        """List all registered component names."""
        return sorted(cls._entries)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a component name is registered."""
        return name in cls._entries
