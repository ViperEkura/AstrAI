"""Base factory with decorator-based registration and kwarg-filtered instantiation."""

import inspect
import sys
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
)
from typing import get_args as _get_args
from typing import get_origin as _get_origin

T = TypeVar("T")


def _resolve_type(
    arg: Union[Type, str, ForwardRef], factory_cls: type
) -> Optional[Type]:
    """Resolve a generic type-arg (str forward-ref, ForwardRef, or class)."""
    if not isinstance(arg, (str, ForwardRef)):
        return arg

    name = arg if isinstance(arg, str) else arg.__forward_arg__
    if name == factory_cls.__name__:
        return factory_cls

    mod = sys.modules.get(factory_cls.__module__)
    if mod is None:
        return None
    ns = vars(mod)

    if isinstance(arg, ForwardRef):
        return arg._evaluate(ns, None, frozenset(), recursive_guard=frozenset())

    return ns.get(name)


class BaseFactory(ABC, Generic[T]):
    """Generic factory with decorator-based component registration.

        class MyFactory(BaseFactory[MyBase]):
            pass

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
            if _get_origin(orig_base) is BaseFactory:
                (arg,) = _get_args(orig_base)
                cls._entries = {}
                cls._component_base = _resolve_type(arg, cls)
                return

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a component class.

        Validates that the decorated class inherits from the generic
        type parameter ``T`` declared on the factory.
        """

        def decorator(component_cls: Type[T]) -> Type[T]:
            cls._validate_component(component_cls)
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
        entry = cls._entries.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown component: '{name}'. Supported types: {sorted(cls._entries)}"
            )
        component_cls = entry
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
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return component_cls(*args, **kwargs)

    @classmethod
    def _validate_component(cls, component_cls: Type[T]):
        """Validate the decorated class inherits from the factory's base type.

        Override for custom validation beyond ``issubclass``.
        """
        base = cls._component_base
        if base is not None and not issubclass(component_cls, base):
            raise TypeError(
                f"{component_cls.__name__} must inherit from {base.__name__}"
            )

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
