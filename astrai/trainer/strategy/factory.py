"""Registry and capability declarations for training strategies."""

import inspect
from typing import Callable, Optional, Set, Type

from astrai.config.train_config import _STRATEGY_CAPABILITIES, StrategyCapabilities
from astrai.factory import BaseFactory
from astrai.trainer.strategy.base import BaseStrategy


class StrategyFactory(BaseFactory[BaseStrategy]):
    _capabilities = _STRATEGY_CAPABILITIES

    @classmethod
    def register(
        cls, name: str, *, capabilities: Optional[StrategyCapabilities] = None
    ) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
        register_component = super().register(name)

        def decorator(component_cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
            component_cls = register_component(component_cls)
            cls._capabilities[name] = capabilities or StrategyCapabilities()
            return component_cls

        return decorator

    @classmethod
    def capabilities(cls, name: str) -> StrategyCapabilities:
        cls.get_component_class(name)
        return cls._capabilities[name]

    @classmethod
    def option_names(cls, name: str) -> Set[str]:
        """Return constructor and explicitly declared options for a strategy."""
        component_cls = cls.get_component_class(name)
        allowed = {
            "executor",
            "moe_aux_loss_coef",
            "rl_update_epochs",
            "rl_minibatch_prompts",
            "gradient_chunked_logprobs",
        }
        for base in component_cls.__mro__:
            if base is object:
                break
            try:
                signature = inspect.signature(base.__init__)
            except (ValueError, TypeError):
                continue
            allowed.update(
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.name != "self"
                and parameter.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            )
        allowed.update(getattr(component_cls, "accepted_options", ()))
        if cls.capabilities(name).online:
            # Rollout setup consumes group_size even for strategies whose
            # loss constructor does not name it (online DPO/PPO).
            allowed.add("group_size")
        return allowed

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> BaseStrategy:
        unknown = sorted(set(kwargs) - cls.option_names(name))
        if unknown:
            raise TypeError(
                f"{name} strategy got unknown arguments: {', '.join(unknown)}"
            )
        return super().create(name, *args, **kwargs)
