"""Registry and capability declarations for training strategies."""

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from astrai.factory import BaseFactory
from astrai.trainer.strategy.base import BaseStrategy


@dataclass(frozen=True)
class StrategyCapabilities:
    online: bool = False
    reference_model: bool = False
    old_model: bool = False
    initialize_old_model: bool = False
    critic: bool = False
    async_round: bool = False
    min_group_size: int = 1


class StrategyFactory(BaseFactory[BaseStrategy]):
    _capabilities: Dict[str, StrategyCapabilities] = {}

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
    def create_checked(cls, name: str, *args, **kwargs) -> BaseStrategy:
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
        unknown = sorted(set(kwargs) - allowed)
        if unknown:
            raise TypeError(
                f"{name} strategy got unknown arguments: {', '.join(unknown)}"
            )
        return super().create_checked(name, *args, **kwargs)
