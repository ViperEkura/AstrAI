"""Assemble online rollout backends for an existing training context.

The builder supplies the reference-model factory and capability validator so
this module owns rollout wiring without owning model restoration or topology.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch

from astrai.config.train_config import TrainConfig
from astrai.parallel.executor import BaseExecutor
from astrai.trainer.backend import ColocatedBackend, P2PCopyPublisher, ReplicaBackend
from astrai.trainer.rollout import (
    RolloutEvaluator,
    RolloutGenerator,
    RolloutRunner,
    SamplingParams,
)
from astrai.trainer.rollout.async_round import AsyncRoundCoordinator

if TYPE_CHECKING:
    from astrai.trainer.train_context import TrainContext


@dataclass(frozen=True)
class ResolvedAsyncRolloutConfig:
    learner_device: str
    devices: Tuple[str, ...]
    params: SamplingParams
    max_seq_len: Optional[int]
    max_prompts_per_worker: int
    startup_timeout_s: float
    generation_timeout_s: float
    weight_timeout_s: float
    max_policy_lag: int
    random_seed: int
    sample_cursor: int

    @property
    def max_batch_size(self) -> int:
        return self.params.group_size * self.max_prompts_per_worker


def resolve_async_rollout(
    config: TrainConfig,
    learner_device: torch.device,
    params: SamplingParams,
    max_seq_len: Optional[int],
    sample_cursor: int,
) -> ResolvedAsyncRolloutConfig:
    devices = tuple(torch.device(name) for name in config.rollout_devices)
    if not devices or any(d.type != "cuda" or d.index is None for d in devices):
        raise ValueError("async_round requires indexed CUDA rollout devices")
    if len(set(devices)) != len(devices):
        raise ValueError("async_round requires distinct rollout_devices")
    available = torch.cuda.device_count()
    if any(d.index >= available for d in devices):
        raise ValueError(f"rollout_devices exceed available CUDA devices ({available})")
    if learner_device.type != "cuda" or learner_device.index is None:
        raise ValueError("async_round learner must use an indexed CUDA device")
    if learner_device in devices:
        raise ValueError("rollout_devices must exclude the learner device")
    if params.group_size < 2:
        raise ValueError("online_grpo group_size must be >= 2")
    return ResolvedAsyncRolloutConfig(
        learner_device=str(learner_device),
        devices=tuple(str(d) for d in devices),
        params=params,
        max_seq_len=max_seq_len,
        max_prompts_per_worker=max(
            1, (config.batch_per_device + len(devices) - 1) // len(devices)
        ),
        startup_timeout_s=config.rollout_startup_timeout_s,
        generation_timeout_s=config.rollout_worker_timeout_s,
        weight_timeout_s=config.rollout_weight_timeout_s,
        max_policy_lag=config.rollout_max_policy_lag,
        random_seed=config.random_seed,
        sample_cursor=sample_cursor,
    )


def configure_rollout(
    context: "TrainContext",
    config: TrainConfig,
    param_path: Optional[str],
    strategy_kwargs: dict,
    create_ref_model: Callable,
    validate: Callable[[BaseExecutor], None],
    scheduler_cls: type,
    tokenizer_cls: type,
) -> None:
    cfg = config
    if not cfg.strategy.startswith("online_"):
        return
    if not context.strategy.supports_online():
        raise ValueError(f"Strategy '{cfg.strategy}' does not support online rollout")
    validate(context.executor)
    inference_model = context.executor.model_for_inference(context.model)
    tokenizer = tokenizer_cls.from_pretrained(param_path)
    group_size = getattr(
        context.strategy, "group_size", strategy_kwargs.get("group_size", 1)
    )
    policy_version = (
        context.checkpoint.meta.get("policy_version", context.optimizer_step)
        if context.checkpoint is not None
        else context.optimizer_step
    )
    max_seq_len = getattr(inference_model.config, "max_position_embeddings", None)
    if cfg.rollout_pool_seq_len is not None:
        # Right-size the KV pool: the default is the model's full
        # context window, but a rollout never needs more than prompt +
        # rollout_max_tokens — the difference is GBs of idle pool
        # (see TrainConfig.rollout_pool_seq_len for the formula).
        max_seq_len = (
            min(max_seq_len, cfg.rollout_pool_seq_len)
            if max_seq_len is not None
            else cfg.rollout_pool_seq_len
        )
    train_device = next(context.model.parameters()).device

    def _resolve_device(name: str, value: str | None) -> str | None:
        if value is None:
            return None
        if value.startswith("cuda"):
            count = torch.cuda.device_count()
            if count == 0:
                raise ValueError(f"{name}={value!r} but no CUDA device is available")
            if ":" in value and int(value.split(":", 1)[1]) >= count:
                raise ValueError(
                    f"{name}={value!r} exceeds available CUDA devices ({count})"
                )
        return value

    rollout_device = _resolve_device("rollout_device", cfg.rollout_device)
    val_device = _resolve_device("rollout_val_device", cfg.rollout_val_device)

    def _colocated(max_batch_size: int) -> ColocatedBackend:
        return ColocatedBackend(
            scheduler_cls(
                model=inference_model,
                tokenizer=tokenizer,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                policy_version=policy_version,
            )
        )

    def _replica(
        device: str, max_batch_size: int, enable_cuda_graph: bool = True
    ) -> ReplicaBackend:
        model = create_ref_model(
            model_fn=cfg.model_fn,
            executor=context.executor,
            model=context.model,
            device=device,
        )
        if model is None:
            raise RuntimeError(f"cannot build rollout replica on {device!r}")
        # Match the training dtype so the replica's sampling space
        # agrees with the training-side logprob recomputation.
        model.to(dtype=next(context.model.parameters()).dtype)
        return ReplicaBackend(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            policy_version=policy_version,
            enable_cuda_graph=enable_cuda_graph,
        )

    if getattr(cfg, "rollout_mode", "sync") == "async_round":
        if str(getattr(inference_model.config, "ffn_type", "mlp")) == "moe":
            raise ValueError(
                "async_round currently supports dense models only: MoE auxiliary "
                "loss is not equivalent across learner microbatches"
            )
        params = SamplingParams(
            max_tokens=cfg.rollout_max_tokens,
            group_size=group_size,
            temperature=cfg.rollout_temperature,
            top_k=cfg.rollout_top_k,
            top_p=cfg.rollout_top_p,
        )
        resolved = resolve_async_rollout(
            cfg, train_device, params, max_seq_len, context.consumed_samples
        )
        context.async_rollout = AsyncRoundCoordinator(
            source=inference_model,
            model_fn=cfg.model_fn,
            param_path=param_path,
            devices=list(resolved.devices),
            params=resolved.params,
            reward_model=cfg.reward_model_fn(),
            policy_version=policy_version,
            max_batch_size=resolved.max_batch_size,
            max_seq_len=resolved.max_seq_len,
            model_dtype=next(context.model.parameters()).dtype,
            max_policy_lag=resolved.max_policy_lag,
            max_prompts_per_worker=resolved.max_prompts_per_worker,
            worker_timeout_s=resolved.generation_timeout_s,
            startup_timeout_s=resolved.startup_timeout_s,
            weight_timeout_s=resolved.weight_timeout_s,
            random_seed=resolved.random_seed,
            sample_cursor=resolved.sample_cursor,
        )
        context.optimizer_steps_completed = (
            context.checkpoint.meta.get("optimizer_step", context.optimizer_step)
            if context.checkpoint is not None
            else context.optimizer_step
        )
        context.strategy.set_rollout_runner(context.async_rollout)
        return

    batch_capacity = group_size * max(1, cfg.batch_per_device)
    publishers: list = []
    if rollout_device is None:
        train_backend = _colocated(batch_capacity)
    else:
        train_backend = _replica(rollout_device, batch_capacity)
        publishers.append(P2PCopyPublisher(train_backend))

    generator = RolloutGenerator(
        backend=train_backend,
        tokenizer=tokenizer,
        params=SamplingParams(
            max_tokens=cfg.rollout_max_tokens,
            group_size=group_size,
            temperature=cfg.rollout_temperature,
            top_k=cfg.rollout_top_k,
            top_p=cfg.rollout_top_p,
        ),
        output_device=train_device,
    )
    reward_model = cfg.reward_model_fn()
    context.strategy.set_rollout_runner(
        RolloutRunner(
            generator=generator,
            reward_model=reward_model,
            rollout_interval=cfg.rollout_interval,
            max_policy_lag=cfg.rollout_max_policy_lag,
        )
    )
    # Validation rolls out under its own sampling params (e.g. greedy
    # decode, val-specific group size), inheriting every unset field
    # from the training rollout; with rollout_val_device set it runs
    # on a dedicated replica instead of the training backend.
    val_params = replace(generator.params, **cfg.rollout_val_overrides())
    if val_device is None:
        val_generator = generator
    else:
        val_backend = _replica(
            val_device, val_params.group_size * max(1, cfg.batch_per_device)
        )
        publishers.append(P2PCopyPublisher(val_backend))
        val_generator = RolloutGenerator(
            backend=val_backend,
            tokenizer=tokenizer,
            params=val_params,
            output_device=train_device,
        )
    context.val_evaluator = RolloutEvaluator(
        generator=val_generator,
        reward_model=reward_model,
        params=val_params,
    )
    if publishers:
        context.strategy.set_weight_publishers(publishers)
