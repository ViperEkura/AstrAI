"""Assemble online rollout backends for an existing training context.

The builder supplies the reference-model factory and capability validator so
this module owns rollout wiring without owning model restoration or topology.
"""

from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Optional

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

if TYPE_CHECKING:
    from astrai.trainer.train_context import TrainContext


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
    group_size = strategy_kwargs.get("group_size", 1)
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
                enable_overlap=getattr(cfg, "rollout_enable_overlap", False),
            )
        )

    def _replica(device: str, max_batch_size: int) -> ReplicaBackend:
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
            enable_overlap=getattr(cfg, "rollout_enable_overlap", False),
        )

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
            seed=getattr(cfg, "rollout_seed", None),
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
