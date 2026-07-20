from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Self

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from astrai.config.train_config import TrainConfig
from astrai.dataset import RDSampler
from astrai.inference.core.scheduler import InferenceScheduler
from astrai.model.components.lora import inject_lora
from astrai.parallel.executor import BaseExecutor, ExecutorFactory
from astrai.parallel.setup import get_current_device, get_rank, get_world_size
from astrai.protocols import OptimizerProtocol, SchedulerProtocol
from astrai.serialization import Checkpoint, load_json
from astrai.tokenize import AutoTokenizer
from astrai.trainer.rollout import RolloutGenerator, RolloutRunner
from astrai.trainer.strategy import BaseStrategy, StrategyFactory, create_ref_model


@dataclass
class TrainContext:
    model: nn.Module = field(default=None)
    strategy: BaseStrategy = field(default=None)
    dataloader: DataLoader = field(default=None)
    optimizer: OptimizerProtocol = field(default=None)
    scheduler: SchedulerProtocol = field(default=None)
    checkpoint: Checkpoint = field(default=None)
    config: TrainConfig = field(default=None)
    model_config: dict = field(default_factory=dict)
    executor: BaseExecutor = field(default=None)
    epoch: int = field(default=0)
    consumed_samples: int = field(default=0)
    loss: float = field(default=0.0)
    grad_norm: Optional[float] = field(default=None)
    val_dataloader: Optional[DataLoader] = field(default=None)
    val_loss: Optional[float] = field(default=None)

    world_size: int = field(default=1)
    rank: int = field(default=0)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def optimizer_step(self) -> int:
        return self.consumed_samples // (
            self.config.batch_per_device
            * self.world_size
            * self.config.grad_accum_steps
        )


class TrainContextBuilder:
    def __init__(
        self,
        config: TrainConfig,
    ):
        self.config = config
        self._param_path: Optional[str] = None
        self._resume: bool = False

    def with_param_path(self, param_path: Optional[str], resume: bool = False) -> Self:
        self._param_path = param_path
        self._resume = resume
        return self

    def build(self) -> TrainContext:
        cfg = self.config
        device = get_current_device()

        executor = ExecutorFactory.create(
            cfg.parallel_mode,
            grad_accum_steps=cfg.grad_accum_steps,
            **cfg.executor_kwargs,
        )

        model_config = {}
        if self._param_path:
            config_path = Path(self._param_path) / "config.json"
            if config_path.exists():
                model_config = load_json(config_path)

        preloaded_state_dict = None
        preloaded_epoch = cfg.start_epoch
        preloaded_consumed = cfg.start_samples * get_world_size()
        preloaded_checkpoint = None
        if self._param_path:
            checkpoint = Checkpoint.load_any(self._param_path)
            if checkpoint is not None:
                preloaded_state_dict = checkpoint.state_dict
                if checkpoint.config:
                    model_config = checkpoint.config
                if self._resume:
                    preloaded_epoch = checkpoint.epoch or cfg.start_epoch
                    if checkpoint.consumed_samples > 0:
                        per_step = (
                            cfg.batch_per_device
                            * get_world_size()
                            * cfg.grad_accum_steps
                        )
                        preloaded_consumed = (
                            checkpoint.consumed_samples // per_step
                        ) * per_step
                    else:
                        preloaded_consumed = cfg.start_samples * get_world_size()
                    preloaded_checkpoint = checkpoint

        if not model_config and hasattr(cfg.model_fn(), "config"):
            model_config = cfg.model_fn().config.to_dict()

        def _before_wrap(m):
            m = m.to(device=device)
            if preloaded_state_dict is not None:
                m.load_state_dict(preloaded_state_dict, strict=False)
            if cfg.lora is not None:
                inject_lora(
                    m,
                    r=cfg.lora.r,
                    alpha=cfg.lora.alpha,
                    target_modules=set(cfg.lora.target_modules),
                )
            return m

        context = TrainContext(
            world_size=get_world_size(),
            rank=get_rank(),
            config=cfg,
            model_config=model_config,
            executor=executor,
            epoch=preloaded_epoch,
            consumed_samples=preloaded_consumed,
            checkpoint=preloaded_checkpoint,
        )

        context.model, context.optimizer, context.scheduler = executor.prepare(
            cfg.model_fn,
            cfg.optimizer_fn,
            cfg.scheduler_fn,
            before_wrap=_before_wrap,
        )

        train_dataset = cfg.dataset
        val_dataset = cfg.val_dataset

        if val_dataset is None and cfg.val_split is not None:
            n_total = len(cfg.dataset)
            n_val = max(1, int(n_total * cfg.val_split))
            n_train = n_total - n_val
            generator = torch.Generator().manual_seed(cfg.random_seed)
            train_dataset, val_dataset = random_split(
                cfg.dataset, [n_train, n_val], generator=generator
            )

        sampler_offset = context.consumed_samples // context.world_size
        sampler = RDSampler(
            data_source=train_dataset,
            start_epoch=context.epoch,
            start_iter=sampler_offset,
            seed=cfg.random_seed,
        )
        context.dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_per_device,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            collate_fn=cfg.collate_fn,
        )

        if val_dataset is not None:
            val_sampler = RDSampler(
                data_source=val_dataset,
                start_epoch=0,
                start_iter=0,
                seed=cfg.random_seed,
                shuffle=False,
            )
            context.val_dataloader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_per_device,
                sampler=val_sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                prefetch_factor=cfg.prefetch_factor,
                collate_fn=cfg.collate_fn,
            )

        if context.checkpoint and context.checkpoint.extra:
            extra = context.checkpoint.extra
            for name in ("optimizer", "scheduler"):
                if name in extra:
                    obj = getattr(context, name, None)
                    if obj is not None:
                        obj.load_state_dict(extra[name])

        strategy_kwargs = dict(cfg.extra_kwargs)

        needs_ref = cfg.strategy in (
            "dpo",
            "grpo",
            "online_grpo",
            "online_dpo",
        )
        needs_old = cfg.strategy in ("grpo", "online_grpo")

        if needs_ref:
            ref_model = create_ref_model(
                cfg.model_fn, executor.unwrap_model(context.model)
            ).to(device=device)
            strategy_kwargs["ref_model"] = ref_model

        old_model = None
        if needs_old:
            old_model = create_ref_model(
                cfg.model_fn, executor.unwrap_model(context.model)
            ).to(device=device)
            strategy_kwargs["old_model"] = old_model

        context.strategy = StrategyFactory.create(
            cfg.strategy,
            model=context.model,
            device=device,
            executor=executor,
            **strategy_kwargs,
        )

        # Enable online rollout when the train_type is an ``online_*`` variant.
        is_online = cfg.strategy.startswith("online_")
        if is_online:
            if not context.strategy.supports_online():
                raise ValueError(
                    f"Strategy '{cfg.strategy}' does not support online rollout"
                )
            if cfg.reward_model_fn is None:
                raise ValueError("reward_model_fn is required for online RL strategies")

            tokenizer = AutoTokenizer.from_pretrained(self._param_path)
            reward_model = cfg.reward_model_fn()

            group_size = strategy_kwargs.get("group_size", 1)
            rollout_batch_size = group_size * max(1, cfg.batch_per_device)
            max_seq_len = getattr(context.model.config, "max_len", None)

            scheduler = InferenceScheduler(
                model=context.model,
                tokenizer=tokenizer,
                max_batch_size=rollout_batch_size,
                max_seq_len=max_seq_len,
                max_prompt_len=max_seq_len or 4096,
            )

            generator = RolloutGenerator(
                scheduler=scheduler,
                tokenizer=tokenizer,
                max_tokens=cfg.rollout_max_tokens,
                group_size=group_size,
                temperature=cfg.rollout_temperature,
                top_k=cfg.rollout_top_k,
                top_p=cfg.rollout_top_p,
            )
            runner = RolloutRunner(
                generator=generator,
                reward_model=reward_model,
                rollout_interval=cfg.rollout_interval,
            )
            context.strategy.set_rollout_runner(runner)

        return context
