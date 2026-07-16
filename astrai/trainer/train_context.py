from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Self

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from astrai.config.train_config import TrainConfig
from astrai.dataset import ResumableDistributedSampler
from astrai.model.components.lora import inject_lora
from astrai.parallel.executor import BaseExecutor, ExecutorFactory
from astrai.parallel.setup import get_current_device, get_rank, get_world_size
from astrai.protocols import OptimizerProtocol, SchedulerProtocol
from astrai.serialization import Checkpoint, load_json
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

        model = cfg.model_fn()
        model = model.to(device=device)

        model_config = {}
        if self._param_path:
            config_path = Path(self._param_path) / "config.json"
            if config_path.exists():
                model_config = load_json(config_path)

        if not model_config and hasattr(model, "config"):
            model_config = model.config.to_dict()

        context = TrainContext(
            model=model,
            world_size=get_world_size(),
            rank=get_rank(),
            config=cfg,
            model_config=model_config,
            executor=executor,
        )

        if self._param_path:
            checkpoint = Checkpoint.load_any(self._param_path)
            if checkpoint is not None:
                model.load_state_dict(checkpoint.state_dict, strict=False)
                if checkpoint.config:
                    context.model_config = checkpoint.config

                if self._resume:
                    context.epoch = checkpoint.epoch or cfg.start_epoch
                    if checkpoint.consumed_samples > 0:
                        per_step = (
                            cfg.batch_per_device
                            * context.world_size
                            * cfg.grad_accum_steps
                        )
                        context.consumed_samples = (
                            checkpoint.consumed_samples // per_step
                        ) * per_step
                    else:
                        context.consumed_samples = (
                            cfg.start_samples * context.world_size
                        )
                    context.checkpoint = checkpoint

        if cfg.lora is not None:
            inject_lora(
                model,
                r=cfg.lora.r,
                alpha=cfg.lora.alpha,
                target_modules=set(cfg.lora.target_modules),
            )

        context.optimizer = cfg.optimizer_fn(model)
        context.scheduler = cfg.scheduler_fn(context.optimizer)

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
        sampler = ResumableDistributedSampler(
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
        )

        if val_dataset is not None:
            val_sampler = ResumableDistributedSampler(
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
            )

        context.model, context.optimizer, context.dataloader, context.scheduler = (
            executor.prepare(
                model,
                context.optimizer,
                context.dataloader,
                context.scheduler,
            )
        )

        if context.checkpoint and context.checkpoint.extra:
            extra = context.checkpoint.extra
            for name in ("optimizer", "scheduler"):
                if name in extra:
                    obj = getattr(context, name, None)
                    if obj is not None:
                        obj.load_state_dict(extra[name])

        strategy_kwargs = dict(cfg.extra_kwargs)

        if cfg.strategy in ("dpo", "grpo"):
            ref_model = create_ref_model(
                cfg.model_fn, executor.unwrap_model(context.model)
            ).to(device=device)
            strategy_kwargs["ref_model"] = ref_model

        if cfg.strategy == "grpo":
            old_model = create_ref_model(
                cfg.model_fn, executor.unwrap_model(context.model)
            ).to(device=device)
            strategy_kwargs["old_model"] = old_model

        context.strategy = StrategyFactory.create(
            cfg.strategy,
            model=context.model,
            device=device,
            executor=executor,
            model_fn=cfg.model_fn,
            **strategy_kwargs,
        )

        return context
