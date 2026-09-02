import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Self

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from astrai.config.model_config import ConfigFactory
from astrai.config.train_config import TrainConfig
from astrai.dataset import RDSampler
from astrai.inference.scheduler import InferenceScheduler
from astrai.model.components.lora import inject_lora
from astrai.parallel.executor import BaseExecutor, ExecutorFactory, create_ref_model
from astrai.parallel.setup import get_current_device, get_rank, get_world_size
from astrai.protocols import OptimizerProtocol, SchedulerProtocol
from astrai.serialization import (
    Checkpoint,
    adapt_config,
    convert_hf_weights,
    load_json,
    looks_like_hf_state_dict,
)
from astrai.tokenize import AutoTokenizer
from astrai.trainer.metric_util import GradSNRTracker
from astrai.trainer.rollout import RolloutGenerator, RolloutRunner
from astrai.trainer.strategy import BaseStrategy, StrategyFactory

logger = logging.getLogger(__name__)


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
    metrics: Dict[str, float] = field(default_factory=dict)
    grad_norm: Optional[float] = field(default=None)
    grad_snr_tracker: GradSNRTracker = field(default_factory=GradSNRTracker)
    val_dataloader: Optional[DataLoader] = field(default=None)
    val_loss: Optional[float] = field(default=None)

    world_size: int = field(default=1)
    rank: int = field(default=0)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    _stop_event: threading.Event = field(default_factory=threading.Event)

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def optimizer_step(self) -> int:
        return self.consumed_samples // (
            self.config.batch_per_device
            * self.world_size
            * self.config.grad_accum_steps
        )


@dataclass
class _PreloadedState:
    model_config: dict = field(default_factory=dict)
    state_dict: Optional[dict] = None
    epoch: int = 0
    consumed_samples: int = 0
    checkpoint: Optional[Checkpoint] = None


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
        # Resolve persisted state.
        preloaded_state = self._load_preloaded_state()

        # Build the core training components and restore their persisted state.
        executor = self._create_executor()
        context = self._create_context(preloaded_state, executor)
        self._prepare_model(context, executor, preloaded_state)
        self._restore_optimizer_state(context)

        # Resolve datasets.
        train_dataset, val_dataset = self._get_datasets()
        self._create_dataloaders(context, train_dataset, val_dataset)

        # Strategies depend on the prepared model; online rollout depends on both.
        strategy_kwargs = self._create_strategy(context, executor)
        self._configure_rollout(context, strategy_kwargs)

        return context

    def _create_executor(self) -> BaseExecutor:
        cfg = self.config
        return ExecutorFactory.create(
            cfg.parallel_mode,
            grad_accum_steps=cfg.grad_accum_steps,
            **cfg.executor_kwargs,
        )

    def _load_preloaded_state(self) -> _PreloadedState:
        cfg = self.config
        state = _PreloadedState(
            epoch=cfg.start_epoch,
            consumed_samples=cfg.start_samples * get_world_size(),
        )
        if self._param_path:
            config_path = Path(self._param_path) / "config.json"
            if config_path.exists():
                state.model_config = adapt_config(load_json(config_path))
            checkpoint = Checkpoint.load_any(self._param_path)
            if checkpoint is not None:
                if checkpoint.config:
                    checkpoint.config = adapt_config(checkpoint.config)
                if checkpoint.state_dict and looks_like_hf_state_dict(
                    checkpoint.state_dict
                ):
                    checkpoint.state_dict = convert_hf_weights(
                        checkpoint.state_dict,
                        ConfigFactory.load(checkpoint.config or state.model_config),
                    )
                state.state_dict = checkpoint.state_dict
                state.model_config = checkpoint.config or state.model_config
                if self._resume:
                    state.epoch = checkpoint.epoch
                    per_step = (
                        cfg.batch_per_device * get_world_size() * cfg.grad_accum_steps
                    )
                    state.consumed_samples = (
                        checkpoint.consumed_samples // per_step * per_step
                    )
                    state.checkpoint = checkpoint
        if not state.model_config:
            model = cfg.model_fn()
            if hasattr(model, "config"):
                state.model_config = model.config.to_dict()
        return state

    def _create_context(
        self, state: _PreloadedState, executor: BaseExecutor
    ) -> TrainContext:
        return TrainContext(
            world_size=get_world_size(),
            rank=get_rank(),
            config=self.config,
            model_config=state.model_config,
            executor=executor,
            epoch=state.epoch,
            consumed_samples=state.consumed_samples,
            checkpoint=state.checkpoint,
        )

    def _prepare_model(
        self, context: TrainContext, executor: BaseExecutor, state: _PreloadedState
    ) -> None:
        cfg = self.config
        device = get_current_device()

        def before_wrap(model):
            model = model.to(device=device)
            if cfg.lora is not None:
                inject_lora(
                    model,
                    r=cfg.lora.r,
                    alpha=cfg.lora.alpha,
                    target_modules=set(cfg.lora.target_modules),
                )
            if state.state_dict is not None:
                model.load_state_dict(state.state_dict, strict=False)
            return model

        def after_wrap(model):
            if cfg.compile_mode is not None:
                logger.info("torch.compile enabled (mode=%s)", cfg.compile_mode)
                model = torch.compile(model, mode=cfg.compile_mode)
            return model

        context.model, context.optimizer, context.scheduler = executor.prepare(
            cfg.model_fn,
            cfg.optimizer_fn,
            cfg.scheduler_fn,
            before_wrap=before_wrap,
            after_wrap=after_wrap,
        )

    def _get_datasets(self):
        cfg = self.config
        if cfg.val_dataset is not None or cfg.val_split is None:
            return cfg.dataset, cfg.val_dataset
        n_val = max(1, int(len(cfg.dataset) * cfg.val_split))
        generator = torch.Generator().manual_seed(cfg.random_seed)
        return random_split(
            cfg.dataset, [len(cfg.dataset) - n_val, n_val], generator=generator
        )

    def _create_dataloaders(
        self, context: TrainContext, train_dataset, val_dataset
    ) -> None:
        sampler_offset = context.consumed_samples // context.world_size
        if self._resume and sampler_offset > 0:
            samples_per_replica = (
                len(train_dataset) + context.world_size - 1
            ) // context.world_size
            if samples_per_replica > 0:
                context.epoch = sampler_offset // samples_per_replica
        context.dataloader = self._create_dataloader(
            train_dataset, context.epoch, sampler_offset
        )
        if val_dataset is not None:
            context.val_dataloader = self._create_dataloader(
                val_dataset, 0, 0, shuffle=False
            )

    def _create_dataloader(
        self, dataset, epoch: int, start_iter: int, shuffle: bool = True
    ):
        cfg = self.config
        sampler = RDSampler(
            dataset,
            start_epoch=epoch,
            start_iter=start_iter,
            seed=cfg.random_seed,
            shuffle=shuffle,
        )
        loader_kwargs = dict(
            dataset=dataset,
            batch_size=cfg.batch_per_device,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            collate_fn=cfg.collate_fn,
        )
        # PyTorch rejects prefetch_factor/persistent_workers when workers=0.
        if cfg.num_workers > 0:
            loader_kwargs["persistent_workers"] = cfg.persistent_workers
            if cfg.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        return DataLoader(
            **loader_kwargs,
        )

    def _restore_optimizer_state(self, context: TrainContext) -> None:
        if context.checkpoint and context.checkpoint.extra:
            for name in ("optimizer", "scheduler"):
                if (
                    name in context.checkpoint.extra
                    and getattr(context, name, None) is not None
                ):
                    getattr(context, name).load_state_dict(
                        context.checkpoint.extra[name]
                    )

    def _create_strategy(self, context: TrainContext, executor: BaseExecutor) -> dict:
        cfg = self.config
        kwargs = dict(cfg.strategy_kwargs)
        if (
            cfg.strategy == "online_grpo"
            and kwargs.get("loss_variant") == "dr_grpo"
            and kwargs.get("max_completion_length") is None
        ):
            kwargs["max_completion_length"] = cfg.rollout_max_tokens
        kwargs.setdefault("moe_aux_loss_coef", cfg.moe_aux_loss_coef)
        if cfg.strategy in ("dpo", "grpo", "online_grpo", "online_dpo"):
            kwargs["ref_model"] = create_ref_model(
                cfg.model_fn,
                executor=executor,
                model=context.model,
                device=get_current_device(),
            )
        if cfg.strategy in ("grpo", "online_grpo"):
            kwargs["old_model"] = create_ref_model(
                cfg.model_fn,
                executor=executor,
                model=context.model,
                device=get_current_device(),
            )
        context.strategy = StrategyFactory.create(
            cfg.strategy,
            model=context.model,
            device=get_current_device(),
            executor=executor,
            **kwargs,
        )
        return kwargs

    def _configure_rollout(self, context: TrainContext, strategy_kwargs: dict) -> None:
        cfg = self.config
        if not cfg.strategy.startswith("online_"):
            return
        if not context.strategy.supports_online():
            raise ValueError(
                f"Strategy '{cfg.strategy}' does not support online rollout"
            )
        tokenizer = AutoTokenizer.from_pretrained(self._param_path)
        group_size = strategy_kwargs.get("group_size", 1)
        scheduler = InferenceScheduler(
            model=context.model,
            tokenizer=tokenizer,
            max_batch_size=group_size * max(1, cfg.batch_per_device),
            max_seq_len=getattr(context.model.config, "max_position_embeddings", None),
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
        context.strategy.set_rollout_runner(
            RolloutRunner(
                generator=generator,
                reward_model=cfg.reward_model_fn(),
                rollout_interval=cfg.rollout_interval,
            )
        )
