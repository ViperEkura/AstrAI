import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Self

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from astrai.config.model_config import ConfigFactory
from astrai.config.train_config import TrainConfig
from astrai.dataset import RDSampler
from astrai.inference.scheduler import InferenceScheduler
from astrai.model.components.lora import inject_lora
from astrai.parallel.cp import CPState, CPStrategy, LossReduction
from astrai.parallel.executor import (
    BaseExecutor,
    ExecutorFactory,
    broadcast_state_dict,
    strip_compile_prefix,
)
from astrai.parallel.setup import get_current_device, get_rank, get_world_size
from astrai.parallel.topology import ParallelTopology, build_topology
from astrai.parallel.tp import TPState
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
    topology: Optional[ParallelTopology] = field(default=None)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    param_path: Optional[str] = field(default=None)

    _stop_event: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        # A topology is always answerable: contexts built outside the
        # builder (tests, tools) still get rank-layout math from their
        # world_size, with a trivial layout by definition.
        if self.topology is None:
            self.topology = ParallelTopology(self.world_size)

    @property
    def dp_size(self) -> int:
        """Data-parallel degree — the unit for data sharding and step math.

        Ranks sharing a dp_rank consume the same batch (cp peers hold
        different sequence slices of it), so sample and step counts derived
        from world size must use this instead.
        """
        return self.topology.dp_size

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def optimizer_step(self) -> int:
        return self.consumed_samples // (
            self.config.batch_per_device * self.dp_size * self.config.grad_accum_steps
        )


@dataclass
class _PreloadedState:
    model_config: dict = field(default_factory=dict)
    state_dict: Optional[dict] = None
    epoch: int = 0
    consumed_samples: int = 0
    checkpoint: Optional[Checkpoint] = None


def create_ref_model(
    model_fn: Callable[[], nn.Module],
    executor: Optional[BaseExecutor] = None,
    model: Optional[nn.Module] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[str] = None,
) -> Optional[nn.Module]:
    """Create a frozen reference model from executor or state dict.

    Training-domain helper for the DPO/GRPO reference and old policies: it
    builds the model, loads the broadcast state dict, and freezes it.  The
    distributed mechanics live in :func:`broadcast_state_dict`.

    In distributed mode (FSDP), ``unwrap_model`` returns ``None`` on
    non-rank-0.  The state_dict is broadcast from rank-0 to all ranks
    so every rank gets a complete copy.
    """
    if state_dict is None and executor is not None and model is not None:
        state_dict = executor.unwrap_model(model)

    # FSDP's unwrap_model returns None on non-rank-0. Broadcast from
    # rank-0 so every rank receives a complete state_dict.
    if executor is not None and executor.use_distributed:
        state_dict = broadcast_state_dict(state_dict)

    if state_dict is None:
        return None

    state_dict = strip_compile_prefix(state_dict)
    ref_model = model_fn()
    ref_model.load_state_dict(state_dict)
    ref_model.requires_grad_(False)
    ref_model.eval()
    if device is not None:
        ref_model = ref_model.to(device=device)
    return ref_model


class TrainContextBuilder:
    def __init__(
        self,
        config: TrainConfig,
    ):
        self.config = config
        self._param_path: Optional[str] = None
        self._resume: bool = False
        self._topology: Optional[ParallelTopology] = None

    def with_param_path(self, param_path: Optional[str], resume: bool = False) -> Self:
        self._param_path = param_path
        self._resume = resume
        return self

    def build(self) -> TrainContext:
        # Resolve the (dp, cp, tp) rank layout before anything consumes it.
        self._topology = build_topology(
            cp_size=self.config.cp_size,
            tp_size=self.config.tp_size,
            device_type=self.config.device_type,
        )
        if self._topology.tp_size > 1:
            self._validate_tp()

        # Resolve persisted state.
        preloaded_state = self._load_preloaded_state()
        if (
            self._topology.tp_size > 1
            and str(preloaded_state.model_config.get("ffn_type", "mlp")) == "moe"
        ):
            raise NotImplementedError(
                "tp_size > 1 does not support MoE yet: expert dispatch and "
                "the per-token aux loss assume full-feature router inputs"
            )

        # Build the core training components and restore their persisted state.
        executor = self._create_executor()
        self._validate_rollout_configuration(executor)
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
        kwargs = dict(cfg.executor_kwargs)
        if self._topology is not None and self._topology.tp_size > 1:
            # Gradient sync must span the dp dimension only: tp peers hold
            # different weight shards, and a world-group default would
            # all-reduce unrelated shards together.
            if cfg.dp_mode == "ddp":
                kwargs.setdefault("process_group", self._topology.dp_group)
            elif cfg.dp_mode == "fsdp":
                kwargs.setdefault("mesh", self._topology.mesh["dp"])
        return ExecutorFactory.create(
            cfg.dp_mode,
            grad_accum_steps=cfg.grad_accum_steps,
            **kwargs,
        )

    def _load_preloaded_state(self) -> _PreloadedState:
        cfg = self.config
        state = _PreloadedState(
            epoch=cfg.start_epoch,
            consumed_samples=cfg.start_samples * self._topology.dp_size,
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
                checkpoint.state_dict = strip_compile_prefix(checkpoint.state_dict)
                state.state_dict = checkpoint.state_dict
                state.model_config = checkpoint.config or state.model_config
                if self._resume:
                    state.epoch = checkpoint.epoch
                    per_step = (
                        cfg.batch_per_device
                        * self._topology.dp_size
                        * cfg.grad_accum_steps
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
            topology=self._topology,
            config=self.config,
            model_config=state.model_config,
            executor=executor,
            epoch=state.epoch,
            consumed_samples=state.consumed_samples,
            checkpoint=state.checkpoint,
            param_path=self._param_path,
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
                result = model.load_state_dict(state.state_dict, strict=False)
                if result.missing_keys or result.unexpected_keys:
                    logger.warning(
                        "preloaded state dict mismatch: %d missing, %d unexpected "
                        "(first missing: %s, first unexpected: %s)",
                        len(result.missing_keys),
                        len(result.unexpected_keys),
                        result.missing_keys[:3],
                        result.unexpected_keys[:3],
                    )
            if self._topology.tp_size > 1:
                # Shard after the full weights are in and before the
                # executor wraps/optimize: the shards are the parameters the
                # optimizer and FSDP/DDP see from here on.  The context exits
                # immediately by design — TPState.shard never restores.
                with TPState(self._topology).shard(model):
                    pass
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
        sampler_offset = context.consumed_samples // context.dp_size
        if self._resume and sampler_offset > 0:
            samples_per_replica = self._topology.samples_per_replica(len(train_dataset))
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
            # dp_group: None on trivial layouts, where the sampler's world-group
            # fallback is correct; a dp singleton keeps cp peers on identical batches.
            process_group=self._topology.dp_group,
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
        cp_state = None
        if self._topology is not None and self._topology.cp_size > 1:
            self._validate_cp(context)
            cp_state = CPState(self._topology)
        kwargs = dict(cfg.strategy_kwargs)
        kwargs.setdefault("moe_aux_loss_coef", cfg.moe_aux_loss_coef)
        if cfg.strategy in ("dpo", "grpo", "online_grpo", "online_dpo", "online_ppo"):
            kwargs["ref_model"] = create_ref_model(
                cfg.model_fn,
                executor=executor,
                model=context.model,
                device=get_current_device(),
            )
        if cfg.strategy == "grpo":
            kwargs["old_model"] = create_ref_model(
                cfg.model_fn,
                executor=executor,
                model=context.model,
                device=get_current_device(),
            )
        elif cfg.strategy == "online_grpo":
            kwargs["old_model"] = None
        if cfg.strategy == "online_ppo":
            critic, critic_optimizer = self._create_critic(context, executor)
            kwargs["critic"] = critic
            kwargs["critic_optimizer"] = critic_optimizer
            kwargs.setdefault("max_grad_norm", cfg.max_grad_norm)
        context.strategy = StrategyFactory.create(
            cfg.strategy,
            model=context.model,
            device=get_current_device(),
            executor=executor,
            **kwargs,
        )
        if cp_state is not None:
            # Decorate, don't branch: the strategy stays CP-oblivious and
            # implements the token-mean two-phase protocol; CPStrategy owns
            # the shard entry and the loss rescale.
            context.strategy = CPStrategy(context.strategy, cp_state)
        return kwargs

    def _validate_tp(self) -> None:
        """Guard the tensor-parallel support surface at build time."""
        cfg = self.config
        if cfg.cp_size > 1:
            raise NotImplementedError(
                "cp_size > 1 combined with tp_size > 1 is not verified yet: "
                "the ring-attention patch and head-sharded projections "
                "interact on the SDPA inputs"
            )
        if cfg.lora is not None:
            raise NotImplementedError(
                "tp_size > 1 does not support LoRA yet: the LoRALinear "
                "wrapper bypasses the sharded Linear forward"
            )

    def _validate_cp(self, context: TrainContext) -> None:
        """Guard the context-parallel support surface at build time."""
        cfg = self.config
        strategy_cls = StrategyFactory.get_component_class(cfg.strategy)
        if strategy_cls.loss_reduction is not LossReduction.TOKEN_MEAN:
            online = cfg.strategy.startswith("online_")
            reason = (
                "rollouts run on the single-GPU inference scheduler, which "
                "has no context-parallel path"
                if online
                else "the logprob shift crosses shard boundaries and the "
                "per-sequence reductions need cp-group all-reduces"
            )
            raise ValueError(
                f"cp_size > 1 supports token-mean strategies (seq, sft) only; "
                f"'{cfg.strategy}' additionally needs: {reason}"
            )
        if cfg.dp_mode not in ("ddp", "fsdp"):
            raise ValueError(
                "cp_size > 1 requires --dp_mode ddp or fsdp: unsynced "
                "cp peers hold gradients of different sequence slices"
            )
        if str(context.model_config.get("ffn_type", "mlp")) == "moe":
            raise NotImplementedError(
                "cp_size > 1 does not support MoE yet: the load-balance aux "
                "loss would be computed per sequence slice instead of per token"
            )

    def _create_critic(
        self, context: TrainContext, executor: BaseExecutor
    ) -> tuple[nn.Module, OptimizerProtocol]:
        """Build the PPO critic and its optimizer, restoring persisted state.

        The critic backbone warm-starts from the policy weights (standard
        actor-critic initialization; the fresh value head is the only
        randomly-initialized part).  On resume, ``value_model`` /
        ``value_optimizer`` checkpoint extras override the warm start —
        and their absence is fatal rather than a silent fresh critic.
        """
        cfg = self.config
        device = get_current_device()
        checkpoint = context.checkpoint
        if checkpoint is not None:
            missing = [
                name
                for name in ("value_model", "value_optimizer")
                if name not in checkpoint.extra
            ]
            if missing:
                raise ValueError(
                    "online_ppo resume requires critic state in the "
                    f"checkpoint; missing extras: {', '.join(missing)}"
                )

        state_dict = executor.unwrap_model(context.model)
        if executor.use_distributed:
            state_dict = broadcast_state_dict(state_dict)
        critic = cfg.critic_model_fn()
        if state_dict is not None:
            state_dict = strip_compile_prefix(state_dict)
            result = critic.load_state_dict(state_dict, strict=False)
            if result.unexpected_keys:
                raise ValueError(
                    "critic model received unexpected keys from the policy "
                    f"state dict: {result.unexpected_keys[:3]}"
                )
            unexpected_missing = [
                key for key in result.missing_keys if not key.startswith("value_head.")
            ]
            if unexpected_missing:
                raise ValueError(
                    "critic backbone is missing policy parameters: "
                    f"{unexpected_missing[:3]}"
                )
        if checkpoint is not None:
            critic.load_state_dict(checkpoint.extra["value_model"])
        critic = critic.to(device)
        critic.train()

        optimizer_factory = cfg.critic_optimizer_fn or cfg.optimizer_fn
        critic_optimizer = optimizer_factory(critic)
        if checkpoint is not None:
            critic_optimizer.load_state_dict(checkpoint.extra["value_optimizer"])
        return critic, critic_optimizer

    def _configure_rollout(self, context: TrainContext, strategy_kwargs: dict) -> None:
        cfg = self.config
        if not cfg.strategy.startswith("online_"):
            return
        if not context.strategy.supports_online():
            raise ValueError(
                f"Strategy '{cfg.strategy}' does not support online rollout"
            )
        self._validate_rollout_configuration(context.executor)
        inference_model = context.executor.model_for_inference(context.model)
        tokenizer = AutoTokenizer.from_pretrained(self._param_path)
        group_size = strategy_kwargs.get("group_size", 1)
        scheduler = InferenceScheduler(
            model=inference_model,
            tokenizer=tokenizer,
            max_batch_size=group_size * max(1, cfg.batch_per_device),
            max_seq_len=getattr(
                inference_model.config, "max_position_embeddings", None
            ),
            policy_version=(
                context.checkpoint.meta.get("policy_version", context.optimizer_step)
                if context.checkpoint is not None
                else context.optimizer_step
            ),
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
                max_policy_lag=cfg.rollout_max_policy_lag,
            )
        )

    def _validate_rollout_configuration(self, executor: BaseExecutor) -> None:
        cfg = self.config
        if not cfg.strategy.startswith("online_"):
            return
        if cfg.compile_mode is not None:
            raise ValueError(
                "Online rollout does not support torch.compile; set compile_mode=None"
            )
        capabilities = executor.rollout_capabilities()
        if not capabilities.supports_in_process:
            raise ValueError(capabilities.reason or "Online rollout is unsupported")
