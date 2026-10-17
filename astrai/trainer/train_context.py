import logging
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Self, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from astrai.config.model_config import ConfigFactory
from astrai.config.train_config import TrainConfig, _TrainRuntimeDependencies
from astrai.inference.core.engine_core import EngineCore
from astrai.model.components.lora import inject_lora
from astrai.optimization.protocols import OptimizerProtocol, SchedulerProtocol
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
from astrai.serialization import (
    Checkpoint,
    adapt_config,
    convert_hf_weights,
    load_json,
    looks_like_hf_state_dict,
)
from astrai.tokenize import AutoTokenizer
from astrai.trainer.callbacks.metric_util import GradSNRTracker
from astrai.trainer.components import (
    build_dataloader,
    build_dataset_component,
    build_model_component,
    build_optimizer_component,
    build_scheduler_component,
    create_ref_model,
    load_optimizer_state_dict,
)
from astrai.trainer.optional_extras import (
    component_extra_keys,
    load_component_extra,
    require_component_extras,
    restore_checkpoint_extras,
)
from astrai.trainer.rollout import RolloutEvaluator, RolloutRunner
from astrai.trainer.rollout.setup import configure_rollout
from astrai.trainer.strategy import BaseStrategy, StrategyFactory

logger = logging.getLogger(__name__)


@dataclass
class TrainContext:
    model: nn.Module = field(default=None)
    strategy: BaseStrategy = field(default=None)
    dataset: Optional[Dataset] = field(default=None)
    val_dataset: Optional[Dataset] = field(default=None)
    dataloader: DataLoader = field(default=None)
    optimizer: OptimizerProtocol = field(default=None)
    scheduler: SchedulerProtocol = field(default=None)
    checkpoint: Checkpoint = field(default=None)
    config: TrainConfig = field(default=None)
    model_config: Dict = field(default_factory=dict)
    executor: BaseExecutor = field(default=None)
    epoch: int = field(default=0)
    consumed_samples: int = field(default=0)
    loss: float = field(default=0.0)
    metrics: Dict[str, float] = field(default_factory=dict)
    grad_norm: Optional[float] = field(default=None)
    grad_snr_tracker: GradSNRTracker = field(default_factory=GradSNRTracker)
    grad_snr_value: Optional[float] = field(default=None)
    val_dataloader: Optional[DataLoader] = field(default=None)
    val_loss: Optional[float] = field(default=None)
    #: Online-strategy validation: reward-statistics evaluator under its
    #: own sampling params. ``None`` keeps the legacy validate_online path.
    val_evaluator: Optional["RolloutEvaluator"] = field(default=None)
    async_rollout: Optional["RolloutRunner"] = field(default=None)
    optimizer_step: int = field(default=0)
    pending_samples: int = field(default=0)
    checkpoint_safe: bool = field(default=True)

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
    def dataset_len(self) -> int:
        if self.dataset is None:
            raise RuntimeError("Training dataset has not been initialized")
        return len(self.dataset)

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    def commit_update(self) -> None:
        """Record a complete optimizer/scheduler update."""
        self.optimizer_step += 1

    def commit_samples(self) -> None:
        """Commit input records after all their learner updates complete."""
        self.consumed_samples += self.pending_samples
        self.pending_samples = 0
        self.checkpoint_safe = True


def restore_training_progress(
    checkpoint: Checkpoint, config: TrainConfig, dp_size: int
) -> Tuple[int, int]:
    """Restore the saved counter and committed cursor without guessing."""
    per_step = config.batch_per_device * dp_size * config.grad_accum_steps
    step = checkpoint.meta.get(
        "optimizer_step", checkpoint.consumed_samples // per_step
    )
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Checkpoint optimizer_step must be nonnegative")
    cursor = checkpoint.consumed_samples
    if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
        raise ValueError("Checkpoint consumed_samples must be nonnegative")
    saved_microbatch = checkpoint.meta.get(
        "batch_per_device", config.batch_per_device
    ) * checkpoint.meta.get("dp_size", dp_size)
    saved_per_step = saved_microbatch * checkpoint.meta.get(
        "grad_accum_steps", config.grad_accum_steps
    )
    scheduler_step = checkpoint.extra.get("scheduler", {}).get("last_epoch")
    periodic_offset = (
        scheduler_step == step + 1
        and cursor == (step + 1) * saved_per_step - saved_microbatch
    )
    partial_cursor = (
        step * saved_per_step < cursor < (step + 1) * saved_per_step
        and cursor % saved_microbatch == 0
    )
    if config.rollout_mode != "async_round" and (periodic_offset or partial_cursor):
        raise ValueError(
            "Checkpoint progress is inconsistent or ambiguous; do not resume it "
            "unchanged. For a confirmed legacy checkpoint, run "
            "python -m scripts.tools.checkpoint SOURCE OUTPUT, then resume OUTPUT."
        )
    return step, cursor


def repair_legacy_training_progress(checkpoint: Checkpoint) -> Tuple[int, int]:
    """Explicitly normalize a checkpoint from the old sample-derived counter.

    This is a migration helper, not part of automatic resume. Callers apply the
    returned values only after identifying a legacy checkpoint.
    """
    step = checkpoint.meta["optimizer_step"]
    microbatch = checkpoint.meta["batch_per_device"] * checkpoint.meta["dp_size"]
    per_step = microbatch * checkpoint.meta["grad_accum_steps"]
    cursor = checkpoint.consumed_samples
    scheduler_step = checkpoint.extra.get("scheduler", {}).get("last_epoch")
    if scheduler_step == step + 1 and cursor == (step + 1) * per_step - microbatch:
        return step + 1, (step + 1) * per_step
    if scheduler_step == step and step * per_step <= cursor < (step + 1) * per_step:
        return step, step * per_step
    raise ValueError("Legacy checkpoint progress cannot be repaired unambiguously")


@dataclass
class _PreloadedState:
    optimizer_step: int = 0
    model_config: Dict = field(default_factory=dict)
    state_dict: Optional[Dict] = None
    epoch: int = 0
    consumed_samples: int = 0
    checkpoint: Optional[Checkpoint] = None
    component_states: Dict = field(default_factory=dict)


class TrainContextBuilder:
    def __init__(
        self,
        config: TrainConfig,
    ):
        self.config = config
        self.runtime = _TrainRuntimeDependencies.capture(config)
        self._param_path: Optional[str] = None
        self._resume: bool = False
        self._topology: Optional[ParallelTopology] = None

    def _instantiate_model(self, model_config: Dict) -> nn.Module:
        return build_model_component(
            self.config.model, self.config.model_dtype, model_config
        )

    def _instantiate_optimizer(self, model: nn.Module):
        return build_optimizer_component(self.config.optimizer, model)

    def _instantiate_scheduler(self, optimizer, dataset_len: int):
        return build_scheduler_component(
            self.config.scheduler, optimizer, dataset_len, self.config
        )

    def with_param_path(self, param_path: Optional[str], resume: bool = False) -> Self:
        self._param_path = param_path
        self._resume = resume
        return self

    def build(self) -> TrainContext:
        if (
            self.config.rollout_mode == "async_round"
            and get_world_size() != self.config.nprocs
        ):
            raise ValueError("async_round world_size must match configured nprocs")
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
        try:
            self._create_datasets(context)
            self._prepare_model(context, executor, preloaded_state)
            self._restore_optimizer_state(context, preloaded_state)
            self._create_dataloaders(context, context.dataset, context.val_dataset)

            # Strategies depend on the prepared model; online rollout depends on both.
            strategy_kwargs = self._create_strategy(context, executor)
            self._configure_rollout(context, strategy_kwargs)
            return context
        except BaseException:
            if context.async_rollout is not None:
                try:
                    context.async_rollout.close()
                except BaseException:
                    logger.exception(
                        "Failed to close rollout after training build error"
                    )
            raise

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
        return ExecutorFactory.create_checked(
            cfg.dp_mode,
            grad_accum_steps=cfg.grad_accum_steps,
            **kwargs,
        )

    def _load_preloaded_state(self) -> _PreloadedState:
        cfg = self.config
        state = _PreloadedState(
            epoch=cfg.start_epoch,
            consumed_samples=cfg.start_samples * self._topology.dp_size,
            optimizer_step=cfg.start_samples
            // (cfg.batch_per_device * cfg.grad_accum_steps),
        )
        if self._param_path:
            config_path = Path(self._param_path) / "config.json"
            if config_path.exists():
                state.model_config = adapt_config(
                    load_json(config_path), self._param_path
                )
            options = cfg.state_dict_load_options(self._resume)
            extra_keys = [name for name in ("optimizer", "scheduler") if options[name]]
            if self._resume:
                extra_keys.extend(
                    path.stem
                    for path in Path(self._param_path).glob("*.pt")
                    if path.stem not in ("optimizer", "scheduler")
                )
            checkpoint = Checkpoint.load_any(
                self._param_path, extra_keys=extra_keys, load_weights=options["model"]
            )
            if checkpoint is not None:
                if checkpoint.config:
                    checkpoint.config = adapt_config(
                        checkpoint.config, self._param_path
                    )
                if checkpoint.state_dict and looks_like_hf_state_dict(
                    checkpoint.state_dict
                ):
                    checkpoint.state_dict = convert_hf_weights(
                        checkpoint.state_dict,
                        ConfigFactory.load(checkpoint.config or state.model_config),
                    )
                checkpoint.state_dict = strip_compile_prefix(checkpoint.state_dict)
                state.state_dict = checkpoint.state_dict if options["model"] else None
                state.model_config = checkpoint.config or state.model_config
                if self._resume:
                    state.epoch = checkpoint.epoch
                    state.optimizer_step, state.consumed_samples = (
                        restore_training_progress(
                            checkpoint, cfg, self._topology.dp_size
                        )
                    )
                    state.checkpoint = checkpoint
                for name in ("optimizer", "scheduler"):
                    if name not in checkpoint.extra:
                        continue
                    saved_spec = checkpoint.meta.get(name)
                    saved_type = (
                        saved_spec.get("type")
                        if isinstance(saved_spec, dict)
                        else checkpoint.meta.get("optimizer_name")
                        if name == "optimizer"
                        else None
                    )
                    current_type = getattr(cfg, name).type
                    if saved_type and saved_type != current_type:
                        raise ValueError(
                            f"Cannot load {name} state from {saved_type!r} into {current_type!r}; "
                            f"set load_state_dict.{name}=False when changing components"
                        )
                    state.component_states[name] = checkpoint.extra[name]
        if not state.model_config:
            model = self._instantiate_model(state.model_config)
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
            optimizer_step=state.optimizer_step,
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

        def scheduler_for_context(optimizer):
            return self._instantiate_scheduler(optimizer, context.dataset_len)

        context.model, context.optimizer, context.scheduler = executor.prepare(
            lambda: self._instantiate_model(context.model_config),
            self._instantiate_optimizer,
            scheduler_for_context,
            before_wrap=before_wrap,
            after_wrap=after_wrap,
        )

    def _create_datasets(self, context: TrainContext) -> None:
        context.dataset, context.val_dataset = build_dataset_component(
            self.config.dataset, self.config.val_split, self.config.random_seed
        )

    def _create_dataloaders(
        self, context: TrainContext, train_dataset, val_dataset
    ) -> None:
        sampler_offset = context.consumed_samples // context.dp_size
        if self._resume and sampler_offset > 0:
            samples_per_replica = self._topology.samples_per_replica(len(train_dataset))
            if samples_per_replica > 0:
                context.epoch = sampler_offset // samples_per_replica
        context.dataloader = build_dataloader(
            train_dataset,
            self.config.dataset,
            self.config,
            self._topology,
            context.epoch,
            sampler_offset,
        )
        if val_dataset is not None:
            context.val_dataloader = build_dataloader(
                val_dataset,
                self.config.dataset,
                self.config,
                self._topology,
                0,
                0,
                shuffle=False,
            )

    def _restore_optimizer_state(
        self, context: TrainContext, state: Optional[_PreloadedState] = None
    ) -> None:
        extras = (
            context.checkpoint.extra
            if context.checkpoint
            else (state.component_states if state is not None else {})
        )
        if "optimizer" in extras:
            if context.checkpoint and "scheduler" in extras:
                context.optimizer.load_state_dict(extras["optimizer"])
            else:
                load_optimizer_state_dict(context.optimizer, extras["optimizer"])
            logger.info("Loaded checkpoint optimizer state")
        if "scheduler" in extras and context.scheduler is not None:
            context.scheduler.load_state_dict(extras["scheduler"])
            for group, lr in zip(
                context.optimizer.param_groups,
                context.scheduler.get_last_lr(),
                strict=True,
            ):
                group["lr"] = lr
            logger.info("Loaded checkpoint scheduler state")
        if context.checkpoint:
            restore_checkpoint_extras(context.checkpoint.extra)

    def _create_strategy(self, context: TrainContext, executor: BaseExecutor) -> Dict:
        cfg = self.config
        cp_state = None
        if self._topology is not None and self._topology.cp_size > 1:
            self._validate_cp(context)
            cp_state = CPState(self._topology)
        kwargs = cfg._validate_strategy_kwargs(dict(cfg.strategy_kwargs))
        kwargs.setdefault("moe_aux_loss_coef", cfg.moe_aux_loss_coef)
        kwargs.setdefault("rl_update_epochs", cfg.rl_update_epochs)
        kwargs.setdefault("rl_minibatch_prompts", cfg.rl_minibatch_prompts)
        kwargs.setdefault("gradient_chunked_logprobs", cfg.gradient_chunked_logprobs)
        capabilities = StrategyFactory.capabilities(cfg.strategy)
        if capabilities.reference_model:
            kwargs["ref_model"] = create_ref_model(
                lambda: self._instantiate_model(context.model_config),
                executor=executor,
                model=context.model,
                device=get_current_device(),
            )
        if capabilities.old_model:
            kwargs["old_model"] = (
                create_ref_model(
                    lambda: self._instantiate_model(context.model_config),
                    executor=executor,
                    model=context.model,
                    device=get_current_device(),
                )
                if capabilities.initialize_old_model
                else None
            )
        if capabilities.critic:
            critic, critic_optimizer = self._create_critic(context, executor)
            kwargs["critic"] = critic
            kwargs["critic_optimizer"] = critic_optimizer
            kwargs.setdefault("max_grad_norm", cfg.max_grad_norm)
        context.strategy = StrategyFactory.create_checked(
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
        self._restore_reference_model(context)
        return kwargs

    def _restore_reference_model(self, context: TrainContext) -> None:
        """Pin the resumed strategy's KL anchor to the checkpoint's reference.

        ``create_ref_model`` always rebuilds the reference from the
        *restored* actor, so without the persisted copy a resume would
        silently move the DPO/GRPO KL target — a different optimization
        objective after the restart.  Fresh runs keep the actor-derived
        reference (that *is* the intended anchor); warm starts via
        ``param_path`` without ``resume=True`` likewise.  The key and the
        missing-entry policy (fatal unless ``allow_reference_reanchor``) are
        declared in ``optional_extras.COMPONENT_EXTRAS``.
        """
        ref_model = getattr(context.strategy, "ref_model", None)
        if ref_model is None:
            return
        if not self._resume or context.checkpoint is None:
            return
        load_component_extra(
            context.checkpoint.extra, "reference_model", ref_model, self.config
        )

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
            online = StrategyFactory.capabilities(cfg.strategy).online
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
    ) -> Tuple[nn.Module, OptimizerProtocol]:
        """Build the PPO critic and its optimizer, restoring persisted state.

        The critic backbone warm-starts from the policy weights (standard
        actor-critic initialization; the fresh value head is the only
        randomly-initialized part).  On resume, the ``value_model`` /
        ``value_optimizer`` checkpoint extras override the warm start —
        and their absence is fatal rather than a silent fresh critic.  Both
        keys, and that fatal policy, are declared in
        ``optional_extras.COMPONENT_EXTRAS``.
        """
        cfg = self.config
        device = get_current_device()
        checkpoint = context.checkpoint
        critic_keys = component_extra_keys("critic", "critic_optimizer")
        if checkpoint is not None:
            require_component_extras(
                checkpoint.extra,
                critic_keys,
                strategy=cfg.strategy,
                requirement="critic state",
            )

        state_dict = executor.unwrap_model(context.model)
        if executor.use_distributed:
            state_dict = broadcast_state_dict(state_dict)
        critic = self.runtime.critic_model_fn()
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
            load_component_extra(checkpoint.extra, "value_model", critic, cfg)
        critic = critic.to(device)
        critic.train()

        optimizer_factory = (
            self.runtime.critic_optimizer_fn or self._instantiate_optimizer
        )
        critic_optimizer = optimizer_factory(critic)
        if checkpoint is not None:
            load_component_extra(
                checkpoint.extra, "value_optimizer", critic_optimizer, cfg
            )
        return critic, critic_optimizer

    def _configure_rollout(self, context: TrainContext, strategy_kwargs: Dict) -> None:
        model_factory = None
        if isinstance(self.config, TrainConfig):
            model_factory = partial(
                build_model_component,
                self.config.model,
                self.config.model_dtype,
                context.model_config,
            )
        configure_rollout(
            context=context,
            config=self.config,
            param_path=self._param_path,
            strategy_kwargs=strategy_kwargs,
            create_ref_model=create_ref_model,
            model_factory=model_factory,
            validate=self._validate_rollout_configuration,
            engine_core_cls=EngineCore,
            tokenizer_cls=AutoTokenizer,
        )

    def _validate_rollout_configuration(self, executor: BaseExecutor) -> None:
        cfg = self.config
        if not StrategyFactory.capabilities(cfg.strategy).online:
            return
        if cfg.compile_mode is not None:
            raise ValueError(
                "Online rollout does not support torch.compile; set compile_mode=None"
            )
        capabilities = executor.rollout_capabilities()
        if not capabilities.supports_in_process:
            raise ValueError(capabilities.reason or "Online rollout is unsupported")
