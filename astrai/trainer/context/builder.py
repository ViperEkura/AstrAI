"""Assemble a training context from the config, the checkpoint, and the runtime."""

import logging
from functools import partial
from typing import Dict, Optional, Self, Tuple

import torch
import torch.distributed as dist
from torch import nn

from astrai.config.train_config import TrainConfig, _TrainRuntimeDependencies
from astrai.inference.core.engine_core import EngineCore
from astrai.model.components.lora import inject_lora
from astrai.optimization.protocols import OptimizerProtocol
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
from astrai.tokenize import AutoTokenizer
from astrai.trainer.components import (
    build_dataloader,
    build_dataset_component,
    build_model_component,
    build_optimizer_component,
    build_scheduler_component,
    create_ref_model,
    stop_loader_workers,
)
from astrai.trainer.context.resume import PreloadedState, ResumeLoader
from astrai.trainer.context.state import TrainContext
from astrai.trainer.optional_extras import (
    component_extra_keys,
    load_component_extra,
    require_component_extras,
    restore_extras,
)
from astrai.trainer.rollout.setup import configure_rollout
from astrai.trainer.strategy import StrategyFactory

logger = logging.getLogger(__name__)


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

    def _instantiate_scheduler(
        self, optimizer, dataset_len: int, *, global_cursor: bool = True
    ):
        return build_scheduler_component(
            self.config.scheduler,
            optimizer,
            dataset_len,
            self.config,
            global_cursor=global_cursor,
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
        if self.config.rollout_mode == "async_round" or (
            not self._resume and self.config.start_samples > 0
        ):
            context.global_cursor = False
        if (
            self.config.strategy in ("grpo", "online_grpo")
            and not context.global_cursor
            and self.config.grad_accum_steps != 1
        ):
            raise ValueError(
                "Legacy GRPO requires grad_accum_steps=1; use "
                "rl_microbatch_prompts within complete update windows"
            )
        try:
            self._create_datasets(context)
            if (
                self._resume
                and context.global_cursor
                and context.checkpoint is not None
            ):
                saved_len = context.checkpoint.meta["dataset_len"]
                if context.dataset_len != saved_len:
                    raise ValueError(
                        f"Cannot resume: dataset length changed from {saved_len} "
                        f"to {context.dataset_len}"
                    )
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

    def _load_preloaded_state(self) -> PreloadedState:
        """Read what the checkpoint contributes; a stub point for tests."""
        return ResumeLoader(
            self.config,
            self._param_path,
            self._resume,
            self._topology,
            self._instantiate_model,
        ).load()

    def _create_context(
        self, state: PreloadedState, executor: BaseExecutor
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
            epoch_sample_offset=state.epoch_sample_offset,
            global_cursor=state.global_cursor,
            optimizer_steps_completed=state.optimizer_steps_completed,
            checkpoint=state.checkpoint,
            param_path=self._param_path,
        )

    def _prepare_model(
        self, context: TrainContext, executor: BaseExecutor, state: PreloadedState
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
            return self._instantiate_scheduler(
                optimizer, context.dataset_len, global_cursor=context.global_cursor
            )

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
        sampler_offset = (
            context.epoch_sample_offset
            if context.global_cursor
            else context.consumed_samples // context.dp_size
        )
        if not context.global_cursor and self._resume and sampler_offset > 0:
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
            global_cursor=context.global_cursor,
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
            context.on_shutdown(
                "val_loader", partial(stop_loader_workers, context.val_dataloader)
            )

    def _restore_optimizer_state(
        self, context: TrainContext, state: Optional[PreloadedState] = None
    ) -> None:
        """Hand every declared extra to its restorer; the table owns the order."""
        extras = (
            context.checkpoint.extra
            if context.checkpoint
            else (state.component_states if state is not None else {})
        )
        if not extras:
            return
        restore_extras(context, extras)
        logger.info("Restored %s from the checkpoint", ", ".join(sorted(extras)))

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
        if cfg.strategy in ("grpo", "online_grpo"):
            kwargs.setdefault("rl_microbatch_prompts", cfg.rl_microbatch_prompts)
            kwargs["loss_process_group"] = getattr(
                context.model, "process_group", self._topology.dp_group
            )
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

        state_dict = executor.model_state_dict(context.model)
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
        if context.dp_size > 1 and dist.is_initialized():
            # The value head is freshly initialized on each rank. Start every
            # replica from rank zero's head before reducing its gradients.
            source = (
                context.topology.cp_rank * context.topology.tp_size
                + context.topology.tp_rank
            )
            for tensor in critic.state_dict().values():
                dist.broadcast(tensor, src=source, group=context.topology.dp_group)
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
        configure_rollout(
            context=context,
            config=self.config,
            param_path=self._param_path,
            strategy_kwargs=strategy_kwargs,
            create_ref_model=create_ref_model,
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
