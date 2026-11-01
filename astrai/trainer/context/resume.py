"""Resolve the persisted state a run starts from.

Everything that reads a checkpoint lives here, so the builder stays an
assembler: what a stored run *means* (cursor, step count, component states,
the refusal rules) is decided in one module.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from torch import nn

from astrai.config.model_config import ConfigFactory
from astrai.config.train_config import TrainConfig
from astrai.parallel.executor import strip_compile_prefix
from astrai.parallel.topology import ParallelTopology
from astrai.serialization import (
    Checkpoint,
    adapt_config,
    convert_hf_weights,
    load_json,
    looks_like_hf_state_dict,
)
from astrai.trainer.context.state import samples_per_step
from astrai.trainer.optional_extras import (
    EXTRA_KEYS,
    EXTRAS,
    missing_resumable_extras,
)

logger = logging.getLogger(__name__)


@dataclass
class PreloadedState:
    """What the checkpoint contributes to a context that is about to be built."""

    model_config: Dict = field(default_factory=dict)
    state_dict: Optional[Dict] = None
    epoch: int = 0
    consumed_samples: int = 0
    epoch_sample_offset: int = 0
    optimizer_steps_completed: int = 0
    checkpoint: Optional[Checkpoint] = None
    component_states: Dict = field(default_factory=dict)


class ResumeLoader:
    """Read the checkpoint (or launch weights) a run starts from.

    A resume restores a whole training state or none of it: a component whose
    state is missing would silently come back fresh, which is a different run
    wearing the same step number.  The same reasoning refuses a cursor that
    cannot describe a resumable state.
    """

    def __init__(
        self,
        config: TrainConfig,
        param_path: Optional[str],
        resume: bool,
        topology: Optional[ParallelTopology],
        model_factory: Callable[[Dict], nn.Module],
    ):
        self.config = config
        self.param_path = param_path
        self.resume = resume
        self.topology = topology
        self.model_factory = model_factory

    def load(self) -> PreloadedState:
        cfg = self.config
        state = PreloadedState(
            epoch=cfg.start_epoch,
            consumed_samples=cfg.start_samples * self.topology.dp_size,
            epoch_sample_offset=cfg.start_samples * self.topology.dp_size,
        )
        if self.param_path:
            self._load_from_param_path(state)
        if not state.model_config:
            model = self.model_factory(state.model_config)
            if hasattr(model, "config"):
                state.model_config = model.config.to_dict()
        state.optimizer_steps_completed = self._step_count(state)
        return state

    def _load_from_param_path(self, state: PreloadedState) -> None:
        cfg = self.config
        config_path = Path(self.param_path) / "config.json"
        if config_path.exists():
            state.model_config = adapt_config(load_json(config_path), self.param_path)

        options = cfg.state_dict_load_options(self.resume)
        # The optional extras carry no resumable state, so a warm start
        # keeps its own RNG instead of inheriting a foreign one.
        extra_keys = [
            entry.key
            for entry in EXTRAS
            if options.get(entry.key, True) and (entry.resumable or self.resume)
        ]
        if self.resume:
            extra_keys.extend(
                path.stem
                for path in Path(self.param_path).glob("*.pt")
                if path.stem not in EXTRA_KEYS
            )
        checkpoint = Checkpoint.load_any(
            self.param_path, extra_keys=extra_keys, load_weights=options["model"]
        )
        if checkpoint is None:
            return

        if checkpoint.config:
            checkpoint.config = adapt_config(checkpoint.config, self.param_path)
        if checkpoint.state_dict and looks_like_hf_state_dict(checkpoint.state_dict):
            checkpoint.state_dict = convert_hf_weights(
                checkpoint.state_dict,
                ConfigFactory.load(checkpoint.config or state.model_config),
            )
        saved_precision = checkpoint.meta.get("precision", {})
        if (
            self.resume
            and options["model"]
            and saved_precision.get("export_dtype") is not None
        ):
            if "training_model" not in checkpoint.extra:
                raise ValueError(
                    "Cannot resume: missing original training_model weights"
                )
            checkpoint.state_dict = checkpoint.extra["training_model"]
        checkpoint.state_dict = strip_compile_prefix(checkpoint.state_dict)
        state.state_dict = checkpoint.state_dict if options["model"] else None
        state.model_config = checkpoint.config or state.model_config

        if self.resume:
            self._take_cursor(state, checkpoint, options)
        for entry in EXTRAS:
            name = entry.key
            if name not in checkpoint.extra:
                continue
            if hasattr(getattr(self.config, name, None), "type"):
                self._check_component_type(name, checkpoint)
            state.component_states[name] = checkpoint.extra[name]

    def _take_cursor(
        self, state: PreloadedState, checkpoint: Checkpoint, options: Dict
    ) -> None:
        """Adopt the checkpoint's epoch and data cursor, refusing stale layouts."""
        saved_precision = checkpoint.meta.get("precision")
        if saved_precision is not None:
            for name in (
                "compute_dtype",
                "gradient_dtype",
                "master_dtype",
                "optimizer_state_dtype",
            ):
                if saved_precision[name] != getattr(self.config.precision, name):
                    raise ValueError(f"Cannot resume: precision.{name} changed")
        saved_tail_policy = checkpoint.meta.get("epoch_tail_policy", "drop")
        if saved_tail_policy != self.config.epoch_tail_policy:
            raise ValueError(
                "Cannot resume with epoch_tail_policy="
                f"{self.config.epoch_tail_policy!r}; checkpoint uses "
                f"{saved_tail_policy!r}"
            )
        missing = missing_resumable_extras(checkpoint.extra, options)
        if missing:
            raise ValueError(
                f"Cannot resume from {self.param_path}: no "
                f"{', '.join(missing)} state in the checkpoint; set "
                f"load_state_dict.{missing[0]}=False to accept a fresh one"
            )
        state.epoch = checkpoint.epoch
        meta = checkpoint.meta
        if meta.get("resume_contract_version") != 1:
            raise ValueError(
                f"Unsupported resume contract version "
                f"{meta.get('resume_contract_version')!r}; "
                "migrate checkpoint metadata before resuming"
            )
        required = (
            "optimizer_step",
            "dataset_len",
            "epoch_sample_offset",
            "random_seed",
            "strategy",
            "dp_size",
            "cp_size",
            "tp_size",
            "dp_mode",
            "batch_per_device",
            "grad_accum_steps",
            "epoch_tail_policy",
            "rollout_mode",
            "rl_update_epochs",
            "rl_minibatch_prompts",
        )
        absent = [key for key in required if key not in meta]
        if absent:
            raise ValueError(f"Incomplete resume contract: missing {', '.join(absent)}")
        step = meta["optimizer_step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Checkpoint has an invalid optimizer_step")
        for key, current in (
            ("random_seed", self.config.random_seed),
            ("cp_size", self.config.cp_size),
            ("tp_size", self.config.tp_size),
            ("rollout_mode", self.config.rollout_mode),
            ("strategy", self.config.strategy),
            ("rl_update_epochs", self.config.rl_update_epochs),
            ("rl_minibatch_prompts", self.config.rl_minibatch_prompts),
        ):
            if meta.get(key) != current:
                raise ValueError(
                    f"Cannot resume: {key} changed from {meta.get(key)!r} "
                    f"to {current!r}"
                )
        saved_dp = meta["dp_size"]
        if isinstance(saved_dp, bool) or not isinstance(saved_dp, int) or saved_dp < 1:
            raise ValueError("Checkpoint has an invalid dp_size")
        old_batch = saved_dp * meta["batch_per_device"] * meta["grad_accum_steps"]
        new_batch = samples_per_step(self.config, self.topology.dp_size)
        if old_batch != new_batch:
            raise ValueError(
                f"Cannot resume: global effective batch changed from "
                f"{old_batch} to {new_batch}"
            )
        if saved_dp != self.topology.dp_size and (
            meta.get("dp_mode") == "fsdp" or self.config.dp_mode == "fsdp"
        ):
            raise ValueError("FSDP optimizer state cannot resume across DP sizes")
        if saved_dp != self.topology.dp_size and (
            (saved_dp > 1 and meta.get("dp_mode") != "ddp")
            or (self.topology.dp_size > 1 and self.config.dp_mode != "ddp")
        ):
            raise ValueError("DP resize requires DDP or single-device optimizer state")
        asynchronous = self.config.rollout_mode == "async_round"
        if asynchronous and any(
            meta[key] != current
            for key, current in (
                ("dp_size", self.topology.dp_size),
                ("batch_per_device", self.config.batch_per_device),
                ("grad_accum_steps", self.config.grad_accum_steps),
            )
        ):
            raise ValueError("Async rounds require the saved learner layout")
        offset = meta.get("epoch_sample_offset")
        length = meta.get("dataset_len")
        # Async rollout rounds use equal padded rank shards. Their cursor is
        # still a global committed count, with a mode-specific epoch extent.
        extent = length
        if asynchronous and isinstance(length, int) and not isinstance(length, bool):
            extent = ((length + saved_dp - 1) // saved_dp) * saved_dp
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 1
            or offset < 0
            or offset > extent
            or (offset != extent and offset % old_batch)
        ):
            raise ValueError("Checkpoint has an invalid global epoch cursor")
        state.epoch_sample_offset = offset
        cursor = checkpoint.consumed_samples
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < offset:
            raise ValueError(
                f"Checkpoint cursor {cursor!r} must be a non-negative sample "
                "count at least as large as the epoch offset"
            )
        state.consumed_samples = cursor
        state.checkpoint = checkpoint

    def _check_component_type(self, name: str, checkpoint: Checkpoint) -> None:
        """Refuse to pour one optimizer's state into another implementation."""
        saved_spec = checkpoint.meta.get(name)
        saved_type = (
            saved_spec.get("type")
            if isinstance(saved_spec, dict)
            else checkpoint.meta.get("optimizer_name")
            if name == "optimizer"
            else None
        )
        current_type = getattr(self.config, name).type
        if saved_type and saved_type != current_type:
            raise ValueError(
                f"Cannot load {name} state from {saved_type!r} into "
                f"{current_type!r}; set load_state_dict.{name}=False when "
                "changing components"
            )

    def _step_count(self, state: PreloadedState) -> int:
        """The committed step count, which is not the cursor divided by a step.

        A replayed batch yields several steps for one data position. A resume
        always uses its explicit count; a fresh start derives its initial count.
        """
        if state.checkpoint is not None:
            return state.checkpoint.meta["optimizer_step"]
        return state.consumed_samples // samples_per_step(
            self.config, self.topology.dp_size
        )
