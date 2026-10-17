"""Rank-local construction for configured training components."""

import importlib
import inspect
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from astrai.config.model_config import ConfigFactory
from astrai.config.train_config import ComponentSpec, TrainConfig
from astrai.dataset import DatasetFactory, RDSampler, dpo_collate_fn, grpo_collate_fn
from astrai.model.automodel import ModelFactory
from astrai.optimization.optimizer import OptimizerFactory
from astrai.optimization.protocols import OptimizerProtocol
from astrai.optimization.scheduler import SchedulerFactory
from astrai.parallel.executor import (
    BaseExecutor,
    broadcast_state_dict,
    strip_compile_prefix,
)
from astrai.parallel.topology import ParallelTopology


def resolve_component(path: str) -> Any:
    module_name, separator, name = path.partition(":")
    if not separator or not module_name or not name:
        raise ValueError(f"Expected module:attribute, got {path!r}")
    value = importlib.import_module(module_name)
    for part in name.split("."):
        value = getattr(value, part)
    return value


def build_model_component(
    spec: ComponentSpec, model_dtype: Optional[str], model_config: Dict
) -> nn.Module:
    """Build one model from serializable settings inside the current rank."""
    params = dict(spec.params)
    model_config = params.pop("config", model_config)
    if ModelFactory.is_registered(spec.type):
        if not model_config:
            raise ValueError("Registered model requires model config")
        model = ModelFactory.create_checked(
            spec.type, ConfigFactory.load(model_config), **params
        )
    else:
        target = resolve_component(spec.type)
        if (
            model_config
            and "config" in inspect.signature(target).parameters
            and "config" not in params
        ):
            params["config"] = dict(model_config)
        model = target(**params)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Model target {spec.type!r} must return nn.Module")
    if model_dtype is not None:
        model = model.to(dtype=getattr(torch, model_dtype))
    return model


def build_optimizer_component(spec: ComponentSpec, model: nn.Module):
    if OptimizerFactory.is_registered(spec.type):
        return OptimizerFactory.create_checked(spec.type, model, **spec.params)
    target = resolve_component(spec.type)
    parameters = list(inspect.signature(target).parameters.values())
    first_arg = (
        model.parameters()
        if parameters and parameters[0].name in ("params", "parameters")
        else model
    )
    return target(first_arg, **spec.params)


def load_optimizer_state_dict(optimizer: OptimizerProtocol, state_dict: Dict) -> None:
    """Load optimizer state_dict while retaining configured parameter-group options."""
    options = [
        deepcopy({key: value for key, value in group.items() if key != "params"})
        for group in optimizer.param_groups
    ]
    try:
        optimizer.load_state_dict(state_dict)
        for group, configured in zip(optimizer.param_groups, options, strict=True):
            parameters = group["params"]
            group.clear()
            group.update(configured, params=parameters)
    except (ValueError, KeyError, RuntimeError) as error:
        raise ValueError(
            "Cannot reuse checkpoint optimizer state with this optimizer; "
            "set load_state_dict.optimizer=False to initialize fresh state"
        ) from error


def build_scheduler_component(
    spec: ComponentSpec, optimizer, dataset_len: int, cfg: TrainConfig
):
    params = dict(spec.params)
    warmup_ratio = params.pop("warmup_ratio", 0.0)

    def ceil_div(a, b):
        return (a + b - 1) // b

    samples_per_replica = ceil_div(dataset_len, cfg.dp_size)
    batches_per_replica = ceil_div(samples_per_replica, cfg.batch_per_device)
    # Accumulation continues across epoch boundaries; only the final
    # incomplete accumulation is discarded.
    total_steps = batches_per_replica * cfg.n_epoch // cfg.grad_accum_steps
    warmup_steps = min(int(warmup_ratio * total_steps), total_steps)
    remaining = total_steps - warmup_steps
    values = {
        "$total_steps": total_steps,
        "$warmup_steps": warmup_steps,
        "$decay_steps": remaining,
        "$dataset_len": dataset_len,
    }

    def resolve(value):
        if isinstance(value, str):
            return values.get(value, value)
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    params = {key: resolve(value) for key, value in params.items()}
    if SchedulerFactory.is_registered(spec.type):
        params.setdefault("warmup_steps", warmup_steps)
        if spec.type == "cosine":
            params.setdefault("lr_decay_steps", remaining)
        elif spec.type == "sgdr":
            params.setdefault("cycle_length", remaining)
        elif spec.type == "wsd":
            stable_steps = max(1, int(remaining * 0.8))
            params.setdefault("stable_steps", stable_steps)
            params.setdefault("decay_steps", max(1, remaining - stable_steps))
        return SchedulerFactory.create_checked(spec.type, optimizer, **params)
    return resolve_component(spec.type)(optimizer, **params)


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


def load_dataset_component(kind: str, path: Optional[str], params: dict) -> Dataset:
    if kind == "packed":
        if not path:
            raise ValueError("Packed dataset requires params.path")
        return DatasetFactory.load(load_path=path, **params)
    return resolve_component(kind)(**params)


def build_dataset_component(
    spec: ComponentSpec, val_split: Optional[float], seed: int
) -> tuple[Dataset, Optional[Dataset]]:
    """Resolve and open datasets inside the training rank."""
    options = dict(spec.params)
    path = options.pop("path", None)
    val_path = options.pop("val_path", None)
    val_params = dict(options.pop("val_params", {}))
    options.pop("collate", None)
    train = load_dataset_component(spec.type, path, options)
    if isinstance(train, tuple):
        if len(train) != 2:
            raise ValueError(
                "Dataset factory must return a dataset or train/validation pair"
            )
        train, validation = train
    else:
        validation = None
    if val_path is not None:
        validation = load_dataset_component(
            spec.type if spec.type == "packed" else val_path,
            val_path if spec.type == "packed" else None,
            {**options, **val_params} if spec.type == "packed" else val_params,
        )
    if validation is None and val_split is not None:
        n_val = max(1, int(len(train) * val_split))
        generator = torch.Generator().manual_seed(seed)
        train, validation = random_split(
            train, [len(train) - n_val, n_val], generator=generator
        )
    return train, validation


def resolve_collate(spec: ComponentSpec):
    collate = spec.params.get("collate")
    if collate is not None:
        return resolve_component(collate)
    if spec.type == "packed":
        train_type = spec.params.get("train_type")
        if train_type == "dpo":
            return dpo_collate_fn
        if train_type == "grpo":
            return grpo_collate_fn
    return None


def build_dataloader(
    dataset: Dataset,
    spec: ComponentSpec,
    config: TrainConfig,
    topology: ParallelTopology,
    epoch: int,
    start_iter: int,
    shuffle: bool = True,
) -> DataLoader:
    sampler = RDSampler(
        dataset,
        start_epoch=epoch,
        start_iter=start_iter,
        seed=config.random_seed,
        shuffle=shuffle,
        process_group=topology.dp_group,
    )
    kwargs = dict(
        dataset=dataset,
        batch_size=config.batch_per_device,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=resolve_collate(spec),
    )
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)
