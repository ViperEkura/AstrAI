"""Composable update precision and configurable persistent optimizer storage."""

from contextlib import contextmanager
from typing import Dict, Iterable, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import SGD, AdamW, Optimizer
from torch.utils._pytree import tree_flatten, tree_unflatten

from astrai.optimization.optimizer.muon.optimizer import MuonAdamW
from astrai.optimization.optimizer.muon.sharded_update import ShardedMuon
from astrai.optimization.optimizer.offload import (
    CPUStateOffload,
    cast_state,
    restore_state_precision,
)
from astrai.parallel.executor.gradients import parameter_gradient


def _parameter_owners(optimizers, world_size):
    costs = {}
    for child in optimizers:
        factor = 3 if isinstance(child, AdamW) else 2
        for group in child.param_groups:
            for parameter in group["params"]:
                costs[parameter] = parameter.numel() * 4 * factor
    loads = [0] * world_size
    owners = {}
    for parameter in sorted(costs, key=costs.__getitem__, reverse=True):
        owner = min(range(world_size), key=loads.__getitem__)
        owners[parameter] = owner
        loads[owner] += costs[parameter]
    return owners


def _gather_tensor_tree(value, source, device, group):
    """Collect CPU snapshots on rank zero without a full GPU staging buffer."""
    rank = dist.get_rank(group)
    source_rank = dist.get_global_rank(group, source)
    writer_rank = dist.get_global_rank(group, 0)
    if rank == source:
        leaves, spec = tree_flatten(value)
        descriptions = [
            (True, tuple(leaf.shape), leaf.dtype)
            if isinstance(leaf, torch.Tensor)
            else (False, leaf)
            for leaf in leaves
        ]
        metadata = [spec, descriptions]
    else:
        leaves = None
        metadata = [None, None]
    dist.broadcast_object_list(metadata, src=source_rank, group=group)
    spec, descriptions = metadata
    result = []
    for index, description in enumerate(descriptions):
        if not description[0]:
            if rank == 0:
                result.append(description[1])
            continue
        if source == 0:
            if rank == 0:
                result.append(leaves[index].detach().to(device="cpu", copy=True))
        elif rank == source:
            dist.send(
                leaves[index].detach().to(device=device).contiguous(),
                dst=writer_rank,
                group=group,
            )
        elif rank == 0:
            received = torch.empty(description[1], dtype=description[2], device=device)
            dist.recv(received, src=source_rank, group=group)
            result.append(received.to(device="cpu", copy=True))
            del received
    return tree_unflatten(result, spec) if rank == 0 else None


def _merge_optimizer_state(destination, source):
    if "state" in destination and "param_groups" in destination:
        destination["state"].update(source["state"])
    else:
        for name in destination:
            _merge_optimizer_state(destination[name], source[name])


def _global_optimizer_state(optimizer, model_groups, masters):
    """Serialize stable full-parameter IDs without changing live groups."""
    saved = optimizer.state_dict()
    indices = {
        model: index
        for index, model in enumerate(
            model for models in model_groups for model in models
        )
    }
    master_indices = {
        master: indices[model] for model, master in masters.items() if model in indices
    }
    remap = {
        index: master_indices[master]
        for saved_group, live_group in zip(
            saved["param_groups"], optimizer.param_groups
        )
        for index, master in zip(saved_group["params"], live_group["params"])
    }
    return {
        **saved,
        "state": {remap[index]: value for index, value in saved["state"].items()},
        "param_groups": [
            {**group, "params": [indices[model] for model in models]}
            for group, models in zip(saved["param_groups"], model_groups)
        ],
    }


def _local_optimizer_state(saved, model_groups, owners, rank):
    groups = []
    selected = set()
    if len(saved["param_groups"]) != len(model_groups):
        raise ValueError("Checkpoint optimizer groups do not match the model")
    for group, models in zip(saved["param_groups"], model_groups):
        if len(group["params"]) != len(models):
            raise ValueError("Checkpoint optimizer groups do not match the model")
        indices = [
            index
            for index, parameter in zip(group["params"], models)
            if owners[parameter] == rank
        ]
        selected.update(indices)
        groups.append({**group, "params": indices})
    return {
        **saved,
        "state": {
            index: state for index, state in saved["state"].items() if index in selected
        },
        "param_groups": groups,
    }


class PrecisionOptimizer(Optimizer):
    """Compute updates in FP32 and persist the configured storage formats.

    This owns update precision only. Gradients arrive after model-side
    accumulation/reduction; casting them here cannot recover lost precision.
    CPU offload stores masters and optimizer state in host memory between
    steps. Pinned host state uses a bounded upload/compute/return pipeline.
    Each step completes host copies before returning; forward/backward do
    not overlap with optimizer mutation. Optional state sharding assigns
    complete parameters to ranks and broadcasts updated compute weights.
    """

    supports_main_grad = True

    def __init__(
        self,
        optimizer: Optimizer,
        cpu_offload: bool = False,
        master_dtype: torch.dtype = torch.float32,
        state_dtype: torch.dtype = torch.float32,
        shard_state: bool = False,
        process_group=None,
    ):
        if not isinstance(shard_state, bool):
            raise ValueError("shard_state must be a boolean")
        if master_dtype not in (torch.float32, torch.bfloat16) or state_dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("Master/state storage supports float32 and bfloat16")
        self.master_dtype = master_dtype
        self.state_dtype = state_dtype
        groups = optimizer.param_groups
        parameters = [p for group in groups for p in group["params"]]
        if any(isinstance(p, DTensor) for p in parameters):
            raise NotImplementedError(
                "FP32 master weights for DTensor/FSDP/TP parameters are not "
                "validated yet; use FP32 model parameters"
            )
        if any(p.dtype not in (torch.float32, torch.bfloat16) for p in parameters):
            raise ValueError(
                "PrecisionOptimizer supports FP32 and BF16 compute parameters"
            )
        self._optimizers = (
            (optimizer.muon, optimizer.adamw)
            if isinstance(optimizer, MuonAdamW)
            else (optimizer,)
        )
        if any(child.state for child in self._optimizers):
            raise ValueError("Wrap the optimizer before its first update")
        if (
            cpu_offload
            and any(p.is_cuda for p in parameters)
            and any(
                not isinstance(child, (AdamW, SGD, ShardedMuon))
                for child in self._optimizers
            )
        ):
            raise NotImplementedError(
                "CPU state offload supports AdamW, SGD and Muon optimizers"
            )
        self._all_model_params = parameters
        self._model_groups = [
            [list(group["params"]) for group in child.param_groups]
            for child in self._optimizers
        ]
        self._owners = None
        self._checkpoint_state = None
        self.process_group = process_group
        if shard_state and dist.is_initialized():
            self.process_group = (
                process_group if process_group is not None else dist.group.WORLD
            )
            if dist.get_rank(self.process_group) < 0:
                raise ValueError("Optimizer rank must belong to its process group")
        if (
            shard_state
            and dist.is_initialized()
            and dist.get_world_size(self.process_group) > 1
        ):
            if any(
                not isinstance(child, (AdamW, SGD, ShardedMuon))
                for child in self._optimizers
            ):
                raise NotImplementedError("State sharding supports AdamW, SGD and Muon")
            if len({parameter.device for parameter in parameters}) != 1:
                raise ValueError("Sharded optimizer parameters must share one device")
            self._owners = _parameter_owners(
                self._optimizers, dist.get_world_size(self.process_group)
            )
            rank = dist.get_rank(self.process_group)
            parameters = [
                parameter for parameter in parameters if self._owners[parameter] == rank
            ]
            for group in groups:
                group["params"] = [
                    parameter
                    for parameter in group["params"]
                    if self._owners[parameter] == rank
                ]
        super().__init__(groups, optimizer.defaults)
        self.optimizer = optimizer
        self._model_params = parameters
        stage_on_gpu = cpu_offload and any(p.is_cuda for p in parameters)
        if stage_on_gpu and len({p.device for p in parameters}) != 1:
            raise ValueError("CPU state offload requires one CUDA model device")
        self._master_params = []
        for model in self._model_params:
            if not cpu_offload and model.dtype == master_dtype == torch.float32:
                self._master_params.append(model)
                continue
            value = model.detach().to(
                device="cpu" if cpu_offload else model.device,
                dtype=master_dtype,
                copy=True,
            )
            if stage_on_gpu:
                value = value.pin_memory()
            if master_dtype == torch.bfloat16 and model.dtype == torch.float32:
                with torch.no_grad():
                    model.copy_(value)
            self._master_params.append(
                nn.Parameter(value, requires_grad=model.requires_grad)
            )
        masters = dict(zip(self._model_params, self._master_params))
        # Composite optimizers share these group objects with their children.
        for group in groups:
            group["params"] = [masters.get(p, p) for p in group["params"]]
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

        self._offload = (
            CPUStateOffload(
                self._model_params, self._master_params, self._optimizers, state_dtype
            )
            if stage_on_gpu
            else None
        )
        self._convert_state = (
            state_dtype != torch.float32 or master_dtype != torch.float32
        )
        if (
            self._owners is not None
            and master_dtype == torch.bfloat16
            and any(
                parameter.dtype == torch.float32 for parameter in self._all_model_params
            )
        ):
            self._publish_models()

    def master_state_dict(
        self, named_parameters: Iterable[Tuple[str, nn.Parameter]]
    ) -> Dict[str, torch.Tensor]:
        """Expose managed master views under the caller's canonical parameter names."""
        if self._owners is not None:
            if self._checkpoint_state is None:
                raise RuntimeError(
                    "Enter optimizer.checkpoint_context() on every rank before exporting sharded masters"
                )
            masters = dict(
                zip(self._all_model_params, self._checkpoint_state["master_params"])
            )
        else:
            masters = dict(zip(self._model_params, self._master_params))
        return {
            name: masters[parameter].detach()
            for name, parameter in named_parameters
            if parameter in masters
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._checkpoint_state = None
        result = (
            self._offload.step() if self._offload is not None else self._step_resident()
        )
        self._publish_models()
        return loss if closure is not None else result

    @torch.no_grad()
    def _step_resident(self):
        if not self._convert_state:
            for model, master in zip(self._model_params, self._master_params):
                gradient = parameter_gradient(model)
                master.grad = None if gradient is None else gradient.detach().float()
            result = self.optimizer.step()
            for model, master in zip(self._model_params, self._master_params):
                if model is not master:
                    model.copy_(master)
            return result

        stored = [master.data for master in self._master_params]
        try:
            for model, master, value in zip(
                self._model_params, self._master_params, stored
            ):
                gradient = parameter_gradient(model)
                master.data = value.float()
                master.grad = None if gradient is None else gradient.detach().float()
            for child in self._optimizers:
                for master in child.state:
                    if master.grad is not None:
                        child.state[master] = cast_state(
                            child.state[master], torch.float32
                        )
            result = self.optimizer.step()
            for model, master, value in zip(
                self._model_params, self._master_params, stored
            ):
                if value.data_ptr() != master.data_ptr():
                    value.copy_(master)
                # Publish from persistent storage, including its quantization.
                model.copy_(value)
            restore_state_precision(self._optimizers, self.state_dtype)
            return result
        finally:
            for master, value in zip(self._master_params, stored):
                if master.dtype != value.dtype:
                    master.grad = None
                master.data = value

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for parameter in self._all_model_params:
            if getattr(parameter, "main_grad", None) is not None:
                if set_to_none:
                    parameter.main_grad = None
                else:
                    parameter.main_grad.zero_()
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    if parameter.grad.grad_fn is not None:
                        parameter.grad.detach_()
                    else:
                        parameter.grad.requires_grad_(False)
                    parameter.grad.zero_()

    def _publish_models(self):
        if self._owners is None:
            return
        pending = [
            dist.broadcast(
                parameter.detach(),
                src=dist.get_global_rank(self.process_group, self._owners[parameter]),
                group=self.process_group,
                async_op=True,
            )
            for parameter in self._all_model_params
        ]
        for work in pending:
            work.wait()

    @contextmanager
    def checkpoint_context(self):
        """All ranks assemble the existing full checkpoint format on rank zero."""
        if self._owners is None:
            yield
            return
        masters = dict(zip(self._model_params, self._master_params))
        states = [
            _global_optimizer_state(child, groups, masters)
            for child, groups in zip(self._optimizers, self._model_groups)
        ]
        local = {
            "optimizer": dict(zip(("muon", "adamw"), states))
            if isinstance(self.optimizer, MuonAdamW)
            else states[0],
            "master_params": {
                index: masters[parameter].detach()
                for index, parameter in enumerate(self._all_model_params)
                if parameter in masters
            },
            "precision": {
                "master_dtype": str(self.master_dtype).removeprefix("torch."),
                "optimizer_state_dtype": str(self.state_dtype).removeprefix("torch."),
            },
        }
        full = None
        for source in range(dist.get_world_size(self.process_group)):
            part = _gather_tensor_tree(
                local, source, self._all_model_params[0].device, self.process_group
            )
            if dist.get_rank(self.process_group) == 0:
                if full is None:
                    full = part
                else:
                    full["master_params"].update(part["master_params"])
                    _merge_optimizer_state(full["optimizer"], part["optimizer"])
        if full is not None:
            full["master_params"] = [
                full["master_params"][index]
                for index in range(len(self._all_model_params))
            ]
        self._checkpoint_state = full
        try:
            yield
        finally:
            self._checkpoint_state = None

    def state_dict(self) -> Dict:
        if self._owners is not None:
            if self._checkpoint_state is None:
                raise RuntimeError(
                    "Enter optimizer.checkpoint_context() on every rank before saving sharded state"
                )
            return self._checkpoint_state
        return {
            "optimizer": self.optimizer.state_dict(),
            "master_params": [p.detach() for p in self._master_params],
            "precision": {
                "master_dtype": str(self.master_dtype).removeprefix("torch."),
                "optimizer_state_dtype": str(self.state_dtype).removeprefix("torch."),
            },
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: Dict):
        if "master_params" not in state_dict or "optimizer" not in state_dict:
            raise ValueError(
                "Checkpoint has no master weights. Start a new run from "
                "its model weights with load_state_dict.optimizer=False; "
                "previously rounded updates cannot be recovered."
            )
        saved = state_dict["master_params"]
        if (
            "precision" not in state_dict
            or "master_dtype" not in state_dict["precision"]
        ):
            raise ValueError("Checkpoint is missing explicit optimizer precision")
        saved_dtype = state_dict["precision"]["master_dtype"]
        if saved_dtype not in ("float32", "bfloat16"):
            raise ValueError("Checkpoint master precision is unsupported")
        if len(saved) != len(self._all_model_params) or any(
            value.dtype != getattr(torch, saved_dtype) or value.shape != model.shape
            for value, model in zip(saved, self._all_model_params)
        ):
            raise ValueError("Checkpoint master weights do not match the model")
        optimizer_state = state_dict["optimizer"]
        if self._owners is not None:
            rank = dist.get_rank(self.process_group)
            if isinstance(self.optimizer, MuonAdamW):
                optimizer_state = {
                    name: _local_optimizer_state(
                        optimizer_state[name], groups, self._owners, rank
                    )
                    for name, groups in zip(("muon", "adamw"), self._model_groups)
                }
            else:
                optimizer_state = _local_optimizer_state(
                    optimizer_state, self._model_groups[0], self._owners, rank
                )
            saved = [
                value
                for model, value in zip(self._all_model_params, saved)
                if self._owners[model] == rank
            ]
        self._checkpoint_state = None
        # Native loaders cast moments to parameter dtype. Present FP32 update
        # parameters while loading to avoid rounding FP32 state via BF16 masters.
        stored = [master.data for master in self._master_params]
        try:
            for master, value in zip(self._master_params, stored):
                master.data = value.float()
            self.optimizer.load_state_dict(optimizer_state)
        finally:
            for master, value in zip(self._master_params, stored):
                master.data = value
        restore_state_precision(self._optimizers, self.state_dtype)
        if self._offload is not None:
            self._offload.pin_state()
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        for model, master, value in zip(self._model_params, self._master_params, saved):
            master.copy_(value)
            model.copy_(master)
        self._publish_models()
