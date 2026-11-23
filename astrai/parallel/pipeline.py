"""Contiguous transformer stages for fixed-length sequence pretraining."""

from collections import OrderedDict
from contextlib import contextmanager
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.optimization.optimizer.precision import _gather_tensor_tree
from astrai.parallel.executor.base import BaseExecutor
from astrai.parallel.executor.gradients import MainGradientAccumulator
from astrai.parallel.setup import get_current_device


class DecoderStage(nn.Module):
    """Take ownership of one layer range, retaining checkpoint parameter names."""

    def __init__(
        self,
        lm: AutoRegressiveLM,
        boundaries: List[int],
        index: int,
        checkpoint_layers: int = 0,
    ):
        super().__init__()
        layers = lm.model.layers
        if (
            not boundaries
            or boundaries[0] != 0
            or boundaries[-1] != len(layers)
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
            or not 0 <= index < len(boundaries) - 1
        ):
            raise ValueError("pipeline boundaries must partition all decoder layers")
        if lm.config.tie_word_embeddings or lm.config.ffn_type != "mlp":
            raise ValueError(
                "pipeline pretraining requires untied embeddings and a dense MLP"
            )
        begin, end = boundaries[index : index + 2]
        if not 0 <= checkpoint_layers <= end - begin:
            raise ValueError("checkpoint count exceeds the layers on this stage")
        self.model = lm.model
        self.model.layers = nn.Sequential(
            OrderedDict(
                (
                    str(i),
                    checkpoint_wrapper(layers[i])
                    if i < begin + checkpoint_layers
                    else layers[i],
                )
                for i in range(begin, end)
            )
        )
        self.model.layer_types = self.model.layer_types[begin:end]
        if begin:
            self.model.embed_tokens = nn.Identity()
        if end == len(layers):
            self.lm_head = lm.lm_head
        else:
            self.model.norm = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)["hidden_states"]


class PipelineUpdate:
    """Run 1F1B for one DP replica, then synchronize same-stage gradients."""

    def __init__(
        self,
        module,
        mesh,
        microbatch_size,
        seq_len,
        microbatches,
        loss_chunk_size=512,
        *,
        loss_backend="cuda_linear_ce",
        label_smoothing=0.0,
        gradients=None,
    ):
        self.module = module
        pp = mesh["pp"]
        self.pp = pp.get_group()
        self.dp = mesh["dp"].get_group()
        self.index = pp.get_local_rank()
        self.last = self.index == pp.size() - 1
        self.local_batch = microbatch_size * microbatches
        self.seq_len = seq_len
        self.normalizer = self.local_batch * seq_len
        self.loss_chunk_size = loss_chunk_size
        self.loss_backend = loss_backend
        self.label_smoothing = label_smoothing
        param = next(module.parameters())
        self.device = param.device
        hidden = torch.empty(
            (microbatch_size, seq_len, module.model.config.hidden_size),
            device="meta",
            dtype=param.dtype,
            requires_grad=True,
        )
        ids = torch.empty((microbatch_size, seq_len), device="meta", dtype=torch.long)
        stage = PipelineStage(
            module,
            self.index,
            pp.size(),
            self.device,
            input_args=ids if self.index == 0 else hidden,
            output_args=hidden,
            group=self.pp,
        )
        self.schedule = Schedule1F1B(
            stage,
            microbatches,
            loss_fn=lambda hidden, targets: (
                self.loss_sum(hidden, targets) / self.normalizer
            ),
            scale_grads=False,
        )
        self.gradients = gradients or MainGradientAccumulator(
            module, synchronize=lambda: False
        )

    def loss_sum(self, hidden: Tensor, targets: Tensor) -> Tensor:
        if self.loss_backend == "torch":
            logits = self.module.lm_head(hidden).flatten(0, 1).float()
            loss = F.cross_entropy(
                logits,
                targets.flatten(),
                label_smoothing=self.label_smoothing,
                reduction="sum",
            )
        else:
            loss = self.module.lm_head(
                hidden,
                targets,
                chunk_size=self.loss_chunk_size,
                label_smoothing=self.label_smoothing,
            )
        return loss

    def backward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        expected = (self.local_batch, self.seq_len)
        if inputs.shape != expected or targets.shape != expected:
            raise ValueError(f"pipeline sequence batches must have shape {expected}")
        losses = []
        args = (inputs,) if self.index == 0 else ()
        self.schedule.step(
            *args,
            target=targets if self.last else None,
            losses=losses,
            return_outputs=False,
        )
        if any(p.main_grad is None for p in self.gradients.parameters):
            raise RuntimeError("a pipeline parameter did not receive a gradient")
        size = dist.get_world_size(self.dp)
        if size > 1:
            buffers = [bucket.buffer for bucket in self.gradients.buckets]
            works = [
                dist.all_reduce(buffer, group=self.dp, async_op=True)
                for buffer in buffers
            ]
            for work in works:
                work.wait()
            torch._foreach_mul_(buffers, 1 / size)
        loss = (
            torch.stack(losses).detach().sum()
            if self.last
            else torch.zeros((), device=self.device)
        )
        if self.last and size > 1:
            dist.all_reduce(loss, group=self.dp)
            loss.div_(size)
        dist.broadcast(loss, group=self.pp, group_src=dist.get_world_size(self.pp) - 1)
        return loss

    def clip_grad_norm(self, max_norm: float) -> Tensor:
        buffers = [b.buffer for b in self.gradients.buckets]
        squared = (
            torch.stack(torch._foreach_norm(buffers, dtype=torch.float32))
            .square()
            .sum()
        )
        dist.all_reduce(squared, group=self.pp)
        norm = squared.sqrt()
        torch._foreach_mul_(buffers, (max_norm / (norm + 1e-6)).clamp(max=1))
        return norm


class PipelineExecutor(BaseExecutor):
    """Own the local stage, its 1F1B schedule and same-stage DP synchronization."""

    def __init__(self, cfg, topology):
        super().__init__(cfg.grad_accum_steps, cfg.precision.gradient_dtype)
        self.cfg = cfg
        self.topology = topology
        self.pipeline = None
        # Stage leaders exchange CPU checkpoint tensors without staging an entire model on CUDA.
        self.writers = (
            dist.new_group(
                list(range(cfg.pp_size)), backend="gloo", use_local_synchronization=True
            )
            if topology.dp_rank == 0
            else None
        )

    @property
    def checkpoint_group(self):
        return self.topology.dp_group

    def _prepare_model(self, model):
        cfg = self.cfg
        index = self.topology.pp_rank
        depth = len(model.model.layers)
        cuts = cfg.pp_split_points or [
            i * depth // cfg.pp_size for i in range(1, cfg.pp_size)
        ]
        counts = cfg.pp_checkpoint_counts or [0] * cfg.pp_size
        self.boundaries = [0, *cuts, depth]
        stage = DecoderStage(model, self.boundaries, index, counts[index]).to(
            get_current_device()
        )
        if self.topology.dp_size > 1:
            for param in stage.parameters():
                dist.broadcast(
                    param.detach(), group=self.topology.dp_group, group_src=0
                )
        dtype = (
            next(stage.parameters()).dtype
            if self.grad_accum_dtype == "model"
            else getattr(torch, self.grad_accum_dtype)
        )
        self._grad_accumulator = MainGradientAccumulator(
            stage, synchronize=lambda: False, dtype=dtype
        )
        return stage

    def backward_window(self, model, chunks, window_samples):
        device = next(model.parameters()).device
        inputs = torch.cat([batch["input_ids"][:count] for batch, count in chunks])
        targets = torch.cat([batch["target_ids"][:count] for batch, count in chunks])
        local_batch = self.cfg.batch_per_device * self.cfg.grad_accum_steps
        if len(inputs) < local_batch:
            # Keep 1F1B message shapes fixed for partially filled final updates.
            missing = local_batch - len(inputs)
            inputs = F.pad(inputs, (0, 0, 0, missing))
            targets = F.pad(targets, (0, 0, 0, missing), value=-100)
        inputs, targets = inputs.to(device), targets.to(device)
        if self.pipeline is None:
            opts = self.cfg.strategy_kwargs
            self.pipeline = PipelineUpdate(
                model,
                self.topology.mesh,
                self.cfg.batch_per_device,
                inputs.shape[1],
                self.cfg.grad_accum_steps,
                opts.get("loss_chunk_size", 512),
                loss_backend=opts.get("loss_backend", "torch"),
                label_smoothing=opts.get("label_smoothing", 0.0),
                gradients=self._grad_accumulator,
            )
        self.pipeline.normalizer = (
            window_samples * inputs.shape[1] / self.topology.dp_size
        )
        return self.pipeline.backward(inputs, targets).item()

    def clip_grad_norm(self, model, max_norm):
        return self.pipeline.clip_grad_norm(max_norm).item()

    def validation_loss(self, model, batch):
        pp = self.topology.mesh["pp"].get_group()
        rank = self.topology.pp_rank
        param = next(model.parameters())
        targets = batch["target_ids"].to(param.device)
        if rank == 0:
            inputs = batch["input_ids"].to(param.device)
        else:
            inputs = param.new_empty((*targets.shape, model.model.config.hidden_size))
            dist.recv(inputs, group=pp, group_src=rank - 1)
        hidden = model(inputs)
        last = self.cfg.pp_size - 1
        if rank == last:
            loss = self.pipeline.loss_sum(hidden, targets) / targets.numel()
        else:
            dist.send(hidden.contiguous(), group=pp, group_dst=rank + 1)
            loss = torch.empty((), device=param.device)
        dist.broadcast(loss, group=pp, group_src=last)
        return loss

    @contextmanager
    def checkpoint_context(
        self, model, *, optimizer, export_dtype, extra_optimizers=()
    ):
        with super().checkpoint_context(
            model,
            optimizer=optimizer,
            export_dtype=export_dtype,
            extra_optimizers=extra_optimizers,
        ) as (weights, extras):
            if self.topology.dp_rank != 0:
                yield None, {}
                return
            merged, training, stages = {}, {}, []
            local = (weights, extras.get("training_model", {}), optimizer.state_dict())
            for source in range(self.cfg.pp_size):
                part = _gather_tensor_tree(
                    local, source, torch.device("cpu"), self.writers
                )
                if self.topology.pp_rank == 0:
                    merged.update(part[0])
                    training.update(part[1])
                    stages.append(part[2])
            payload = {
                "optimizer": {"pipeline_stages": stages, "boundaries": self.boundaries}
            }
            if training:
                payload["training_model"] = training
            yield (merged if self.topology.pp_rank == 0 else None), payload
