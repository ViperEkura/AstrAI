"""Contiguous transformer stages for fixed-length sequence pretraining."""

from collections import OrderedDict
from typing import List

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from astrai.model.autoregressive_lm import AutoRegressiveLM
from astrai.parallel.executor.gradients import MainGradientAccumulator


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
        self, module, mesh, microbatch_size, seq_len, microbatches, loss_chunk_size=512
    ):
        self.module = module
        pp = mesh["pp"]
        self.pp = pp.get_group()
        self.dp = mesh["dp"].get_group()
        self.index = pp.get_local_rank()
        self.last = self.index == pp.size() - 1
        self.local_batch = microbatch_size * microbatches
        self.seq_len = seq_len
        self.loss_chunk_size = loss_chunk_size
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
            stage, microbatches, loss_fn=self.loss, scale_grads=False
        )
        self.gradients = MainGradientAccumulator(module, synchronize=lambda: False)

    def loss(self, hidden: Tensor, targets: Tensor) -> Tensor:
        return self.module.lm_head(hidden, targets, chunk_size=self.loss_chunk_size) / (
            self.local_batch * self.seq_len
        )

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
        squared = torch.stack(torch._foreach_norm(buffers)).square().sum()
        dist.all_reduce(squared, group=self.pp)
        norm = squared.sqrt()
        torch._foreach_mul_(buffers, (max_norm / (norm + 1e-6)).clamp(max=1))
        return norm
