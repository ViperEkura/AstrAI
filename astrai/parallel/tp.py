"""Tensor parallelism primitives — universal, executor-free.

Shards the *feature* dimension across a tp group by hijacking the one
operator that carries weights: :class:`~astrai.model.components.linear.Linear`.
A plan maps module keys (``fnmatch`` patterns over ``named_modules()``) to
split styles:

* ``colwise`` — weight ``[out, in]`` chunks along dim 0.  Each rank computes
  its own output features (attention heads / ffn channels); no communication.
* ``rowwise`` — weight chunks along dim 1.  Each rank computes a partial sum
  over its input features; the patched forward all-reduces once at the end.

Everything between a colwise and a rowwise projection is per-feature
elementwise (RoPE is per head, SiLU per channel, norms run on the fully
reduced hidden), so the model code between the two ends of a shard never
observes the split — attention runs on ``H/tp`` local heads, the MLP on
``ffn/tp`` local channels, and the hidden state is whole at every residual
add.  ``lm_head`` and the embedding stay replicated (v1).

Like CP, sharding rides a context manager: :meth:`TPState.shard` mutates
matched parameters in place and patches their forward, then exits without
restoring — autograd saves parameter views and the optimizer is built from
the shards, so the shard is final, not a scope.  Weights are broadcast from
the tp-group leader before chunking so peers shard identical tensors
(regardless of per-rank init or checkpoint reads).

Loss/gradient protocol: every rank evaluates the *full* loss (the hidden
state is replicated at each rowwise boundary), so a rank's gradient on its
own shard is already the exact single-device gradient — no rescaling, and
data-parallel reducers must average over the dp dimension only (the topology
hands out that group; executors must not fall back to the world group).
"""

import fnmatch
import logging
from contextlib import contextmanager
from typing import Dict, Generator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.model.components.linear import Linear
from astrai.parallel.topology import ParallelTopology

logger = logging.getLogger(__name__)

#: Default plan for the standard AutoRegressiveLM module layout: attention
#: projections split over heads, MLP over ffn channels.  The lm_head and
#: embedding stay replicated.  MoE expert MLPs also match (``*.mlp`` under
#: experts), but MoE + tp is refused at the builder level.
DEFAULT_TP_PLAN: Dict[str, str] = {
    "*.attention.q_proj": "colwise",
    "*.attention.k_proj": "colwise",
    "*.attention.v_proj": "colwise",
    "*.attention.o_proj": "rowwise",
    "*.mlp.up": "colwise",
    "*.mlp.gate": "colwise",
    "*.mlp.down": "rowwise",
}

_SPLIT_STYLES = ("colwise", "rowwise")


class _SumAcrossRanks(torch.autograd.Function):
    """Rowwise companion (Megatron's ``g``): forward all-reduce, backward identity.

    The output row is the same sum on every rank, so a downstream gradient
    is already each rank's full contribution to its own partial — the
    adjoint of summing partials is passing ``g`` through untouched.
    """

    @staticmethod
    def forward(ctx, x: Tensor, group) -> Tensor:
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
        return y

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad, None


class _GradSumAcrossRanks(torch.autograd.Function):
    """Colwise companion (Megatron's ``f``): forward identity, backward all-reduce.

    The input is replicated, but each rank's linear backward produces only
    its own output features' slice of ``dL/dx``; summing the slices over
    the tp group reconstructs the full input gradient.
    """

    @staticmethod
    def forward(ctx, x: Tensor, group) -> Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad: Tensor):
        g = grad.clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=ctx.group)
        return g, None


def _tp_forward(self: Linear, x: Tensor) -> Tensor:
    if self.tp_split == "colwise":
        # Gradients w.r.t. the (replicated) input accumulate across ranks
        # in backward; the local output features need no forward traffic.
        x = _GradSumAcrossRanks.apply(x, self.tp_group)
        return F.linear(x, self.weight, self.bias)
    y = F.linear(x, self.weight, self.bias)
    return _SumAcrossRanks.apply(y, self.tp_group)


class TPState:
    """Per-run tensor-parallel coordinator.

    Held by whoever prepares the model (the trainer builder today); knows
    only the topology, never an executor or a strategy.
    """

    def __init__(self, topology: ParallelTopology):
        if topology.tp_size <= 1 or topology.tp_group is None:
            raise ValueError(
                f"TPState requires an active tp dimension, got tp_size="
                f"{topology.tp_size}"
            )
        self.topology = topology

    @property
    def size(self) -> int:
        return self.topology.tp_size

    @property
    def rank(self) -> int:
        return self.topology.tp_rank

    @property
    def group(self):
        return self.topology.tp_group

    def _shard_parameter(self, module: Linear, name: str, split: str) -> None:
        """Broadcast then chunk one Linear's weight (and bias) in place."""
        requires_grad = module.weight.requires_grad
        weight = module.weight.data
        dim = 0 if split == "colwise" else 1
        if weight.shape[dim] % self.size != 0:
            raise ValueError(
                f"tp plan entry '{name}' ({split}) cannot shard weight "
                f"shape {tuple(weight.shape)} along dim {dim} by tp_size="
                f"{self.size}"
            )
        # Peers must shard identical tensors: per-rank init or checkpoint
        # reads are not guaranteed to agree, so take the leader's copy.
        src = dist.get_global_rank(self.group, 0)
        dist.broadcast(weight, src=src, group=self.group)
        shard = weight.chunk(self.size, dim=dim)[self.rank].contiguous()
        module.weight = nn.Parameter(shard, requires_grad=requires_grad)

        if module.bias is not None:
            requires_grad = module.bias.requires_grad
            bias = module.bias.data
            if split == "colwise":
                dist.broadcast(bias, src=src, group=self.group)
                bias_shard = bias.chunk(self.size, dim=0)[self.rank].contiguous()
                module.bias = nn.Parameter(bias_shard, requires_grad=requires_grad)
            else:
                # Every rank's partial output adds the full bias before the
                # all-reduce, so scale by 1/tp to keep the sum at one bias.
                dist.broadcast(bias, src=src, group=self.group)
                module.bias = nn.Parameter(
                    bias / self.size, requires_grad=requires_grad
                )

        module.tp_split = split
        module.tp_group = self.group
        module.forward = _tp_forward.__get__(module, type(module))

    @contextmanager
    def shard(
        self, model: nn.Module, plan: Optional[Dict[str, str]] = None
    ) -> Generator[nn.Module, None, None]:
        """Shard ``model``'s matched Linears across the tp group.

        Parameters are replaced by their rank-local chunk and the module's
        forward is patched in place; nothing is restored on exit (see the
        module docstring).  Yields the model for use inside the context;
        callers normally apply the plan once at build time and keep training
        inside it.
        """
        resolved = plan if plan is not None else DEFAULT_TP_PLAN
        bad_styles = {
            key: style for key, style in resolved.items() if style not in _SPLIT_STYLES
        }
        if bad_styles:
            raise ValueError(
                f"tp plan split styles must be one of {_SPLIT_STYLES}, got {bad_styles}"
            )

        hits = {key: 0 for key in resolved}
        sharded = 0
        for name, module in model.named_modules():
            if not isinstance(module, Linear):
                continue
            split = next(
                (
                    style
                    for pattern, style in resolved.items()
                    if fnmatch.fnmatch(name, pattern)
                ),
                None,
            )
            if split is None:
                continue
            hits[next(p for p in resolved if fnmatch.fnmatch(name, p))] += 1
            self._shard_parameter(module, name, split)
            sharded += 1

        self._rescale_attention_heads(model, resolved)

        unmatched = [key for key, count in hits.items() if count == 0]
        if unmatched:
            # Fail loudly: a typo'd pattern would silently leave projections
            # replicated and the colwise/rowwise pairing broken.
            available = [
                name
                for name, module in model.named_modules()
                if isinstance(module, Linear)
            ]
            raise ValueError(
                f"tp plan entries matched no module: {sorted(unmatched)}; "
                f"available Linear keys include {available[:8]}"
            )

        logger.info(
            "tensor parallelism: sharded %d Linear modules over tp_size=%d (rank %d)",
            sharded,
            self.size,
            self.rank,
        )
        try:
            yield model
        finally:
            # No restore: autograd saves parameter views across the context
            # boundary and the optimizer is built from the shards.
            pass

    def _rescale_attention_heads(
        self, model: nn.Module, resolved: Dict[str, str]
    ) -> None:
        """Divide head metadata on attention modules whose q_proj was sharded.

        Colwise-sharded projections emit ``n_heads / tp`` heads per rank,
        but the module's ``_split_heads`` reshapes with the global counts.
        kv-head replication (GQA with tp > n_kv_heads) is not supported —
        the chunk boundary would cut through a kv head.
        """
        for name, module in model.named_modules():
            if not hasattr(module, "n_heads") or not hasattr(module, "q_proj"):
                continue
            if not any(
                fnmatch.fnmatch(f"{name}.q_proj", pattern)
                for pattern in resolved
                if resolved[pattern] == "colwise"
            ):
                continue
            n_heads = module.n_heads
            n_kv_heads = getattr(module, "n_kv_heads", n_heads)
            if n_heads % self.size or n_kv_heads % self.size:
                raise ValueError(
                    f"attention '{name}' cannot shard heads by tp_size="
                    f"{self.size}: n_heads={n_heads}, n_kv_heads={n_kv_heads}; "
                    f"kv-head replication is not supported"
                )
            module.n_heads = n_heads // self.size
            module.n_kv_heads = n_kv_heads // self.size
            if hasattr(module, "n_rep"):
                module.n_rep = module.n_heads // module.n_kv_heads
