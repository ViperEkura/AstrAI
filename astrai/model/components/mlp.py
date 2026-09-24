from typing import Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.factory import BaseFactory
from astrai.model.components.linear import Linear


class FFNFactory(BaseFactory[nn.Module]):
    pass


class RouterStats(TypedDict):
    """Per-layer MoE routing statistics for training diagnostics.

    Both tensors are detached monitoring data produced during forward.
    """

    probs: Tensor
    topk_indices: Tensor


class FFNOutput(TypedDict):
    hidden_states: Tensor
    aux_loss: Optional[Tensor]
    router_stats: Optional[RouterStats]


@FFNFactory.register("mlp")
class MLP(nn.Module):
    def __init__(self, dim: int, dim_ffn: int, down_init_std: float = 0.02):
        super().__init__()
        self.up = Linear(dim, dim_ffn)
        self.gate = Linear(dim, dim_ffn)
        self.down = Linear(dim_ffn, dim, init_std=down_init_std)

    def forward(self, x: Tensor) -> FFNOutput:
        gated = self.up(x) * F.silu(self.gate(x))
        out = self.down(gated)
        return {"hidden_states": out, "aux_loss": None, "router_stats": None}


@FFNFactory.register("moe")
class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_ffn: int,
        n_routed_experts: int,
        n_shared_experts: int = 1,
        n_activated_experts: int = 2,
        topk_method: str = "greedy",
        n_layers: int = 1,
        moe_intermediate_size: Optional[int] = None,
        shared_expert_intermediate_size: Optional[int] = None,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        if n_routed_experts <= 0:
            raise ValueError("n_routed_experts must be positive")
        if n_shared_experts < 0:
            raise ValueError("n_shared_experts must be non-negative")
        if n_activated_experts <= 0:
            raise ValueError("n_activated_experts must be positive")
        if n_activated_experts > n_routed_experts:
            raise ValueError("n_activated_experts cannot exceed n_routed_experts")
        if topk_method not in (None, "greedy"):
            raise ValueError(f"unsupported topk_method: {topk_method!r}")
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.topk_method = topk_method or "greedy"
        self.norm_topk_prob = norm_topk_prob

        expert_dim_ffn = (
            moe_intermediate_size if moe_intermediate_size is not None else dim_ffn
        )
        shared_dim_ffn = (
            shared_expert_intermediate_size
            if shared_expert_intermediate_size is not None
            else dim_ffn
        )

        self.router = Linear(dim, n_routed_experts, bias=False)
        moe_scale = 1 / max(n_shared_experts, 1) + 1 / n_activated_experts
        down_init_std = 0.02 / (2 * n_layers * moe_scale) ** 0.5

        self.shared_experts = nn.ModuleList(
            [
                MLP(dim, shared_dim_ffn, down_init_std=down_init_std)
                for _ in range(n_shared_experts)
            ]
        )
        self.routed_experts = nn.ModuleList(
            [
                MLP(dim, expert_dim_ffn, down_init_std=down_init_std)
                for _ in range(n_routed_experts)
            ]
        )

    def forward(self, x: Tensor) -> FFNOutput:
        include_aux_loss = self.training and torch.is_grad_enabled()
        shape = x.shape
        dim = shape[-1]
        x_flat = x.view(-1, dim)

        shared_out = self._shared_forward(x_flat)
        routed_output = self._routed_forward(x_flat, include_aux_loss)

        out = (shared_out + routed_output["hidden_states"]).view(shape)
        return {
            "hidden_states": out,
            "aux_loss": routed_output["aux_loss"],
            "router_stats": routed_output["router_stats"],
        }

    def _shared_forward(self, x: Tensor) -> Tensor:
        if self.n_shared_experts == 0:
            return torch.zeros_like(x)
        return (
            sum(e(x)["hidden_states"] for e in self.shared_experts)
            / self.n_shared_experts
        )

    def _select_experts(self, router_logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return FP32 probabilities and the actual dispatch decision."""
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(
            router_probs,
            self.n_activated_experts,
            dim=-1,
            sorted=False,
        )
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return router_probs, topk_weights, topk_indices

    def _routed_forward(self, x: Tensor, include_aux_loss: bool) -> FFNOutput:
        N, D = x.shape
        K = self.n_activated_experts
        E = self.n_routed_experts

        router_logits = self.router(x)
        # Select experts from FP32 probabilities. Casting the full distribution
        # to BF16 first can collapse close candidates into ties and change the
        # selected expert set.
        router_probs, topk_weights, topk_indices = self._select_experts(router_logits)

        aux_loss = None
        router_stats = None
        if include_aux_loss:
            expert_load = F.one_hot(topk_indices, num_classes=E).float()
            expert_load = expert_load.mean(dim=(0, 1))
            router_prob = router_probs.mean(dim=0)
            aux_loss = E * (expert_load * router_prob).sum()
            router_stats = {
                "probs": router_probs.detach(),
                "topk_indices": topk_indices,
            }

        # Grouped dispatch: sort (token, slot) pairs by expert so each expert
        # consumes one contiguous slice instead of a per-expert mask scan.
        flat_experts = topk_indices.reshape(-1)
        sorted_experts, order = torch.sort(flat_experts)
        flat_tokens = x.repeat_interleave(K, dim=0)[order]
        flat_weights = topk_weights.to(x.dtype).reshape(-1, 1)[order]
        boundaries = torch.cumsum(
            torch.bincount(sorted_experts, minlength=E), dim=0
        ).tolist()

        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        start = 0
        for expert_idx, end in enumerate(boundaries):
            if end == start:
                continue
            expert_output = self.routed_experts[expert_idx](flat_tokens[start:end])[
                "hidden_states"
            ]
            output.index_add_(
                0,
                order[start:end] // K,
                expert_output * flat_weights[start:end],
            )
            start = end

        return {
            "hidden_states": output,
            "aux_loss": aux_loss,
            "router_stats": router_stats,
        }
