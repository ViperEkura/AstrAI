from typing import List, Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.factory import BaseFactory
from astrai.model.components.linear import Linear


class FFNFactory(BaseFactory[nn.Module]):
    pass


class FFNOutput(TypedDict):
    hidden_states: Tensor
    aux_loss: Optional[Tensor]


class RoutedOutput(TypedDict):
    hidden_states: Tensor
    aux_loss: Optional[Tensor]


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
        return {"hidden_states": out, "aux_loss": None}


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
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.topk_method = topk_method
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
        self._router_probs: Optional[Tensor] = None
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
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        shared_out = self._shared_forward(x_flat)
        routed_output = self._routed_forward(x_flat, include_aux_loss)

        out = (shared_out + routed_output["hidden_states"]).view(bsz, seq_len, dim)
        return {"hidden_states": out, "aux_loss": routed_output["aux_loss"]}

    def _shared_forward(self, x: Tensor) -> Tensor:
        if self.n_shared_experts == 0:
            return torch.zeros_like(x)
        return (
            sum(e(x)["hidden_states"] for e in self.shared_experts)
            / self.n_shared_experts
        )

    def _routed_forward(self, x: Tensor, include_aux_loss: bool) -> RoutedOutput:
        N, D = x.shape
        K = self.n_activated_experts

        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits.float(), dim=-1).to(x.dtype)
        self._router_probs = router_probs.detach()

        topk_weights, topk_indices = torch.topk(router_probs, K, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        aux_loss = None
        if include_aux_loss:
            expert_load = F.one_hot(
                topk_indices, num_classes=self.n_routed_experts
            ).float()
            expert_load = expert_load.mean(dim=(0, 1))
            router_prob = router_probs.float().mean(dim=0)
            aux_loss = self.n_routed_experts * (expert_load * router_prob).sum()

        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        for expert_idx in range(self.n_routed_experts):
            expert_mask = topk_indices == expert_idx
            token_idx, k_idx = expert_mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert = self.routed_experts[expert_idx]
            expert_input = x[token_idx]
            expert_output = expert(expert_input)["hidden_states"]

            weights = topk_weights[token_idx, k_idx].unsqueeze(-1)
            output.index_add_(0, token_idx, expert_output * weights)

        return {"hidden_states": output, "aux_loss": aux_loss}

    @staticmethod
    def collect_router_probs(module: nn.Module) -> List[Tensor]:
        """Recursively collect router_probs from all DeepSeekMoE submodules."""
        probs: List[Tensor] = []
        for m in module.modules():
            if isinstance(m, DeepSeekMoE) and m._router_probs is not None:
                probs.append(m._router_probs)
        return probs
