import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.factory import BaseFactory
from astrai.model.components.linear import Linear


class FFNFactory(BaseFactory[nn.Module]):
    pass


@FFNFactory.register("mlp")
class MLP(nn.Module):
    def __init__(self, dim: int, dim_ffn: int, down_init_std: float = 0.02):
        super().__init__()
        self.up = Linear(dim, dim_ffn)
        self.gate = Linear(dim, dim_ffn)
        self.down = Linear(dim_ffn, dim, init_std=down_init_std)

    def forward(self, x: Tensor) -> Tensor:
        gated = self.up(x) * F.silu(self.gate(x))
        out = self.down(gated)
        return out


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
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.topk_method = topk_method or "greedy"

        if self.topk_method != "greedy":
            raise ValueError(
                f"Unsupported MoE top-k method: {self.topk_method!r}"
            )
        if not 0 < n_activated_experts <= n_routed_experts:
            raise ValueError(
                "n_activated_experts must be in [1, n_routed_experts]"
            )

        self.router = Linear(dim, n_routed_experts, bias=False)
        moe_scale = 1 / max(n_shared_experts, 1) + 1 / n_activated_experts
        down_init_std = 0.02 / (2 * n_layers * moe_scale) ** 0.5

        self.shared_experts = nn.ModuleList(
            [
                MLP(dim, dim_ffn, down_init_std=down_init_std)
                for _ in range(n_shared_experts)
            ]
        )
        self.routed_experts = nn.ModuleList(
            [
                MLP(dim, dim_ffn, down_init_std=down_init_std)
                for _ in range(n_routed_experts)
            ]
        )

    def forward(self, x: Tensor):
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        shared_out = self._shared_forward(x_flat)
        routed_out, aux_loss, z_loss, expert_load, router_entropy = (
            self._routed_forward(x_flat)
        )

        out = (shared_out + routed_out).view(bsz, seq_len, dim)
        return out, aux_loss, z_loss, expert_load, router_entropy

    def _shared_forward(self, x: Tensor) -> Tensor:
        if self.n_shared_experts == 0:
            return torch.zeros_like(x)
        return sum(e(x) for e in self.shared_experts) / self.n_shared_experts

    def _routed_forward(self, x: Tensor):
        N, D = x.shape
        K = self.n_activated_experts

        router_logits = self.router(x)
        router_logits_fp32 = router_logits.float()
        router_probs_fp32 = torch.softmax(router_logits_fp32, dim=-1)
        router_probs = router_probs_fp32.to(x.dtype)

        topk_weights, topk_indices = torch.topk(router_probs, K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Switch-style differentiable load-balancing loss.  Expert load is
        # measured over all top-k assignments, so a uniform router has loss 1.
        assignments = F.one_hot(
            topk_indices, num_classes=self.n_routed_experts
        ).float()
        expert_load = assignments.mean(dim=(0, 1))
        mean_router_prob = router_probs_fp32.mean(dim=0)
        aux_loss = self.n_routed_experts * torch.sum(
            expert_load * mean_router_prob
        )
        z_loss = torch.logsumexp(router_logits_fp32, dim=-1).square().mean()
        router_entropy = -torch.sum(
            router_probs_fp32
            * torch.log(router_probs_fp32.clamp_min(torch.finfo(torch.float32).tiny)),
            dim=-1,
        ).mean()

        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        for expert_idx in range(self.n_routed_experts):
            expert_mask = topk_indices == expert_idx
            token_idx, k_idx = expert_mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_input = x[token_idx]
            expert_output = self.routed_experts[expert_idx](expert_input)
            weights = topk_weights[token_idx, k_idx].unsqueeze(-1)
            output.index_add_(0, token_idx, expert_output * weights)

        return (
            output,
            aux_loss,
            z_loss,
            expert_load.detach(),
            router_entropy.detach(),
        )
