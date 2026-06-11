import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, embedding_dim)))
        self.neftune_noise_alpha = 0.0

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        out = F.embedding(x, self.weight)
        if self.training and self.neftune_noise_alpha > 0.0:
            eps = self.neftune_noise_alpha / math.sqrt(out.size(1))
            out = out + eps * torch.randn_like(out)
        return out
