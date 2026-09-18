import math

import torch
import torch.nn.functional as F
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, phenotype_dim: int, construct_dim: int, key_dim: int) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.query = nn.Linear(phenotype_dim, key_dim, bias=False)
        self.key = nn.Linear(construct_dim, key_dim, bias=False)
        self.value = nn.Linear(construct_dim, key_dim, bias=False)

    def forward(self, phenotype: torch.Tensor, construct: torch.Tensor) -> torch.Tensor:
        q = self.query(phenotype)
        k = self.key(construct)
        v = self.value(construct)
        scores = torch.einsum("bi,bj->bij", q, k) / math.sqrt(self.key_dim)
        attn = F.softmax(scores, dim=-1)
        return torch.einsum("bij,bj->bi", attn, v)
