import torch
from torch import nn


class ConstructCodec(nn.Module):
    def __init__(self, n_actions: int, construct_dim: int) -> None:
        super().__init__()
        self.table = nn.Embedding(n_actions, construct_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.table(action)

    def at(self, index: int) -> torch.Tensor:
        return self.table(torch.tensor(index, dtype=torch.long))
