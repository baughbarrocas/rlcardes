from collections.abc import Sequence

import torch
from torch import nn


class PhenotypeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(hidden[-1]) != latent_dim:
            raise ValueError("final hidden width must equal latent_dim")
        widths = [input_dim, *hidden]
        blocks: list[nn.Module] = []
        for i in range(len(widths) - 1):
            blocks.append(nn.Linear(widths[i], widths[i + 1]))
            blocks.append(nn.LayerNorm(widths[i + 1]))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)
