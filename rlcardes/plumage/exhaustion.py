import torch
from torch import nn


class ExhaustionScorer(nn.Module):
    def __init__(self, n_markers: int = 5, beta0: float = 0.7) -> None:
        super().__init__()
        self.marker_weights = nn.Parameter(torch.full((n_markers,), 1.0 / n_markers))
        self.beta0 = beta0

    def forward(self, markers: torch.Tensor) -> torch.Tensor:
        logged = torch.log1p(markers)
        pooled = logged @ self.marker_weights
        return torch.sigmoid(pooled - self.beta0)
