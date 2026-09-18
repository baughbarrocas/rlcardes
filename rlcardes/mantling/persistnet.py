import torch
import torch.nn.functional as F
from torch import nn

from rlcardes.jesses.config import PersistSpec
from rlcardes.mantling.attention import CrossAttention


class PersistNet(nn.Module):
    def __init__(self, spec: PersistSpec) -> None:
        super().__init__()
        self.fuse = CrossAttention(spec.phenotype_dim, spec.construct_dim, spec.key_dim)
        widths = [spec.key_dim, *spec.hidden]
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(spec.dropout))
        self.head = nn.Sequential(*layers)
        self.readout = nn.Linear(widths[-1], 1)

    def forward(self, phenotype: torch.Tensor, construct: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(phenotype, construct)
        logit = self.readout(self.head(fused)).squeeze(-1)
        return torch.sigmoid(logit)


def persistence_loss(probability: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(probability, label)
