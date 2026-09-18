import dataclasses

import torch
from torch import nn

from rlcardes.jesses.config import EncoderSpec
from rlcardes.mews.cohort import MARKER_INDEX, Cohort
from rlcardes.plumage.encoder import PhenotypeEncoder
from rlcardes.plumage.exhaustion import ExhaustionScorer


@dataclasses.dataclass(frozen=True)
class PhenoBatch:
    features: torch.Tensor
    markers: torch.Tensor
    subpop: torch.Tensor
    reference: torch.Tensor
    exhausted: torch.Tensor


@dataclasses.dataclass(frozen=True)
class PhenoOutput:
    latent: torch.Tensor
    subpop_logits: torch.Tensor
    exhaustion: torch.Tensor


def prepare(cohort: Cohort, target_sum: float = 1e4) -> PhenoBatch:
    counts = cohort.counts
    library = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    normed = counts / library * target_sum
    features = torch.log1p(normed)
    markers = normed[:, list(MARKER_INDEX)]
    return PhenoBatch(
        features=features,
        markers=markers,
        subpop=cohort.subpop,
        reference=cohort.reference,
        exhausted=cohort.exhausted,
    )


class ScPhenoScore(nn.Module):
    def __init__(self, spec: EncoderSpec, n_markers: int = 5) -> None:
        super().__init__()
        self.encoder = PhenotypeEncoder(
            input_dim=spec.input_dim,
            hidden=spec.hidden,
            latent_dim=spec.latent_dim,
            dropout=spec.dropout,
        )
        self.subpop_head = nn.Linear(spec.latent_dim, spec.n_subpops)
        self.scorer = ExhaustionScorer(n_markers=n_markers)

    def forward(self, features: torch.Tensor, markers: torch.Tensor) -> PhenoOutput:
        latent = self.encoder(features)
        return PhenoOutput(
            latent=latent,
            subpop_logits=self.subpop_head(latent),
            exhaustion=self.scorer(markers),
        )

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)
