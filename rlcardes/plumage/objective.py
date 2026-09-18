import torch
import torch.nn.functional as F

from rlcardes.plumage.score import PhenoOutput


def phenotype_loss(
    output: PhenoOutput,
    subpop: torch.Tensor,
    reference: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    classification = F.cross_entropy(output.subpop_logits, subpop)
    regression = F.mse_loss(output.exhaustion, reference)
    return classification + beta * regression
