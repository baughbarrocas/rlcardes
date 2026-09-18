from collections.abc import Callable

import torch

from rlcardes.mantling.persistnet import PersistNet
from rlcardes.mews.library import ITAM, Construct
from rlcardes.stoop.actions import ConstructCodec

RewardFn = Callable[[Construct, torch.Tensor, float], float]


def construct_cost(construct: Construct) -> float:
    itam_load = (ITAM[construct.itam] - 1) / (len(ITAM) - 1)
    hinge_load = construct.hinge / (3 - 1)
    return float(0.7 * itam_load + 0.3 * hinge_load)


class RewardComposer:
    def __init__(
        self,
        persistnet: PersistNet,
        codec: ConstructCodec,
        weights: tuple[float, float, float],
    ) -> None:
        self.persistnet = persistnet
        self.codec = codec
        self.alpha = weights

    def persistence(self, phenotype: torch.Tensor, action: int) -> float:
        with torch.no_grad():
            z = self.codec.at(action).unsqueeze(0)
            prob = self.persistnet(phenotype.unsqueeze(0), z)
        return float(prob.item())

    def reward_fn(self) -> RewardFn:
        def compose(construct: Construct, phenotype: torch.Tensor, cyto: float) -> float:
            persistence = self.persistence(phenotype, construct.index)
            cost = construct_cost(construct)
            return self.alpha[0] * cyto + self.alpha[1] * persistence - self.alpha[2] * cost

        return compose
