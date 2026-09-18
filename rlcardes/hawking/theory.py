import dataclasses
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class TheoremGap:
    value_conditioned: float
    value_blind: float
    gap: float
    lower_bound: float
    holds: bool


def theorem1_gap(reward_matrix: np.ndarray) -> TheoremGap:
    per_phenotype_best = reward_matrix.max(axis=1)
    value_conditioned = float(per_phenotype_best.mean())

    blind_values = reward_matrix.mean(axis=0)
    blind_action = int(blind_values.argmax())
    value_blind = float(blind_values[blind_action])

    conditioned_action = reward_matrix.argmax(axis=1)
    differs = conditioned_action != blind_action
    probability = float(differs.mean())
    if differs.any():
        margin = per_phenotype_best[differs] - reward_matrix[differs, blind_action]
        delta_min = float(margin.min())
    else:
        delta_min = 0.0

    gap = value_conditioned - value_blind
    lower_bound = probability * delta_min
    return TheoremGap(
        value_conditioned=value_conditioned,
        value_blind=value_blind,
        gap=gap,
        lower_bound=lower_bound,
        holds=gap >= lower_bound - 1e-9,
    )


def proposition1_bound(n: int, n_actions: int, phenotype_dim: int, delta: float = 0.05) -> float:
    return math.sqrt(n_actions * phenotype_dim * math.log(1.0 / delta) / n)
