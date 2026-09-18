import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class SimParams:
    rho: float
    sigma: float
    k_exh: float
    kappa: float
    k1: float
    k2: float
    delta_a: float
    delta_pe: float
    delta_ee: float
    delta_te: float
    growth: float
    t_max: float
    b_max: float
    eta_pe: float


def derivatives(state: torch.Tensor, params: SimParams) -> torch.Tensor:
    activated = state[..., 0]
    pre = state[..., 1]
    early = state[..., 2]
    terminal = state[..., 3]
    tumour = state[..., 4]

    expansion = params.rho * params.sigma * activated * (1.0 - activated / params.t_max)
    d_activated = expansion - params.delta_a * activated - params.k_exh * activated
    d_pre = params.k_exh * activated - params.k1 * pre - params.delta_pe * pre
    d_early = params.k1 * pre - params.k2 * early - params.delta_ee * early
    d_terminal = params.k2 * early - params.delta_te * terminal

    cytotoxic = activated + params.eta_pe * pre
    d_tumour = (
        params.growth * tumour * (1.0 - tumour / params.b_max) - params.kappa * cytotoxic * tumour
    )

    return torch.stack([d_activated, d_pre, d_early, d_terminal, d_tumour], dim=-1)
