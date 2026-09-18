from collections.abc import Callable

import torch

Field = Callable[[torch.Tensor], torch.Tensor]


def rk4_step(field: Field, state: torch.Tensor, dt: float) -> torch.Tensor:
    k1 = field(state)
    k2 = field(state + 0.5 * dt * k1)
    k3 = field(state + 0.5 * dt * k2)
    k4 = field(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(field: Field, state0: torch.Tensor, dt: float, horizon: float) -> torch.Tensor:
    steps = int(round(horizon / dt))
    state = state0
    for _ in range(steps):
        state = rk4_step(field, state, dt)
    return state
