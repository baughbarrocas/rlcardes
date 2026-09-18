import dataclasses
from collections.abc import Callable

import torch

from rlcardes.jesses.config import SimulatorSpec
from rlcardes.mews.library import Construct, decode
from rlcardes.quarry.integrator import integrate
from rlcardes.quarry.kinetics import derivatives
from rlcardes.quarry.mapping import map_construct

ACTIVATED_SEED: float = 1e6
TUMOUR_SEED: float = 1e9

RewardFn = Callable[[Construct, torch.Tensor, float], float]
Embedder = Callable[[int], torch.Tensor]


@dataclasses.dataclass(frozen=True)
class StepResult:
    state: torch.Tensor
    reward: float
    cytotoxicity: float
    done: bool
    construct: Construct


def simulate_cytotoxicity(construct: Construct, fitness: float, spec: SimulatorSpec) -> float:
    params = map_construct(construct, spec)
    state0 = torch.tensor(
        [ACTIVATED_SEED * fitness, 0.0, 0.0, 0.0, TUMOUR_SEED], dtype=torch.float64
    )
    final = integrate(lambda s: derivatives(s, params), state0, spec.dt, spec.horizon_days)
    ratio = float(final[4] / state0[4])
    return float(min(max(1.0 - ratio, 0.0), 1.0))


class ConstructEnv:
    def __init__(
        self,
        phenotype: torch.Tensor,
        fitness: float,
        spec: SimulatorSpec,
        embed: Embedder,
        reward_fn: RewardFn,
        episode_len: int,
        n_actions: int = 45,
    ) -> None:
        self.phenotype = phenotype.detach()
        self.fitness = fitness
        self.spec = spec
        self.embed = embed
        self.reward_fn = reward_fn
        self.episode_len = episode_len
        self.n_actions = n_actions
        self.current = 0
        self.elapsed = 0

    def _state(self) -> torch.Tensor:
        z = self.embed(self.current).detach()
        return torch.cat([self.phenotype, z], dim=-1)

    def reset(self) -> torch.Tensor:
        self.current = 0
        self.elapsed = 0
        return self._state()

    def step(self, action: int) -> StepResult:
        construct = decode(action)
        self.current = action
        self.elapsed += 1
        cyto = simulate_cytotoxicity(construct, self.fitness, self.spec)
        reward = self.reward_fn(construct, self.phenotype, cyto)
        done = self.elapsed >= self.episode_len
        return StepResult(
            state=self._state(),
            reward=reward,
            cytotoxicity=cyto,
            done=done,
            construct=construct,
        )
