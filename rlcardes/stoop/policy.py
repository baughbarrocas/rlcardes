import dataclasses
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical


@dataclasses.dataclass(frozen=True)
class PolicyStep:
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


def _trunk(in_dim: int, hidden: Sequence[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    width = in_dim
    for h in hidden:
        layers.append(nn.Linear(width, h))
        layers.append(nn.Tanh())
        width = h
    return nn.Sequential(*layers), width


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: Sequence[int], n_actions: int) -> None:
        super().__init__()
        self.actor_body, actor_out = _trunk(state_dim, hidden)
        self.actor_head = nn.Linear(actor_out, n_actions)
        self.critic_body, critic_out = _trunk(state_dim, hidden)
        self.critic_head = nn.Linear(critic_out, 1)

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor_head(self.actor_body(state))

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic_head(self.critic_body(state)).squeeze(-1)

    def act(self, state: torch.Tensor) -> PolicyStep:
        dist = Categorical(logits=self.logits(state))
        action = dist.sample()
        return PolicyStep(action=action, log_prob=dist.log_prob(action), value=self.value(state))

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=self.logits(state))
        return dist.log_prob(action), dist.entropy(), self.value(state)
