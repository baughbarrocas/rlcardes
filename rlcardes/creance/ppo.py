import dataclasses

import torch
from torch import nn

from rlcardes.jesses.config import PPOSpec
from rlcardes.stoop.policy import ActorCritic


@dataclasses.dataclass(frozen=True)
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float


def _flatten(rollout_field: torch.Tensor) -> torch.Tensor:
    return rollout_field.reshape(-1, *rollout_field.shape[2:])


def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    spec: PPOSpec,
) -> UpdateStats:
    states = _flatten(states)
    actions = _flatten(actions)
    old_log_probs = _flatten(old_log_probs)
    returns = _flatten(returns)
    advantages = _flatten(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = states.shape[0]
    minibatch = min(spec.minibatch, n)
    last = UpdateStats(0.0, 0.0, 0.0)
    for _ in range(spec.update_epochs):
        order = torch.randperm(n)
        for start in range(0, n, minibatch):
            idx = order[start : start + minibatch]
            log_prob, entropy, value = policy.evaluate(states[idx], actions[idx])
            ratio = torch.exp(log_prob - old_log_probs[idx])
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - spec.clip, 1.0 + spec.clip) * advantages[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (returns[idx] - value).pow(2).mean()
            entropy_term = entropy.mean()
            loss = policy_loss + spec.value_coef * value_loss - spec.entropy_coef * entropy_term

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), spec.max_grad_norm)
            optimizer.step()
            last = UpdateStats(
                policy_loss=float(policy_loss.item()),
                value_loss=float(value_loss.item()),
                entropy=float(entropy_term.item()),
            )
    return last
