import dataclasses

import torch

from rlcardes.quarry.environment import ConstructEnv
from rlcardes.stoop.policy import ActorCritic


@dataclasses.dataclass
class Rollout:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    last_value: torch.Tensor


def generalised_advantage(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1])
    next_value = last_value
    for t in range(n_steps - 1, -1, -1):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def collect(envs: list[ConstructEnv], policy: ActorCritic, n_steps: int) -> Rollout:
    states = torch.stack([env.reset() for env in envs], dim=0)
    state_log: list[torch.Tensor] = []
    action_log: list[torch.Tensor] = []
    logprob_log: list[torch.Tensor] = []
    reward_log: list[torch.Tensor] = []
    value_log: list[torch.Tensor] = []
    done_log: list[torch.Tensor] = []

    for _ in range(n_steps):
        with torch.no_grad():
            step = policy.act(states)
        rewards = torch.empty(len(envs))
        dones = torch.empty(len(envs))
        next_states: list[torch.Tensor] = []
        for i, env in enumerate(envs):
            result = env.step(int(step.action[i].item()))
            rewards[i] = result.reward
            dones[i] = 1.0 if result.done else 0.0
            next_states.append(env.reset() if result.done else result.state)

        state_log.append(states)
        action_log.append(step.action)
        logprob_log.append(step.log_prob)
        value_log.append(step.value)
        reward_log.append(rewards)
        done_log.append(dones)
        states = torch.stack(next_states, dim=0)

    with torch.no_grad():
        last_value = policy.value(states)

    return Rollout(
        states=torch.stack(state_log, dim=0),
        actions=torch.stack(action_log, dim=0),
        log_probs=torch.stack(logprob_log, dim=0),
        rewards=torch.stack(reward_log, dim=0),
        values=torch.stack(value_log, dim=0),
        dones=torch.stack(done_log, dim=0),
        last_value=last_value,
    )
