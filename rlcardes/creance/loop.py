import dataclasses

import torch

from rlcardes.creance.buffer import collect, generalised_advantage
from rlcardes.creance.ppo import ppo_update
from rlcardes.jesses.config import Experiment
from rlcardes.quarry.environment import ConstructEnv
from rlcardes.stoop.policy import ActorCritic


@dataclasses.dataclass(frozen=True)
class IterationReport:
    mean_reward: float
    policy_loss: float
    entropy: float


def train_policy(
    envs: list[ConstructEnv],
    policy: ActorCritic,
    experiment: Experiment,
    iterations: int,
    steps_per_iter: int,
) -> list[IterationReport]:
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=experiment.optim.lr,
        weight_decay=experiment.optim.weight_decay,
    )
    history: list[IterationReport] = []
    for _ in range(iterations):
        rollout = collect(envs, policy, steps_per_iter)
        advantages, returns = generalised_advantage(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            rollout.last_value,
            experiment.ppo.gamma,
            experiment.ppo.gae_lambda,
        )
        stats = ppo_update(
            policy,
            optimizer,
            rollout.states,
            rollout.actions,
            rollout.log_probs,
            returns,
            advantages,
            experiment.ppo,
        )
        history.append(
            IterationReport(
                mean_reward=float(rollout.rewards.mean().item()),
                policy_loss=stats.policy_loss,
                entropy=stats.entropy,
            )
        )
    return history
