import dataclasses

import torch

from rlcardes.creance.loop import IterationReport, train_policy
from rlcardes.jesses.config import Experiment
from rlcardes.lure.reward import RewardComposer
from rlcardes.mantling.persistnet import PersistNet, persistence_loss
from rlcardes.plumage.objective import phenotype_loss
from rlcardes.plumage.score import PhenoBatch, ScPhenoScore
from rlcardes.quarry.environment import ConstructEnv
from rlcardes.stoop.actions import ConstructCodec
from rlcardes.stoop.policy import ActorCritic


@dataclasses.dataclass(frozen=True)
class Models:
    phenotype: ScPhenoScore
    codec: ConstructCodec
    persistnet: PersistNet
    policy: ActorCritic


def build_models(experiment: Experiment) -> Models:
    return Models(
        phenotype=ScPhenoScore(experiment.encoder),
        codec=ConstructCodec(experiment.policy.n_actions, experiment.policy.construct_dim),
        persistnet=PersistNet(experiment.persist),
        policy=ActorCritic(
            experiment.policy.state_dim,
            experiment.policy.hidden,
            experiment.policy.n_actions,
        ),
    )


def fit_phenotype(
    model: ScPhenoScore, batch: PhenoBatch, experiment: Experiment, steps: int
) -> list[float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment.optim.lr,
        weight_decay=experiment.optim.weight_decay,
    )
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        output = model(batch.features, batch.markers)
        loss = phenotype_loss(output, batch.subpop, batch.reference, experiment.optim.beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def fit_persistence(
    model: PersistNet,
    phenotype: torch.Tensor,
    construct: torch.Tensor,
    label: torch.Tensor,
    experiment: Experiment,
    steps: int,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.optim.lr)
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        probability = model(phenotype, construct)
        loss = persistence_loss(probability, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def population_summary(model: ScPhenoScore, batch: PhenoBatch) -> tuple[torch.Tensor, float]:
    model.eval()
    with torch.no_grad():
        latent = model.embed(batch.features).mean(dim=0)
    fitness = 1.0 - float(batch.exhausted.float().mean().item())
    return latent, fitness


def build_envs(
    experiment: Experiment,
    phenotype: torch.Tensor,
    fitness: float,
    persistnet: PersistNet,
    codec: ConstructCodec,
    n_envs: int,
    episode_len: int,
) -> list[ConstructEnv]:
    composer = RewardComposer(persistnet, codec, experiment.optim.reward_weights)
    reward_fn = composer.reward_fn()
    return [
        ConstructEnv(
            phenotype=phenotype,
            fitness=fitness,
            spec=experiment.simulator,
            embed=codec.at,
            reward_fn=reward_fn,
            episode_len=episode_len,
            n_actions=experiment.policy.n_actions,
        )
        for _ in range(n_envs)
    ]


def run_optimisation(
    experiment: Experiment,
    envs: list[ConstructEnv],
    policy: ActorCritic,
    iterations: int,
    steps_per_iter: int,
) -> list[IterationReport]:
    return train_policy(envs, policy, experiment, iterations, steps_per_iter)
