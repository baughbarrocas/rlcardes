from collections.abc import Callable

import numpy as np
import torch

from rlcardes.hawking.metrics import macro_f1, misclassification_rate, roc_auc
from rlcardes.mantling.persistnet import PersistNet
from rlcardes.plumage.score import PhenoBatch, ScPhenoScore
from rlcardes.stoop.policy import ActorCritic


def evaluate_phenotype(model: ScPhenoScore, batch: PhenoBatch) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        output = model(batch.features, batch.markers)
        prediction = output.subpop_logits.argmax(dim=1).cpu().numpy()
        exhaustion = output.exhaustion.cpu().numpy()
    target = batch.subpop.cpu().numpy()
    labels = batch.exhausted.cpu().numpy()
    return {
        "exhaustion_auc": roc_auc(exhaustion, labels),
        "subpop_f1": macro_f1(prediction, target, int(target.max()) + 1),
        "misclassification_rate": misclassification_rate(prediction, target),
    }


def evaluate_persistence(
    model: PersistNet,
    phenotype: torch.Tensor,
    construct: torch.Tensor,
    label: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        probability = model(phenotype, construct).cpu().numpy()
    labels = label.cpu().numpy()
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "accuracy": float((prediction == labels).mean()) * 100.0,
        "auc": roc_auc(probability, labels.astype(np.int64)),
        "f1": macro_f1(prediction, labels.astype(np.int64), 2),
    }


def iteration_reduction(
    reward_of: Callable[[int], float],
    policy: ActorCritic,
    state: torch.Tensor,
    n_actions: int,
) -> dict[str, float]:
    rewards = np.array([reward_of(a) for a in range(n_actions)])
    target = float(rewards.max())

    grid_iters = int(np.argmax(rewards >= target - 1e-12)) + 1

    with torch.no_grad():
        logits = policy.logits(state.unsqueeze(0)).squeeze(0).cpu().numpy()
    ranking = np.argsort(-logits)
    reached = np.argmax(rewards[ranking] >= target - 1e-12)
    rl_iters = int(reached) + 1

    reduction = (grid_iters - rl_iters) / grid_iters * 100.0
    return {
        "grid_iterations": float(grid_iters),
        "policy_iterations": float(rl_iters),
        "iteration_reduction": float(reduction),
    }
