import argparse
import os

import torch

from rlcardes.falconer.assembly import (
    build_envs,
    build_models,
    fit_persistence,
    fit_phenotype,
    population_summary,
    run_optimisation,
)
from rlcardes.hawking.appraise import evaluate_persistence, evaluate_phenotype, iteration_reduction
from rlcardes.hawking.theory import theorem1_gap
from rlcardes.jesses.config import load_experiment
from rlcardes.jesses.ledger import get_logger, save_checkpoint
from rlcardes.jesses.seed import set_seed
from rlcardes.mews.cohort import planted_cohort, split_cohort
from rlcardes.mews.library import decode, persistence_table
from rlcardes.plumage.score import PhenoBatch, prepare
from rlcardes.quarry.environment import simulate_cytotoxicity

_LOG = get_logger("rlcardes.falconer")


def _cohort_batch(
    cells: int, input_dim: int, n_subpops: int, seed: int
) -> tuple[PhenoBatch, PhenoBatch]:
    cohort = planted_cohort(cells, n_genes=input_dim, n_subpops=n_subpops, seed=seed)
    split = split_cohort(cohort, (0.7, 0.1, 0.2))
    return prepare(split.train), prepare(split.test)


def cmd_train(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.recipe)
    set_seed(experiment.seed)
    models = build_models(experiment)

    train_batch, _ = _cohort_batch(
        args.cells, experiment.encoder.input_dim, experiment.encoder.n_subpops, experiment.seed
    )
    phen_losses = fit_phenotype(models.phenotype, train_batch, experiment, args.epochs)
    _LOG.info("phenotype loss %.4f -> %.4f", phen_losses[0], phen_losses[-1])

    phenotype, construct, label = persistence_table(
        args.records,
        phenotype_dim=experiment.persist.phenotype_dim,
        construct_dim=experiment.persist.construct_dim,
        seed=experiment.seed,
    )
    persist_losses = fit_persistence(
        models.persistnet, phenotype, construct, label, experiment, args.epochs
    )
    _LOG.info("persistence loss %.4f -> %.4f", persist_losses[0], persist_losses[-1])

    summary, fitness = population_summary(models.phenotype, train_batch)
    envs = build_envs(
        experiment, summary, fitness, models.persistnet, models.codec, args.envs, args.episode
    )
    history = run_optimisation(experiment, envs, models.policy, args.iterations, args.rollout)
    _LOG.info("policy reward %.4f -> %.4f", history[0].mean_reward, history[-1].mean_reward)

    os.makedirs(args.out, exist_ok=True)
    save_checkpoint(
        {
            "seed": experiment.seed,
            "phenotype": models.phenotype.state_dict(),
            "persistnet": models.persistnet.state_dict(),
            "codec": models.codec.state_dict(),
            "policy": models.policy.state_dict(),
        },
        os.path.join(args.out, "mews.pt"),
    )
    return 0


def cmd_hunt(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.recipe)
    set_seed(experiment.seed)
    models = build_models(experiment)

    train_batch, _ = _cohort_batch(
        args.cells, experiment.encoder.input_dim, experiment.encoder.n_subpops, experiment.seed
    )
    fit_phenotype(models.phenotype, train_batch, experiment, args.warmup)
    summary, fitness = population_summary(models.phenotype, train_batch)
    envs = build_envs(
        experiment, summary, fitness, models.persistnet, models.codec, args.envs, args.episode
    )
    run_optimisation(experiment, envs, models.policy, args.iterations, args.rollout)

    state = envs[0].reset()
    with torch.no_grad():
        best = int(models.policy.logits(state.unsqueeze(0)).squeeze(0).argmax().item())
    construct = decode(best)
    cyto = simulate_cytotoxicity(construct, fitness, experiment.simulator)
    _LOG.info("recommended construct %s cytotoxicity %.3f", construct.label, cyto)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.recipe)
    set_seed(experiment.seed)
    models = build_models(experiment)

    train_batch, test_batch = _cohort_batch(
        args.cells, experiment.encoder.input_dim, experiment.encoder.n_subpops, experiment.seed
    )
    fit_phenotype(models.phenotype, train_batch, experiment, args.epochs)
    phen = evaluate_phenotype(models.phenotype, test_batch)
    _LOG.info(
        "exhaustion_auc=%.3f subpop_f1=%.3f mr=%.2f",
        phen["exhaustion_auc"],
        phen["subpop_f1"],
        phen["misclassification_rate"],
    )

    phenotype, construct, label = persistence_table(
        args.records,
        phenotype_dim=experiment.persist.phenotype_dim,
        construct_dim=experiment.persist.construct_dim,
        seed=experiment.seed + 1,
    )
    fit_persistence(models.persistnet, phenotype, construct, label, experiment, args.epochs)
    persist = evaluate_persistence(models.persistnet, phenotype, construct, label)
    _LOG.info("persistence_acc=%.2f auc=%.3f", persist["accuracy"], persist["auc"])

    summary, fitness = population_summary(models.phenotype, train_batch)
    envs = build_envs(
        experiment, summary, fitness, models.persistnet, models.codec, args.envs, args.episode
    )
    run_optimisation(experiment, envs, models.policy, args.iterations, args.rollout)
    env = envs[0]

    def reward_of(action: int) -> float:
        return env.step(action).reward

    reduction = iteration_reduction(
        reward_of, models.policy, env.reset(), experiment.policy.n_actions
    )
    _LOG.info("iteration_reduction=%.1f%%", reduction["iteration_reduction"])

    matrix = torch.rand(64, experiment.policy.n_actions).numpy()
    gap = theorem1_gap(matrix)
    _LOG.info("theorem1 gap=%.4f bound=%.4f holds=%s", gap.gap, gap.lower_bound, gap.holds)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.recipe)
    set_seed(experiment.seed)
    models = build_models(experiment)
    models.persistnet.eval()
    phenotype = torch.randn(1, experiment.persist.phenotype_dim)
    construct = torch.randn(1, experiment.persist.construct_dim)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        models.persistnet,
        (phenotype, construct),
        args.out,
        input_names=["phenotype", "construct"],
        output_names=["persistence"],
        opset_version=17,
        dynamo=False,
    )
    _LOG.info("exported persistnet to %s", args.out)
    return 0
