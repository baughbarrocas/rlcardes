import dataclasses

import numpy as np
import torch

EXHAUSTED_SUBPOPS: tuple[int, ...] = (5, 6, 7)
MARKER_INDEX: tuple[int, int, int, int, int] = (0, 1, 2, 3, 4)
BETA0: float = 0.7


@dataclasses.dataclass(frozen=True)
class Cohort:
    counts: torch.Tensor
    subpop: torch.Tensor
    reference: torch.Tensor
    exhausted: torch.Tensor

    def __len__(self) -> int:
        return int(self.counts.shape[0])


@dataclasses.dataclass(frozen=True)
class Split:
    train: Cohort
    val: Cohort
    test: Cohort


def _reference_score(counts: np.ndarray) -> np.ndarray:
    markers = counts[:, list(MARKER_INDEX)]
    normed = np.log1p(markers / (counts.sum(axis=1, keepdims=True) + 1.0) * 1e4)
    pooled = normed.mean(axis=1)
    return 1.0 / (1.0 + np.exp(-(pooled - BETA0)))


def planted_cohort(
    n_cells: int,
    n_genes: int = 3000,
    n_subpops: int = 8,
    seed: int = 0,
) -> Cohort:
    rng = np.random.default_rng(seed)
    subpop = rng.integers(0, n_subpops, size=n_cells)
    exhausted = np.isin(subpop, EXHAUSTED_SUBPOPS).astype(np.int64)

    axis = exhausted.astype(np.float64) + rng.normal(0.0, 0.25, size=n_cells)
    loadings = rng.normal(0.0, 1.0, size=n_genes)
    program = np.outer(axis, loadings)

    subpop_centers = rng.normal(0.0, 0.6, size=(n_subpops, n_genes))
    base = subpop_centers[subpop] + program + rng.normal(0.0, 0.4, size=(n_cells, n_genes))
    rate = np.exp(base - base.mean(axis=0, keepdims=True))

    counts = rng.poisson(rate * 3.0).astype(np.float32)
    marker_boost = (exhausted[:, None] * rng.uniform(6.0, 12.0, size=(1, 5))).astype(np.float32)
    counts[:, list(MARKER_INDEX)] += marker_boost + rng.poisson(2.0, size=(n_cells, 5)).astype(
        np.float32
    )

    reference = _reference_score(counts.astype(np.float64)).astype(np.float32)
    return Cohort(
        counts=torch.from_numpy(counts),
        subpop=torch.from_numpy(subpop.astype(np.int64)),
        reference=torch.from_numpy(reference),
        exhausted=torch.from_numpy(exhausted),
    )


def _slice(cohort: Cohort, lo: int, hi: int) -> Cohort:
    return Cohort(
        counts=cohort.counts[lo:hi].clone(),
        subpop=cohort.subpop[lo:hi].clone(),
        reference=cohort.reference[lo:hi].clone(),
        exhausted=cohort.exhausted[lo:hi].clone(),
    )


def split_cohort(cohort: Cohort, fractions: tuple[float, float, float]) -> Split:
    total = len(cohort)
    n_train = int(total * fractions[0])
    n_val = int(total * fractions[1])
    return Split(
        train=_slice(cohort, 0, n_train),
        val=_slice(cohort, n_train, n_train + n_val),
        test=_slice(cohort, n_train + n_val, total),
    )
