import dataclasses

import numpy as np
import torch

COSTIM: tuple[str, ...] = ("CD28", "4-1BB", "ICOS", "OX40", "CD27")
ITAM: tuple[int, ...] = (1, 2, 3)
HINGE: tuple[str, ...] = ("IgG4", "CD8a", "CD28")


@dataclasses.dataclass(frozen=True)
class Construct:
    index: int
    costim: int
    itam: int
    hinge: int

    @property
    def label(self) -> str:
        return f"{COSTIM[self.costim]}-{ITAM[self.itam]}xCD3z-{HINGE[self.hinge]}"


def enumerate_constructs() -> list[Construct]:
    out: list[Construct] = []
    idx = 0
    for c in range(len(COSTIM)):
        for i in range(len(ITAM)):
            for h in range(len(HINGE)):
                out.append(Construct(index=idx, costim=c, itam=i, hinge=h))
                idx += 1
    return out


def decode(action: int) -> Construct:
    h = action % len(HINGE)
    i = (action // len(HINGE)) % len(ITAM)
    c = action // (len(HINGE) * len(ITAM))
    return Construct(index=action, costim=c, itam=i, hinge=h)


def persistence_table(
    n_records: int,
    phenotype_dim: int = 128,
    construct_dim: int = 64,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    phenotype = rng.normal(0.0, 1.0, size=(n_records, phenotype_dim)).astype(np.float32)
    construct = rng.normal(0.0, 1.0, size=(n_records, construct_dim)).astype(np.float32)

    w_phen = rng.normal(0.0, 1.0, size=phenotype_dim) / np.sqrt(phenotype_dim)
    w_con = rng.normal(0.0, 1.0, size=construct_dim) / np.sqrt(construct_dim)
    cross = (phenotype @ w_phen) * (construct @ w_con)
    logit = 0.9 * (phenotype @ w_phen) + 0.6 * (construct @ w_con) + 0.8 * cross
    prob = 1.0 / (1.0 + np.exp(-logit))
    label = (rng.uniform(size=n_records) < prob).astype(np.float32)
    return (
        torch.from_numpy(phenotype),
        torch.from_numpy(construct),
        torch.from_numpy(label),
    )
