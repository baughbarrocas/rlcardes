import os

import numpy as np

MARKER_GENES: tuple[str, str, str, str, str] = (
    "PDCD1",
    "LAG3",
    "HAVCR2",
    "TIGIT",
    "TOX",
)


def load_geo_counts(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"count matrix not found at {path}; fetch the GEO accession listed in plate_sources.txt"
        )
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D cell-by-gene matrix, got shape {array.shape}")
    return array.astype(np.float32)


def preprocess_counts(
    counts: np.ndarray,
    min_genes: int = 200,
    max_genes: int = 5000,
    target_sum: float = 1e4,
) -> np.ndarray:
    detected = (counts > 0).sum(axis=1)
    keep = (detected >= min_genes) & (detected <= max_genes)
    filtered = counts[keep]
    library = filtered.sum(axis=1, keepdims=True)
    library[library == 0.0] = 1.0
    normed = filtered / library * target_sum
    return np.log1p(normed).astype(np.float32)


def select_variable_genes(matrix: np.ndarray, n_top: int = 3000) -> np.ndarray:
    if n_top >= matrix.shape[1]:
        return np.arange(matrix.shape[1])
    variance = matrix.var(axis=0)
    return np.argsort(variance)[::-1][:n_top].copy()
