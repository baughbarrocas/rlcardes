import numpy as np
from scipy.stats import norm, ttest_rel


def paired_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    result = ttest_rel(a, b)
    return float(result.statistic), float(result.pvalue)


def bootstrap_ci(
    differences: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = differences.shape[0]
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = differences[rng.integers(0, n, size=n)]
        means[i] = sample.mean()
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.shape[0]
    cumulative = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        cumulative[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = cumulative
    return out


def _delong_components(predictions_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    k = predictions_sorted.shape[0]
    n = predictions_sorted.shape[1] - m
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(predictions_sorted[r, :m])
        ty[r] = _midrank(predictions_sorted[r, m:])
        tz[r] = _midrank(predictions_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    return aucs, covariance


def delong_test(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    order = np.argsort(-labels)
    m = int(labels.sum())
    predictions_sorted = np.vstack([scores_a[order], scores_b[order]])
    aucs, covariance = _delong_components(predictions_sorted, m)
    weights = np.array([1.0, -1.0])
    variance = float(weights @ covariance @ weights)
    if variance <= 0.0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), p_value
