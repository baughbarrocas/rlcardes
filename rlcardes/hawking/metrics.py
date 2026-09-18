import numpy as np
from scipy.stats import rankdata


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float((prediction == target).mean())


def misclassification_rate(prediction: np.ndarray, target: np.ndarray) -> float:
    return float((prediction != target).mean()) * 100.0


def macro_f1(prediction: np.ndarray, target: np.ndarray, n_classes: int) -> float:
    scores = np.zeros(n_classes)
    for k in range(n_classes):
        tp = float(((prediction == k) & (target == k)).sum())
        fp = float(((prediction == k) & (target != k)).sum())
        fn = float(((prediction != k) & (target == k)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        scores[k] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(scores.mean())
