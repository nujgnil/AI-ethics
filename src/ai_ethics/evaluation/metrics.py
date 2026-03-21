from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def expected_calibration_error(
    y_true: np.ndarray,
    probas: np.ndarray,
    n_bins: int = 10,
) -> float:
    if probas.ndim != 2 or len(y_true) == 0:
        return float("nan")

    confidences = probas.max(axis=1)
    predictions = probas.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        acc = correct[in_bin].mean()
        conf = confidences[in_bin].mean()
        ece += (in_bin.mean()) * abs(acc - conf)
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, probas: np.ndarray) -> float:
    n_classes = probas.shape[1]
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probas - one_hot) ** 2, axis=1)))


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    probas: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if probas is None:
        metrics["auroc_ovr"] = float("nan")
        metrics["pr_auc_macro"] = float("nan")
        metrics["brier_score"] = float("nan")
        metrics["ece_10bin"] = float("nan")
        return metrics

    n_classes = probas.shape[1]
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        if n_classes == 2:
            auroc = roc_auc_score(y_true, probas[:, 1])
            pr_auc = average_precision_score(y_true, probas[:, 1])
        else:
            auroc = roc_auc_score(y_true_bin, probas, multi_class="ovr", average="macro")
            pr_auc = average_precision_score(y_true_bin, probas, average="macro")
        metrics["auroc_ovr"] = float(auroc)
        metrics["pr_auc_macro"] = float(pr_auc)
    except Exception:
        metrics["auroc_ovr"] = float("nan")
        metrics["pr_auc_macro"] = float("nan")

    try:
        metrics["brier_score"] = multiclass_brier_score(y_true, probas)
    except Exception:
        metrics["brier_score"] = float("nan")

    metrics["ece_10bin"] = expected_calibration_error(y_true, probas, n_bins=10)
    return metrics
