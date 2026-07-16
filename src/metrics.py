from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    enable_auc: bool = False,
) -> dict[str, Any]:
    preds = probabilities.argmax(axis=1)
    label_ids = list(range(len(class_names)))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "weighted_recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=label_ids).tolist(),
    }

    per_precision = precision_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    per_recall = recall_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    per_f1 = f1_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    for idx, class_name in enumerate(class_names):
        prefix = f"per_class/{class_name}"
        metrics[f"{prefix}_precision"] = float(per_precision[idx])
        metrics[f"{prefix}_recall"] = float(per_recall[idx])
        metrics[f"{prefix}_f1"] = float(per_f1[idx])

    if enable_auc:
        one_hot = label_binarize(labels, classes=label_ids)
        try:
            metrics["macro_roc_auc"] = float(
                roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")
            )
            metrics["weighted_roc_auc"] = float(
                roc_auc_score(one_hot, probabilities, average="weighted", multi_class="ovr")
            )
        except ValueError:
            metrics["macro_roc_auc"] = None
            metrics["weighted_roc_auc"] = None
        try:
            metrics["macro_pr_auc"] = float(average_precision_score(one_hot, probabilities, average="macro"))
            metrics["weighted_pr_auc"] = float(
                average_precision_score(one_hot, probabilities, average="weighted")
            )
        except ValueError:
            metrics["macro_pr_auc"] = None
            metrics["weighted_pr_auc"] = None

    return metrics


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
