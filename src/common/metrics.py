"""Reusable classification metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> dict[str, Any]:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        conf[int(t), int(p)] += 1

    precisions, recalls, f1s = [], [], []
    per_class: dict[str, dict[str, float]] = {}
    for class_id in range(num_classes):
        tp = conf[class_id, class_id]
        fp = conf[:, class_id].sum() - tp
        fn = conf[class_id, :].sum() - tp
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        per_class[str(class_id)] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
    }
