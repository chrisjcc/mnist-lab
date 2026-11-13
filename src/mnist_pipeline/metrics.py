"""Classification metrics computation for MNIST evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Structured container for multi-class classification evaluation metrics.

    Attributes:
        test_loss: Mean cross-entropy loss across the evaluation set.
        test_acc: Overall classification accuracy in `[0, 1]`.
        macro_precision: Unweighted mean precision across all classes.
        macro_recall: Unweighted mean recall across all classes.
        macro_f1: Unweighted mean F1-score across all classes.
        weighted_precision: Support-weighted precision across all classes.
        weighted_recall: Support-weighted recall across all classes.
        weighted_f1: Support-weighted F1-score across all classes.
        num_samples: Total number of evaluated samples.
        per_class: Mapping from class digit string (`"0"`–`"9"`) to precision,
            recall, F1-score, and sample support.
        confusion_matrix: Nested list (`num_classes x num_classes`) where rows
            are ground-truth classes and columns are predicted classes.
        classification_report_text: Formatted text report produced by
            `sklearn.metrics.classification_report`.
    """

    test_loss: float
    test_acc: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    num_samples: int
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]
    classification_report_text: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics into a JSON-serializable dictionary (excluding confusion matrix).

        Returns:
            Dictionary suitable for saving to `artifacts/evaluation/test_metrics.json`.
        """
        return {
            "test_loss": round(self.test_loss, 6),
            "test_acc": round(self.test_acc, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "macro_f1": round(self.macro_f1, 6),
            "weighted_precision": round(self.weighted_precision, 6),
            "weighted_recall": round(self.weighted_recall, 6),
            "weighted_f1": round(self.weighted_f1, 6),
            "num_samples": self.num_samples,
            "per_class": self.per_class,
        }


def compute_classification_metrics(
    y_true: list[int] | np.ndarray[Any, Any],
    y_pred: list[int] | np.ndarray[Any, Any],
    avg_loss: float,
    num_classes: int = 10,
) -> ClassificationMetrics:
    """Compute overall, macro, weighted, and per-class classification metrics.

    Args:
        y_true: Ground-truth integer class labels in `[0, num_classes - 1]`.
        y_pred: Predicted integer class labels in `[0, num_classes - 1]`.
        avg_loss: Average cross-entropy loss across the evaluated dataset.
        num_classes: Total number of target digit classes.

    Returns:
        Populated `ClassificationMetrics` instance.

    Raises:
        ValueError: If `y_true` and `y_pred` are empty or have mismatched lengths.
    """
    true_arr = np.asarray(y_true, dtype=np.int64)
    pred_arr = np.asarray(y_pred, dtype=np.int64)

    if true_arr.ndim != 1 or pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays of class indices.")
    if len(true_arr) == 0:
        raise ValueError("Cannot compute classification metrics on empty predictions.")
    if len(true_arr) != len(pred_arr):
        raise ValueError(
            f"Length mismatch between y_true ({len(true_arr)}) and y_pred ({len(pred_arr)})."
        )

    labels = list(range(num_classes))
    target_names = [f"Digit {c}" for c in labels]

    overall_acc = float(accuracy_score(true_arr, pred_arr))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        true_arr,
        pred_arr,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        true_arr,
        pred_arr,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    class_p, class_r, class_f1, class_support = precision_recall_fscore_support(
        true_arr,
        pred_arr,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for idx, cls_id in enumerate(labels):
        per_class[str(cls_id)] = {
            "precision": round(float(np.asarray(class_p)[idx]), 6),
            "recall": round(float(np.asarray(class_r)[idx]), 6),
            "f1": round(float(np.asarray(class_f1)[idx]), 6),
            "support": int(np.asarray(class_support)[idx]),
        }

    cm = confusion_matrix(true_arr, pred_arr, labels=labels)
    cm_list: list[list[int]] = [[int(val) for val in row] for row in cm.tolist()]

    report_str = str(
        classification_report(
            true_arr,
            pred_arr,
            labels=labels,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    return ClassificationMetrics(
        test_loss=float(avg_loss),
        test_acc=overall_acc,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        weighted_precision=float(weighted_p),
        weighted_recall=float(weighted_r),
        weighted_f1=float(weighted_f1),
        num_samples=int(len(true_arr)),
        per_class=per_class,
        confusion_matrix=cm_list,
        classification_report_text=report_str,
    )
