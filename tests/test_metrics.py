"""Unit tests for classification metrics, held-out evaluation, and benchmarking."""

from __future__ import annotations

import numpy as np
import pytest

from mnist_pipeline.metrics import compute_classification_metrics


def test_compute_classification_metrics_perfect_predictions() -> None:
    """Verify exact 1.0 accuracy, precision, recall, F1, and diagonal confusion matrix."""
    y_true = [i % 10 for i in range(100)]
    y_pred = list(y_true)
    metrics = compute_classification_metrics(y_true, y_pred, avg_loss=0.0123, num_classes=10)

    assert metrics.test_acc == pytest.approx(1.0)
    assert metrics.macro_precision == pytest.approx(1.0)
    assert metrics.macro_recall == pytest.approx(1.0)
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.weighted_f1 == pytest.approx(1.0)
    assert metrics.num_samples == 100

    cm = np.asarray(metrics.confusion_matrix)
    assert cm.shape == (10, 10)
    assert int(cm.sum()) == 100
    assert np.all(np.diag(cm) == 10)
    assert "Digit 0" in metrics.classification_report_text


def test_compute_classification_metrics_with_known_errors() -> None:
    """Verify confusion matrix and macro metrics when known misclassifications occur."""
    y_true = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_pred = [0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    metrics = compute_classification_metrics(y_true, y_pred, avg_loss=0.15, num_classes=10)

    assert metrics.test_acc == pytest.approx(11.0 / 12.0)
    assert metrics.confusion_matrix[0][0] == 1
    assert metrics.confusion_matrix[0][1] == 1
    assert metrics.per_class["0"]["recall"] == pytest.approx(0.5)
    assert metrics.per_class["1"]["precision"] == pytest.approx(2.0 / 3.0)


def test_compute_classification_metrics_invalid_inputs_raise() -> None:
    """Verify empty or mismatched length arrays raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_classification_metrics([], [], avg_loss=0.0)

    with pytest.raises(ValueError, match="Length mismatch"):
        compute_classification_metrics([0, 1], [0], avg_loss=0.0)
