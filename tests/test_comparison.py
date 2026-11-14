"""Unit tests for model comparison engine and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mnist_pipeline.comparison import (
    generate_model_comparison,
    load_model_artifacts,
    plot_accuracy_latency_pareto,
    plot_latency_throughput_curves,
    plot_per_class_f1_comparison,
    render_comparison_markdown,
)
from mnist_pipeline.utils import save_json


@pytest.fixture
def dummy_model_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create two mock model artifact directories with evaluation and benchmark metrics."""
    cnn_dir = tmp_path / "models" / "mnist_cnn"
    vit_dir = tmp_path / "models" / "mnist_vit"

    (cnn_dir / "evaluation").mkdir(parents=True)
    (cnn_dir / "benchmarks").mkdir(parents=True)
    (cnn_dir / "checkpoints").mkdir(parents=True)

    (vit_dir / "evaluation").mkdir(parents=True)
    (vit_dir / "benchmarks").mkdir(parents=True)
    (vit_dir / "checkpoints").mkdir(parents=True)

    # Mock CNN metrics
    cnn_eval: dict[str, Any] = {
        "test_acc": 0.9902,
        "test_loss": 0.0315,
        "macro_f1": 0.9901,
        "per_class": {
            str(i): {"precision": 0.99, "recall": 0.99, "f1-score": 0.99, "support": 1000}
            for i in range(10)
        },
    }
    cnn_bench: dict[str, Any] = {
        "trainable_params": 421834,
        "checkpoint_size_mb": 4.85,
        "safetensors_size_mb": 1.62,
        "batch_metrics": {
            "1": {"p50_ms": 0.25, "p95_ms": 0.38, "samples_per_sec": 3800.0},
            "32": {"p50_ms": 1.10, "p95_ms": 1.45, "samples_per_sec": 26000.0},
            "128": {"p50_ms": 3.20, "p95_ms": 3.90, "samples_per_sec": 38500.0},
        },
    }
    save_json(cnn_eval, cnn_dir / "evaluation" / "test_metrics.json")
    save_json(cnn_bench, cnn_dir / "benchmarks" / "benchmark_metrics.json")
    save_json({"architecture": "cnn", "name": "mnist_cnn"}, cnn_dir / "checkpoints" / "config.json")

    # Mock ViT metrics
    vit_eval: dict[str, Any] = {
        "test_acc": 0.9880,
        "test_loss": 0.0390,
        "macro_f1": 0.9878,
        "per_class": {
            str(i): {"precision": 0.988, "recall": 0.988, "f1-score": 0.988, "support": 1000}
            for i in range(10)
        },
    }
    vit_bench: dict[str, Any] = {
        "trainable_params": 404874,
        "checkpoint_size_mb": 4.65,
        "safetensors_size_mb": 1.55,
        "batch_metrics": {
            "1": {"p50_ms": 1.80, "p95_ms": 2.20, "samples_per_sec": 520.0},
            "32": {"p50_ms": 5.40, "p95_ms": 6.80, "samples_per_sec": 5500.0},
            "128": {"p50_ms": 18.5, "p95_ms": 22.1, "samples_per_sec": 6800.0},
        },
    }
    save_json(vit_eval, vit_dir / "evaluation" / "test_metrics.json")
    save_json(vit_bench, vit_dir / "benchmarks" / "benchmark_metrics.json")
    save_json(
        {"architecture": "transformer", "name": "mnist_vit"},
        vit_dir / "checkpoints" / "config.json",
    )

    return cnn_dir, vit_dir


def test_load_model_artifacts(dummy_model_dirs: tuple[Path, Path]) -> None:
    cnn_dir, vit_dir = dummy_model_dirs
    cnn = load_model_artifacts("CNN", cnn_dir)
    assert cnn.name == "CNN"
    assert cnn.architecture == "cnn"
    assert cnn.test_metrics["test_acc"] == 0.9902
    assert cnn.benchmark_metrics["trainable_params"] == 421834

    vit = load_model_artifacts("ViT", vit_dir)
    assert vit.name == "ViT"
    assert vit.architecture == "transformer"
    assert vit.test_metrics["test_acc"] == 0.9880


def test_load_model_artifacts_missing_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_model_artifacts("Empty", empty_dir)


def test_comparison_plots_generation(dummy_model_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    cnn_dir, vit_dir = dummy_model_dirs
    models = [
        load_model_artifacts("CNN", cnn_dir),
        load_model_artifacts("ViT", vit_dir),
    ]
    plots_dir = tmp_path / "plots"

    pareto_p = plot_accuracy_latency_pareto(models, plots_dir / "pareto.png")
    assert pareto_p.exists()
    assert pareto_p.stat().st_size > 0

    f1_p = plot_per_class_f1_comparison(models, plots_dir / "f1.png")
    assert f1_p.exists()
    assert f1_p.stat().st_size > 0

    scaling_p = plot_latency_throughput_curves(models, plots_dir / "scaling.png")
    assert scaling_p.exists()
    assert scaling_p.stat().st_size > 0


def test_render_comparison_markdown(dummy_model_dirs: tuple[Path, Path]) -> None:
    cnn_dir, vit_dir = dummy_model_dirs
    models = [
        load_model_artifacts("CNN", cnn_dir),
        load_model_artifacts("ViT", vit_dir),
    ]
    md = render_comparison_markdown(
        models,
        {
            "pareto": "plots/pareto.png",
            "per_class_f1": "plots/f1.png",
            "scaling": "plots/scaling.png",
        },
    )
    assert "# Systematic Model Comparison" in md
    assert "99.02%" in md
    assert "98.80%" in md
    assert "421,834" in md
    assert "![Pareto Frontier](plots/pareto.png)" in md


def test_generate_model_comparison_end_to_end(
    dummy_model_dirs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    cnn_dir, vit_dir = dummy_model_dirs
    models_dict = {
        "CNN": cnn_dir,
        "ViT": vit_dir,
    }
    out_dir = tmp_path / "comparisons"
    docs_target = tmp_path / "docs" / "study.md"

    report_p, plots = generate_model_comparison(
        models_dict=models_dict,
        output_dir=out_dir,
        copy_to_docs=docs_target,
    )

    assert report_p.exists()
    assert len(plots) == 3
    for p in plots:
        assert p.exists()
    assert docs_target.exists()
