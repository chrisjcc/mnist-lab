"""Unit tests for evaluation, benchmarking, plot generation, and report/card rendering."""

from __future__ import annotations

from pathlib import Path

from mnist_pipeline.config import load_hydra_config
from mnist_pipeline.data import preprocess_mnist_data
from mnist_pipeline.evaluate import benchmark_checkpoint, evaluate_checkpoint
from mnist_pipeline.reporting import generate_all_reports
from mnist_pipeline.train import train_model


def test_end_to_end_eval_benchmark_and_reporting(tmp_path: Path) -> None:
    """Verify full evaluation, benchmarking, plot generation, and card rendering."""
    raw_dir = tmp_path / "data/raw"
    processed_dir = tmp_path / "data/processed"
    ckpt_dir = tmp_path / "artifacts/checkpoints"
    eval_dir = tmp_path / "artifacts/evaluation"
    report_dir = tmp_path / "artifacts/reports"
    plots_dir = tmp_path / "artifacts/reports/plots"
    bench_dir = tmp_path / "artifacts/benchmarks"
    docs_dir = tmp_path / "docs"
    repo_root = Path(__file__).resolve().parent.parent

    cfg = load_hydra_config(
        overrides=[
            f"data.raw_dir={raw_dir}",
            f"data.processed_dir={processed_dir}",
            "data.download=false",
            "data.allow_synthetic_fallback=true",
            "data.num_synthetic_train=120",
            "data.num_synthetic_test=40",
            "data.batch_size=32",
            "train.epochs=1",
            "train.device=cpu",
            "evaluate.benchmark_warmup_runs=1",
            "evaluate.benchmark_timed_runs=3",
            "evaluate.benchmark_batch_sizes=[1,16]",
            "wandb.mode=disabled",
            f"paths.artifacts_dir={tmp_path / 'artifacts'}",
            f"paths.checkpoints_dir={ckpt_dir}",
            f"paths.evaluation_dir={eval_dir}",
            f"paths.benchmarks_dir={bench_dir}",
            f"paths.reports_dir={report_dir}",
            f"paths.plots_dir={plots_dir}",
            f"paths.templates_dir={repo_root / 'templates'}",
            f"paths.docs_dir={docs_dir}",
        ]
    )

    preprocess_mnist_data(cfg.data)
    train_model(cfg)

    test_metrics = evaluate_checkpoint(cfg)
    assert "test_acc" in test_metrics
    assert (eval_dir / "test_metrics.json").exists()
    assert (eval_dir / "classification_report.txt").exists()
    assert (eval_dir / "confusion_matrix.csv").exists()

    bench_metrics = benchmark_checkpoint(cfg)
    assert "1" in bench_metrics["batch_metrics"]
    assert "16" in bench_metrics["batch_metrics"]
    assert (bench_dir / "benchmark_metrics.json").exists()

    outputs = generate_all_reports(
        cfg,
        template_dir=repo_root / "templates",
        docs_dir=docs_dir,
    )

    for artifact_path_str in outputs.values():
        path = Path(artifact_path_str)
        assert path.exists(), f"Expected generated artifact {path} to exist"
        assert path.stat().st_size > 0, f"Expected generated artifact {path} to be non-empty"

    model_card_text = (docs_dir / "model_card.md").read_text(encoding="utf-8")
    dataset_card_text = (docs_dir / "dataset_card.md").read_text(encoding="utf-8")
    assert "MNISTConvNet" in model_card_text
    assert "stratified_train_val_split" in dataset_card_text
