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
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"
    ckpt_dir = artifacts_dir / "checkpoints"
    eval_dir = artifacts_dir / "evaluation"
    report_dir = artifacts_dir / "reports"
    plots_dir = report_dir / "plots"
    bench_dir = artifacts_dir / "benchmarks"
    docs_dir = tmp_path / "docs"
    repo_root = Path(__file__).resolve().parent.parent

    cfg = load_hydra_config(
        overrides=[
            f"data.raw_dir={raw_dir.as_posix()}",
            f"data.processed_dir={processed_dir.as_posix()}",
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
            f"paths.artifacts_dir={artifacts_dir.as_posix()}",
            f"paths.checkpoints_dir={ckpt_dir.as_posix()}",
            f"paths.evaluation_dir={eval_dir.as_posix()}",
            f"paths.benchmarks_dir={bench_dir.as_posix()}",
            f"paths.reports_dir={report_dir.as_posix()}",
            f"paths.plots_dir={plots_dir.as_posix()}",
            f"paths.templates_dir={repo_root / 'templates'}",
            f"paths.docs_dir={docs_dir.as_posix()}",
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


def test_vit_model_card_rendering(tmp_path: Path) -> None:
    """Verify render_model_card correctly outputs ViT architecture, specs, and CNN cross-link."""
    from mnist_pipeline.reporting import render_model_card

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_hydra_config(overrides=["model=transformer"])

    test_metrics = {
        "test_acc": 0.985,
        "test_loss": 0.05,
        "macro_f1": 0.984,
        "macro_precision": 0.985,
        "macro_recall": 0.984,
        "weighted_f1": 0.985,
        "num_samples": 10000,
        "per_class": {
            str(i): {"precision": 0.98, "recall": 0.98, "f1": 0.98, "support": 1000}
            for i in range(10)
        },
    }
    benchmark_metrics = {
        "trainable_params": 318000,
        "checkpoint_size_mb": 1.25,
        "safetensors_size_mb": 1.22,
        "device": "cpu",
        "timed_runs": 10,
        "warmup_runs": 2,
        "batch_metrics": {
            "1": {
                "mean_ms": 1.5,
                "std_ms": 0.1,
                "p50_ms": 1.4,
                "p95_ms": 1.6,
                "p99_ms": 1.7,
                "samples_per_sec": 650.0,
            }
        },
    }
    out_path = tmp_path / "vit_model_card.md"
    rendered_path = render_model_card(
        cfg=cfg,
        test_metrics=test_metrics,
        benchmark_metrics=benchmark_metrics,
        env_meta={"torch_version": "2.5.1"},
        template_path=repo_root / "templates" / "model_card_template.md",
        output_path=out_path,
    )
    assert rendered_path.exists()
    content = rendered_path.read_text(encoding="utf-8")
    assert "MNISTViT" in content
    assert "Vision Transformer" in content
    assert "patch size 4x4" in content
    assert "chrisjcc/mnist-vit" in content
    assert "chrisjcc/mnist-cnn" in content  # Sibling model card cross-link
