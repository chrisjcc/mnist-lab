"""Unit tests for Hugging Face Model Hub and Dataset Hub staging and dry-run uploads."""

from __future__ import annotations

from pathlib import Path

from mnist_pipeline.config import load_hydra_config
from mnist_pipeline.data import preprocess_mnist_data
from mnist_pipeline.evaluate import benchmark_checkpoint, evaluate_checkpoint
from mnist_pipeline.hf_hub import stage_dataset_release, stage_model_release, upload_to_hf_hub
from mnist_pipeline.reporting import generate_all_reports
from mnist_pipeline.train import train_model


def test_hf_hub_staging_and_dry_run_upload(tmp_path: Path) -> None:
    """Verify stage_model_release, stage_dataset_release, and upload_to_hf_hub dry-run."""
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_hydra_config(
        overrides=[
            f"data.raw_dir={tmp_path / 'data/raw'}",
            f"data.processed_dir={tmp_path / 'data/processed'}",
            "data.download=false",
            "data.allow_synthetic_fallback=true",
            "data.num_synthetic_train=100",
            "data.num_synthetic_test=40",
            "data.batch_size=32",
            "train.epochs=1",
            "train.device=cpu",
            "evaluate.benchmark_warmup_runs=1",
            "evaluate.benchmark_timed_runs=2",
            "evaluate.benchmark_batch_sizes=[1,16]",
            "wandb.mode=disabled",
            "hf.dry_run=true",
            f"paths.artifacts_dir={tmp_path / 'artifacts'}",
            f"paths.checkpoints_dir={tmp_path / 'artifacts/checkpoints'}",
            f"paths.evaluation_dir={tmp_path / 'artifacts/evaluation'}",
            f"paths.benchmarks_dir={tmp_path / 'artifacts/benchmarks'}",
            f"paths.reports_dir={tmp_path / 'artifacts/reports'}",
            f"paths.plots_dir={tmp_path / 'artifacts/reports/plots'}",
            f"paths.templates_dir={repo_root / 'templates'}",
            f"paths.docs_dir={tmp_path / 'docs'}",
        ]
    )

    preprocess_mnist_data(cfg.data)
    train_model(cfg)
    evaluate_checkpoint(cfg)
    benchmark_checkpoint(cfg)
    generate_all_reports(
        cfg,
        template_dir=repo_root / "templates",
        docs_dir=tmp_path / "docs",
    )

    model_stage = tmp_path / "staged_model"
    staged_model_files = stage_model_release(cfg, model_stage)
    assert "model.safetensors" in staged_model_files
    assert "config.json" in staged_model_files
    assert "checkpoint_best.pt" in staged_model_files
    assert "README.md" in staged_model_files
    assert "test_metrics.json" in staged_model_files
    assert "benchmark_metrics.json" in staged_model_files
    assert "plots/loss_curve.png" in staged_model_files

    dataset_stage = tmp_path / "staged_dataset"
    staged_dataset_files = stage_dataset_release(cfg, dataset_stage)
    assert "README.md" in staged_dataset_files
    assert "split_metadata.json" in staged_dataset_files

    summary = upload_to_hf_hub(cfg, staging_root=tmp_path / "upload_stage")
    assert summary["dry_run"] is True
    assert "model.safetensors" in summary["staged_model_files"]
    assert "README.md" in summary["staged_dataset_files"]
