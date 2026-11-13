"""Unit tests for the training loop, checkpoint resuming, and Weights & Biases tracker."""

from __future__ import annotations

from pathlib import Path

from mnist_pipeline.config import (
    DataConfig,
    ModelConfig,
    PathsConfig,
    PipelineConfig,
    TrainConfig,
    WandbConfig,
)
from mnist_pipeline.train import train_model


def test_training_smoke_run_and_checkpoint_resume(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "artifacts" / "checkpoints"
    reports_dir = tmp_path / "artifacts" / "reports"
    cfg = PipelineConfig(
        data=DataConfig(
            raw_dir=str(tmp_path / "data" / "raw"),
            processed_dir=str(tmp_path / "data" / "processed"),
            download=False,
            val_fraction=0.2,
            batch_size=32,
            seed=42,
            allow_synthetic_fallback=True,
            num_synthetic_train=160,
            num_synthetic_test=40,
        ),
        model=ModelConfig(conv_channels=[16, 32], hidden_dim=64, dropout=0.1),
        train=TrainConfig(
            epochs=2,
            lr=1e-3,
            optimizer="adam",
            seed=42,
            deterministic=True,
            device="cpu",
        ),
        wandb=WandbConfig(enabled=False, mode="disabled"),
        paths=PathsConfig(
            artifacts_dir=str(tmp_path / "artifacts"),
            checkpoints_dir=str(ckpt_dir),
            reports_dir=str(reports_dir),
            plots_dir=str(reports_dir / "plots"),
        ),
    )

    summary = train_model(cfg)
    assert len(summary.history) == 2
    assert summary.history[-1].train_loss < summary.history[0].train_loss
    assert (ckpt_dir / "checkpoint_best.pt").exists()
    assert (ckpt_dir / "checkpoint_last.pt").exists()
    assert (ckpt_dir / "model.safetensors").exists()
    assert (ckpt_dir / "config.json").exists()
    assert (reports_dir / "train_history.csv").exists()
    assert (reports_dir / "environment.json").exists()

    # Resume from checkpoint_last.pt for 1 additional epoch (epoch 3)
    cfg.train.epochs = 3
    cfg.train.resume_from = str(ckpt_dir / "checkpoint_last.pt")
    resumed_summary = train_model(cfg)
    assert len(resumed_summary.history) == 1
    assert resumed_summary.history[0].epoch == 3
