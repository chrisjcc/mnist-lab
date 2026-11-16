"""Unit tests for W&B hyperparameter sweep trial runner and translation utilities."""

from __future__ import annotations

from typing import Any

import pytest
from scripts.run_sweep_trial import build_hydra_overrides, parse_args, run_trial

from mnist_pipeline.config import load_hydra_config


def test_build_hydra_overrides_alias_translation() -> None:
    sweep_params: dict[str, Any] = {
        "learning_rate": 0.0005,
        "dropout": 0.35,
        "batch_size": 64,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "epochs": 3,
        "unknown_flag": "ignored",
    }
    overrides = build_hydra_overrides(sweep_params)

    assert "train.lr=0.0005" in overrides
    assert "model.dropout=0.35" in overrides
    assert "data.batch_size=64" in overrides
    assert "train.optimizer=adamw" in overrides
    assert "train.weight_decay=0.0001" in overrides
    assert "train.epochs=3" in overrides
    assert not any("unknown_flag" in o for o in overrides)


def test_build_hydra_overrides_smoke_mode() -> None:
    overrides = build_hydra_overrides({"learning_rate": 0.001}, smoke=True)
    assert "train.lr=0.001" in overrides
    assert "train.epochs=1" in overrides
    assert "train.max_train_batches=2" in overrides
    assert "train.max_val_batches=2" in overrides


def test_parse_args_converts_numeric_and_extracts_smoke() -> None:
    cli_dict, smoke, extras = parse_args(
        [
            "--learning_rate",
            "0.002",
            "--dropout",
            "0.25",
            "--batch_size",
            "128",
            "--smoke",
            "train.seed=99",
        ]
    )
    assert smoke is True
    assert cli_dict["learning_rate"] == pytest.approx(0.002)
    assert cli_dict["dropout"] == pytest.approx(0.25)
    assert cli_dict["batch_size"] == 128
    assert extras == ["train.seed=99"]


def test_run_trial_smoke_execution_and_ssot_preservation() -> None:
    summary = run_trial(
        cli_args={"learning_rate": 0.003, "dropout": 0.3},
        smoke=True,
    )
    assert summary.best_epoch == 1
    assert 0.0 <= summary.best_val_accuracy <= 1.0

    # Verify that non-swept parameters respected params.yaml defaults
    cfg = load_hydra_config(overrides=["train.lr=0.003"])
    assert cfg.train.lr == pytest.approx(0.003)
    assert cfg.train.optimizer == "adam"
    assert cfg.model.in_channels == 1
    assert cfg.data.val_fraction == pytest.approx(0.1)


def test_build_hydra_overrides_transformer_parameters() -> None:
    """Verify build_hydra_overrides translates ViT architecture hyperparameters."""
    sweep_params: dict[str, Any] = {
        "model": "transformer",
        "depth": 3,
        "num_heads": 4,
        "embed_dim": 128,
        "mlp_ratio": 2.0,
        "attention_dropout": 0.05,
        "patch_size": 4,
        "dropout": 0.1,
    }
    overrides = build_hydra_overrides(sweep_params)

    assert "model=transformer" in overrides
    assert "model.depth=3" in overrides
    assert "model.num_heads=4" in overrides
    assert "model.embed_dim=128" in overrides
    assert "model.mlp_ratio=2.0" in overrides
    assert "model.attention_dropout=0.05" in overrides
    assert "model.patch_size=4" in overrides
    assert "model.dropout=0.1" in overrides

    # Verify Hydra loads cleanly with these translated overrides
    cfg = load_hydra_config(overrides=overrides)
    assert cfg.model.architecture == "transformer"
    assert cfg.model.depth == 3
    assert cfg.model.num_heads == 4
    assert cfg.model.embed_dim == 128
    assert cfg.model.mlp_ratio == 2.0
    assert cfg.model.attention_dropout == 0.05


def test_transformer_sweep_yaml_validity() -> None:
    """Verify sweeps/transformer_bayes.yaml parses as valid W&B sweep configuration."""
    from pathlib import Path

    import yaml

    sweep_file = Path("sweeps") / "transformer_bayes.yaml"
    assert sweep_file.exists()

    with sweep_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["name"] == "mnist-vit-bayes-sweep"
    assert data["method"] == "bayes"
    assert data["program"] == "scripts/run_sweep_trial.py"
    assert data["metric"]["name"] == "val/accuracy"
    assert data["metric"]["goal"] == "maximize"

    params = data["parameters"]
    assert params["model"]["value"] == "transformer"
    assert "depth" in params
    assert "num_heads" in params
    assert "embed_dim" in params
    assert "learning_rate" in params
    assert "optimizer" in params
