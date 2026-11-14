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
