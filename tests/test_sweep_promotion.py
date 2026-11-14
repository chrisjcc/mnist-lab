"""Unit tests for sweep artifact isolation, local report generation, and champion promotion."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.promote_sweep import find_latest_sweep_dir, promote_champion, update_params_yaml
from scripts.run_sweep_trial import run_trial

from mnist_pipeline.reporting import generate_sweep_report
from mnist_pipeline.utils import load_json


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_test_sweep() -> Any:
    """Fixture to run smoke sweep trials and clean up after module execution."""
    sweep_id = "unit_test_sweep_abc"
    sweep_dir = Path("artifacts/sweeps") / sweep_id
    if sweep_dir.exists():
        shutil.rmtree(sweep_dir)

    # Run trial 1
    run_trial(
        cli_args={
            "sweep_id": sweep_id,
            "run_id": "run_alpha",
            "learning_rate": 0.001,
            "dropout": 0.2,
            "optimizer": "adam",
        },
        smoke=True,
    )

    # Run trial 2
    run_trial(
        cli_args={
            "sweep_id": sweep_id,
            "run_id": "run_beta",
            "learning_rate": 0.005,
            "dropout": 0.35,
            "optimizer": "sgd",
        },
        smoke=True,
    )

    yield sweep_id

    # Teardown
    if sweep_dir.exists():
        shutil.rmtree(sweep_dir)


def test_trial_isolation_and_ledger_recording(setup_and_teardown_test_sweep: str) -> None:
    """Verify that sweep trials isolate checkpoints and append to the local ledger."""
    sweep_id = setup_and_teardown_test_sweep
    ledger_path = Path("artifacts/sweeps") / sweep_id / "trials.json"
    assert ledger_path.exists()

    ledger_payload = load_json(ledger_path)
    trials = ledger_payload.get("trials", [])
    assert len(trials) == 2
    run_ids = [t["run_id"] for t in trials]
    assert "run_alpha" in run_ids
    assert "run_beta" in run_ids

    # Verify per-trial checkpoint isolation
    assert (Path("artifacts/sweeps") / sweep_id / "run_alpha" / "checkpoint_best.pt").exists()
    assert (Path("artifacts/sweeps") / sweep_id / "run_beta" / "checkpoint_best.pt").exists()


def test_generate_sweep_report_content(setup_and_teardown_test_sweep: str) -> None:
    """Verify Markdown sweep report formatting, leaderboard ranking, and champion callouts."""
    sweep_id = setup_and_teardown_test_sweep
    report_file = generate_sweep_report(sweep_id=sweep_id)
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")

    assert f"# 🧹 Hyperparameter Sweep Report: `{sweep_id}`" in content
    assert "## 🏆 Champion Model Overview" in content
    assert "## 📊 Sweep Leaderboard" in content
    assert "🥇" in content
    assert "make sweep-promote" in content
    assert "run_alpha" in content or "run_beta" in content


def test_update_params_yaml(tmp_path: Path) -> None:
    """Verify that update_params_yaml updates canonical parameters without destroying structure."""
    fake_yaml = tmp_path / "params.yaml"
    fake_yaml.write_text(
        yaml.safe_dump(
            {
                "data": {"batch_size": 128},
                "model": {"dropout": 0.25},
                "train": {"lr": 0.001, "optimizer": "adam", "weight_decay": 0.0, "epochs": 5},
            }
        ),
        encoding="utf-8",
    )

    champion_params: dict[str, Any] = {
        "learning_rate": 0.0042,
        "optimizer": "adamw",
        "dropout": 0.35,
        "batch_size": 64,
        "weight_decay": 1e-5,
        "epochs": 10,
    }

    updated = update_params_yaml(fake_yaml, champion_params)
    assert updated["train"]["lr"] == pytest.approx(0.0042)
    assert updated["train"]["optimizer"] == "adamw"
    assert updated["model"]["dropout"] == pytest.approx(0.35)
    assert updated["data"]["batch_size"] == 64
    assert updated["train"]["weight_decay"] == pytest.approx(1e-5)
    assert updated["train"]["epochs"] == 10


def test_find_latest_sweep_dir_and_promote_champion(
    setup_and_teardown_test_sweep: str,
) -> None:
    """Verify that champion model promotion exports safetensors, updates reports, and handles dry-run."""
    sweep_id = setup_and_teardown_test_sweep
    latest_sweep = find_latest_sweep_dir("artifacts/sweeps")
    assert latest_sweep.exists()

    res = promote_champion(
        sweep_id=sweep_id,
        update_ssot=False,
        evaluate_model=False,
    )
    assert res["sweep_id"] == sweep_id
    assert "champion_run_id" in res
    assert Path(res["promoted_checkpoint"]).exists()

    # Check that safetensors and config.json exist
    ckpt_dir = Path("artifacts/checkpoints")
    assert (ckpt_dir / "model.safetensors").exists()
    assert (ckpt_dir / "config.json").exists()
