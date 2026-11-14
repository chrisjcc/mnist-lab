#!/usr/bin/env python3
"""Execute a single hyperparameter sweep trial driven by Weights & Biases."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is available for module resolution
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import wandb  # noqa: E402
from mnist_pipeline.config import PipelineConfig, load_hydra_config  # noqa: E402
from mnist_pipeline.train import TrainingSummary, train_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_sweep_trial")

PARAM_MAPPING: dict[str, str] = {
    "learning_rate": "train.lr",
    "lr": "train.lr",
    "train.lr": "train.lr",
    "weight_decay": "train.weight_decay",
    "train.weight_decay": "train.weight_decay",
    "optimizer": "train.optimizer",
    "train.optimizer": "train.optimizer",
    "epochs": "train.epochs",
    "train.epochs": "train.epochs",
    "batch_size": "data.batch_size",
    "data.batch_size": "data.batch_size",
    "dropout": "model.dropout",
    "model.dropout": "model.dropout",
    "hidden_dim": "model.hidden_dim",
    "model.hidden_dim": "model.hidden_dim",
}


def build_hydra_overrides(
    sweep_params: dict[str, Any],
    smoke: bool = False,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    """Translate flat or aliased sweep parameter keys into Hydra dotlist overrides.

    Args:
        sweep_params: Dictionary of hyperparameters from W&B config or CLI.
        smoke: Whether to constrain trial to a fast 1-epoch smoke check.
        extra_overrides: Optional list of additional Hydra overrides.

    Returns:
        List of formatted Hydra CLI override strings (e.g. ['train.lr=0.001']).
    """
    overrides: list[str] = []
    for raw_key, value in sweep_params.items():
        if value is None:
            continue
        cleaned_key = raw_key.lstrip("-")
        target_key = PARAM_MAPPING.get(cleaned_key, cleaned_key)
        # Only process recognized configuration namespaces
        if target_key.startswith(("train.", "model.", "data.", "evaluate.", "wandb.", "paths.")):
            overrides.append(f"{target_key}={value}")

    if smoke:
        overrides.append("train.epochs=1")
        overrides.append("train.max_train_batches=2")
        overrides.append("train.max_val_batches=2")

    if extra_overrides:
        overrides.extend(extra_overrides)

    return overrides


def run_trial(
    cli_args: dict[str, Any] | None = None,
    smoke: bool = False,
    extra_overrides: list[str] | None = None,
) -> TrainingSummary:
    """Execute a single sweep trial and report metrics to Weights & Biases.

    Args:
        cli_args: Optional dictionary of parsed command-line arguments.
        smoke: Whether to run as a fast smoke check.
        extra_overrides: Optional list of raw Hydra overrides.

    Returns:
        TrainingSummary dataclass summarizing trial performance.
    """
    # 1. Initialize or attach to W&B run
    run = None
    if wandb.run is not None:
        run = wandb.run
    else:
        default_mode = "disabled" if smoke else "online"
        mode = os.environ.get("WANDB_MODE", default_mode).lower()
        if mode != "disabled":
            try:
                run = wandb.init(mode=mode, reinit=True)  # type: ignore[arg-type]
            except Exception as exc:
                logger.warning("Could not initialize W&B (%s); continuing in disabled mode.", exc)
                run = None

    wandb_config: dict[str, Any] = (
        dict(wandb.config) if (run is not None and wandb.config is not None) else {}
    )

    # Merge CLI arguments (CLI takes precedence if explicitly passed)
    merged_params: dict[str, Any] = {**wandb_config, **(cli_args or {})}

    # 2. Build Hydra overrides
    overrides = build_hydra_overrides(merged_params, smoke=smoke, extra_overrides=extra_overrides)

    # If W&B run is active, ensure pipeline config reflects active tracking
    run_mode = getattr(run, "mode", None)
    if run is not None and run_mode in ("online", "offline"):
        overrides.append(f"wandb.mode={run_mode}")

    logger.info("Composing PipelineConfig with sweep overrides: %s", overrides)
    pipeline_cfg: PipelineConfig = load_hydra_config(overrides=overrides)

    # 3. Execute training
    summary: TrainingSummary = train_model(pipeline_cfg)

    # 4. Log summary metrics to W&B
    if run is not None:
        run.summary["best_val_accuracy"] = summary.best_val_accuracy
        run.summary["best_val_loss"] = summary.best_val_loss
        run.summary["best_epoch"] = summary.best_epoch

    logger.info(
        "Trial completed: best_val_acc=%.4f (epoch %d), best_val_loss=%.4f",
        summary.best_val_accuracy,
        summary.best_epoch,
        summary.best_val_loss,
    )
    return summary


def parse_args(argv: list[str]) -> tuple[dict[str, Any], bool, list[str]]:
    """Parse dynamic command-line arguments passed by wandb agent or user.

    Args:
        argv: Command line arguments list (excluding script name).

    Returns:
        Tuple of (known_param_dict, is_smoke, remaining_overrides).
    """
    parser = argparse.ArgumentParser(description="Execute a single W&B sweep trial.")
    parser.add_argument("--smoke", action="store_true", help="Run fast 1-epoch smoke trial.")
    parser.add_argument("--stage", type=str, default=None, help="Stage name identifier.")

    # Known sweep hyperparameter flags
    for key in PARAM_MAPPING:
        flag = f"--{key.lstrip('-')}"
        parser.add_argument(flag, type=str, default=None)

    known, unknown = parser.parse_known_args(argv)
    cli_dict: dict[str, Any] = {}
    for k, v in vars(known).items():
        if v is not None and k not in ("smoke", "stage"):
            # Attempt numeric conversion
            try:
                cli_dict[k] = float(v) if "." in v or "e" in v.lower() else int(v)
            except ValueError:
                cli_dict[k] = v

    extra_overrides: list[str] = [arg for arg in unknown if "=" in arg]
    return cli_dict, known.smoke, extra_overrides


def main() -> None:
    """CLI entrypoint for running a sweep trial."""
    cli_dict, smoke, extra_overrides = parse_args(sys.argv[1:])
    run_trial(cli_args=cli_dict, smoke=smoke, extra_overrides=extra_overrides)


if __name__ == "__main__":
    main()
