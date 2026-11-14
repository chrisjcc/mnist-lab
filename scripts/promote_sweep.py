#!/usr/bin/env python3
"""CLI utility to generate sweep reports, promote champion models, and release to Hugging Face."""

from __future__ import annotations

import argparse
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Any

import yaml

from mnist_pipeline.config import PipelineConfig, load_hydra_config
from mnist_pipeline.evaluate import benchmark_checkpoint, evaluate_checkpoint
from mnist_pipeline.hf_hub import upload_to_hf_hub
from mnist_pipeline.model import load_checkpoint
from mnist_pipeline.reporting import generate_all_reports, generate_sweep_report
from mnist_pipeline.utils import load_json, resolve_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("promote_sweep")


def find_latest_sweep_dir(sweeps_dir: str | Path) -> Path:
    """Find the most recently modified sweep directory containing a trials ledger.

    Args:
        sweeps_dir: Root sweeps directory path.

    Returns:
        Path to the latest sweep directory.

    Raises:
        FileNotFoundError: If no sweep directories containing trials.json are found.
    """
    root = Path(sweeps_dir)
    if not root.exists():
        raise FileNotFoundError(f"Sweeps directory '{root}' does not exist.")
    candidates = [d for d in root.iterdir() if d.is_dir() and (d / "trials.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No sweep directories with 'trials.json' found in '{root}'.")
    candidates.sort(key=lambda p: (p / "trials.json").stat().st_mtime, reverse=True)
    return candidates[0]


def update_params_yaml(
    params_yaml_path: str | Path,
    champion_params: dict[str, Any],
) -> dict[str, Any]:
    """Update canonical params.yaml Single Source of Truth with champion hyperparameters.

    Args:
        params_yaml_path: Path to canonical params.yaml file.
        champion_params: Hyperparameter dictionary from the champion trial.

    Returns:
        Updated parameters dictionary.
    """
    path = Path(params_yaml_path)
    if not path.exists():
        LOGGER.warning("params.yaml not found at '%s'; skipping SSoT update.", path)
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("train", {})
    data.setdefault("model", {})
    data.setdefault("data", {})

    # Map champion parameters
    if "learning_rate" in champion_params or "train.lr" in champion_params:
        val = champion_params.get("learning_rate", champion_params.get("train.lr"))
        data["train"]["lr"] = float(val)
    if "optimizer" in champion_params or "train.optimizer" in champion_params:
        val = champion_params.get("optimizer", champion_params.get("train.optimizer"))
        data["train"]["optimizer"] = str(val)
    if "weight_decay" in champion_params or "train.weight_decay" in champion_params:
        val = champion_params.get("weight_decay", champion_params.get("train.weight_decay"))
        data["train"]["weight_decay"] = float(val)
    if "dropout" in champion_params or "model.dropout" in champion_params:
        val = champion_params.get("dropout", champion_params.get("model.dropout"))
        data["model"]["dropout"] = float(val)
    if "batch_size" in champion_params or "data.batch_size" in champion_params:
        val = champion_params.get("batch_size", champion_params.get("data.batch_size"))
        data["data"]["batch_size"] = int(val)
    if "epochs" in champion_params or "train.epochs" in champion_params:
        val = champion_params.get("epochs", champion_params.get("train.epochs"))
        data["train"]["epochs"] = int(val)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    LOGGER.info("Updated params.yaml with champion parameters: %s", path)
    return data


def promote_champion(
    sweep_id: str | None = None,
    run_id: str | None = None,
    sweeps_dir: str | Path = "artifacts/sweeps",
    update_ssot: bool = True,
    evaluate_model: bool = True,
    publish_hf: bool = False,
    dry_run: bool = False,
    model_repo: str | None = None,
    dataset_repo: str | None = None,
) -> dict[str, Any]:
    """Promote the best performing trial from a sweep into canonical production artifacts.

    Args:
        sweep_id: Target sweep ID. If None, auto-resolves the latest sweep.
        run_id: Specific run ID to promote. If None, auto-selects the top validation performer.
        sweeps_dir: Base directory containing per-sweep trials.
        update_ssot: If True, writes champion hyperparameters back to `params.yaml`.
        evaluate_model: If True, runs full test evaluation, benchmark, and regenerates model card.
        publish_hf: If True, stages and releases model and cards to Hugging Face Hub.
        dry_run: If True, stages HF Hub release locally without remote network push.
        model_repo: Optional override for HF model repository ID.
        dataset_repo: Optional override for HF dataset repository ID.

    Returns:
        Summary dictionary containing champion metadata, promoted paths, and release details.

    Raises:
        FileNotFoundError: If the champion trial or checkpoint does not exist.
        ValueError: If no trials are present in the sweep ledger.
    """
    root_sweeps = Path(sweeps_dir)
    if sweep_id is None:
        sweep_dir = find_latest_sweep_dir(root_sweeps)
        resolved_sweep_id = sweep_dir.name
    else:
        resolved_sweep_id = sweep_id
        sweep_dir = root_sweeps / resolved_sweep_id

    trials_file = sweep_dir / "trials.json"
    if not trials_file.exists():
        with contextlib.suppress(Exception):
            generate_sweep_report(sweep_id=resolved_sweep_id, sweeps_dir=root_sweeps)

    if not trials_file.exists():
        raise FileNotFoundError(f"Trials ledger '{trials_file}' does not exist.")

    trials_raw = load_json(trials_file)
    trials_data = trials_raw.get("trials", []) if isinstance(trials_raw, dict) else trials_raw
    if not isinstance(trials_data, list) or not trials_data:
        raise ValueError(f"No trials found in '{trials_file}'.")

    # Select champion
    if run_id is not None:
        matching = [t for t in trials_data if t.get("run_id") == run_id]
        if not matching:
            raise FileNotFoundError(f"Run ID '{run_id}' not found in sweep '{resolved_sweep_id}'.")
        champion = matching[0]
    else:
        # Sort by validation accuracy descending, tie-break by validation loss ascending
        def sort_key(t: dict[str, Any]) -> tuple[float, float]:
            val_acc = float(t.get("best_val_accuracy", 0.0))
            val_loss = float(t.get("best_val_loss", 999.0))
            return (-val_acc, val_loss)

        ranked = sorted(trials_data, key=sort_key)
        champion = ranked[0]

    champion_run_id = champion["run_id"]
    champion_run_name = champion.get("run_name", champion_run_id)
    champion_params = champion.get("params", {})
    LOGGER.info(
        "Promoting champion model from sweep '%s': %s (ID: %s, Best Val Acc: %.4f)",
        resolved_sweep_id,
        champion_run_name,
        champion_run_id,
        float(champion.get("best_val_accuracy", 0.0)),
    )

    # 1. Promote Checkpoint
    repo_root = Path(__file__).resolve().parent.parent
    trial_ckpt = Path(
        champion.get("checkpoint_path", sweep_dir / champion_run_id / "checkpoint_best.pt")
    )
    if not trial_ckpt.exists():
        fallback_ckpt = repo_root / "artifacts" / "checkpoints" / "checkpoint_best.pt"
        if fallback_ckpt.exists():
            LOGGER.warning(
                "Trial checkpoint not found at '%s'; falling back to existing canonical checkpoint '%s'.",
                trial_ckpt,
                fallback_ckpt,
            )
            trial_ckpt = fallback_ckpt
        else:
            raise FileNotFoundError(f"Champion checkpoint not found at '{trial_ckpt}'.")

    canonical_ckpt_dir = repo_root / "artifacts" / "checkpoints"
    canonical_ckpt_dir.mkdir(parents=True, exist_ok=True)
    canonical_ckpt = canonical_ckpt_dir / "checkpoint_best.pt"
    shutil.copy2(trial_ckpt, canonical_ckpt)
    LOGGER.info("Promoted checkpoint -> %s", canonical_ckpt)

    # Promote train history if available
    trial_hist = Path(
        champion.get("history_path", sweep_dir / champion_run_id / "train_history.csv")
    )
    if trial_hist.exists():
        canon_reports_dir = repo_root / "artifacts" / "reports"
        canon_reports_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trial_hist, canon_reports_dir / "train_history.csv")
        LOGGER.info("Promoted train history -> %s", canon_reports_dir / "train_history.csv")

    # 2. Update params.yaml SSoT
    if update_ssot:
        params_file = repo_root / "params.yaml"
        update_params_yaml(params_file, champion_params)

    # 3. Export safetensors and config.json for pretrained loading
    dev = resolve_device("cpu")
    model, _ = load_checkpoint(canonical_ckpt, device=dev)
    model.save_pretrained(canonical_ckpt_dir)
    LOGGER.info("Exported model.safetensors and config.json in %s", canonical_ckpt_dir)

    # 4. Generate local sweep report
    report_file = generate_sweep_report(
        sweep_id=resolved_sweep_id,
        sweeps_dir=root_sweeps,
        output_path=repo_root / "artifacts" / "reports" / "sweep_report.md",
    )

    result_summary: dict[str, Any] = {
        "sweep_id": resolved_sweep_id,
        "champion_run_id": champion_run_id,
        "champion_run_name": champion_run_name,
        "champion_params": champion_params,
        "best_val_accuracy": champion.get("best_val_accuracy"),
        "best_val_loss": champion.get("best_val_loss"),
        "promoted_checkpoint": str(canonical_ckpt),
        "sweep_report": str(report_file),
    }

    # 5. Evaluate and benchmark champion model
    if evaluate_model:
        LOGGER.info("Running held-out test evaluation on promoted champion...")
        overrides: list[str] = []
        if model_repo:
            overrides.append(f"hf.repo_id={model_repo}")
        if dataset_repo:
            overrides.append(f"hf.dataset_repo_id={dataset_repo}")
        if dry_run or not publish_hf:
            overrides.append("hf.dry_run=true")

        cfg: PipelineConfig = load_hydra_config(overrides=overrides)
        test_summary = evaluate_checkpoint(cfg)
        LOGGER.info(
            "Champion test evaluation: test_accuracy=%.4f", test_summary.get("test_accuracy", 0.0)
        )
        bench_summary = benchmark_checkpoint(cfg)
        LOGGER.info("Champion latency benchmark complete.")

        report_outputs = generate_all_reports(
            cfg,
            template_dir=repo_root / "templates",
            docs_dir=repo_root / "docs",
        )
        LOGGER.info("Regenerated docs/model_card.md -> %s", report_outputs.get("model_card"))
        result_summary["test_metrics"] = test_summary
        result_summary["benchmark_metrics"] = bench_summary
        result_summary["model_card"] = report_outputs.get("model_card")

        # 6. Publish to Hugging Face Hub
        if publish_hf:
            LOGGER.info("Publishing promoted champion to Hugging Face Hub (dry_run=%s)...", dry_run)
            hf_res = upload_to_hf_hub(cfg)
            result_summary["hf_hub_release"] = hf_res
            LOGGER.info("Hugging Face release complete: %s", hf_res)

    return result_summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for sweep promotion utility."""
    parser = argparse.ArgumentParser(
        description="Generate sweep report and promote champion model to production and HF Hub."
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="Target sweep ID. If omitted, the latest sweep is selected automatically.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional specific trial run ID to promote instead of the top performer.",
    )
    parser.add_argument(
        "--sweeps-dir",
        type=str,
        default="artifacts/sweeps",
        help="Path to directory containing sweep trial ledgers.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate or refresh the sweep Markdown report without promoting models.",
    )
    parser.add_argument(
        "--update-ssot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update params.yaml with champion hyperparameters (default: true).",
    )
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run held-out evaluation, benchmark, and update model_card.md (default: true).",
    )
    parser.add_argument(
        "--publish-hf",
        action="store_true",
        help="Publish promoted champion model and cards to Hugging Face Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage Hugging Face release payloads locally without making remote uploads.",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=None,
        help="Hugging Face model repository ID override.",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=None,
        help="Hugging Face dataset repository ID override.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for promote_sweep script."""
    args = parse_args()

    if args.report_only:
        report_path = generate_sweep_report(
            sweep_id=args.sweep_id,
            sweeps_dir=args.sweeps_dir,
        )
        LOGGER.info("Report generation complete: %s", report_path)
        return

    summary = promote_champion(
        sweep_id=args.sweep_id,
        run_id=args.run_id,
        sweeps_dir=args.sweeps_dir,
        update_ssot=args.update_ssot,
        evaluate_model=args.evaluate,
        publish_hf=args.publish_hf,
        dry_run=args.dry_run,
        model_repo=args.model_repo,
        dataset_repo=args.dataset_repo,
    )
    LOGGER.info(
        "Successfully promoted champion run '%s' from sweep '%s' (val_acc=%.4f).",
        summary["champion_run_id"],
        summary["sweep_id"],
        float(summary.get("best_val_accuracy", 0.0)),
    )


if __name__ == "__main__":
    main()
