"""Hugging Face Model Hub and Dataset Hub staging and release utilities."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from mnist_pipeline.config import PipelineConfig
from mnist_pipeline.model import load_checkpoint

LOGGER = logging.getLogger(__name__)


def stage_model_release(
    cfg: PipelineConfig,
    staging_dir: str | Path,
) -> dict[str, str]:
    """Assemble all Model Hub release artifacts into a clean local staging directory.

    Staged files:
      - `model.safetensors` & `config.json` (generated via `save_pretrained` if missing)
      - `checkpoint_best.pt`
      - `README.md` (copied from `<paths.docs_dir>/model_card.md`)
      - `test_metrics.json` & `confusion_matrix.csv` (from `<paths.evaluation_dir>`)
      - `benchmark_metrics.json` (from `<paths.benchmarks_dir>`)
      - `plots/*.png` (from `<paths.plots_dir>`)

    Args:
        cfg: Validated `PipelineConfig`.
        staging_dir: Target directory to populate with Hub release artifacts.

    Returns:
        Dictionary mapping relative staged filenames to their absolute paths.

    Raises:
        FileNotFoundError: If `checkpoint_best.pt` or `model.safetensors` cannot be found.
    """
    stage_root = Path(staging_dir)
    stage_root.mkdir(parents=True, exist_ok=True)
    staged_files: dict[str, str] = {}

    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_best = ckpt_dir / "checkpoint_best.pt"
    if not ckpt_best.exists():
        alt_ckpt = Path(cfg.evaluate.checkpoint_path)
        if alt_ckpt.exists():
            ckpt_best = alt_ckpt
        else:
            raise FileNotFoundError(
                f"Best checkpoint not found at '{ckpt_best}'. Run `make train` before staging."
            )

    safetensors_src = ckpt_best.parent / "model.safetensors"
    config_src = ckpt_best.parent / "config.json"
    if not safetensors_src.exists() or not config_src.exists():
        model, payload = load_checkpoint(ckpt_best, device="cpu")
        model.save_pretrained(
            stage_root,
            extra_metadata={
                "best_epoch": int(payload.get("epoch", 0)),
                "best_val_accuracy": float(payload.get("best_val_accuracy", 0.0)),
            },
        )
        staged_files["model.safetensors"] = str(stage_root / "model.safetensors")
        staged_files["config.json"] = str(stage_root / "config.json")
    else:
        shutil.copy2(safetensors_src, stage_root / "model.safetensors")
        shutil.copy2(config_src, stage_root / "config.json")
        staged_files["model.safetensors"] = str(stage_root / "model.safetensors")
        staged_files["config.json"] = str(stage_root / "config.json")

    shutil.copy2(ckpt_best, stage_root / "checkpoint_best.pt")
    staged_files["checkpoint_best.pt"] = str(stage_root / "checkpoint_best.pt")

    model_card_path = Path(cfg.paths.docs_dir) / "model_card.md"
    if model_card_path.exists():
        shutil.copy2(model_card_path, stage_root / "README.md")
        staged_files["README.md"] = str(stage_root / "README.md")

    eval_dir = Path(cfg.paths.evaluation_dir)
    for fname in ("test_metrics.json", "confusion_matrix.csv", "classification_report.txt"):
        src = eval_dir / fname
        if src.exists():
            shutil.copy2(src, stage_root / fname)
            staged_files[fname] = str(stage_root / fname)

    bench_src = Path(cfg.paths.benchmarks_dir) / "benchmark_metrics.json"
    if bench_src.exists():
        shutil.copy2(bench_src, stage_root / "benchmark_metrics.json")
        staged_files["benchmark_metrics.json"] = str(stage_root / "benchmark_metrics.json")

    plots_dir = Path(cfg.paths.plots_dir)
    if plots_dir.exists():
        stage_plots = stage_root / "plots"
        stage_plots.mkdir(parents=True, exist_ok=True)
        for png_file in sorted(plots_dir.glob("*.png")):
            dest = stage_plots / png_file.name
            shutil.copy2(png_file, dest)
            staged_files[f"plots/{png_file.name}"] = str(dest)

    return staged_files


def stage_dataset_release(
    cfg: PipelineConfig,
    staging_dir: str | Path,
) -> dict[str, str]:
    """Stage the Dataset Card (`README.md`) and `split_metadata.json` for Dataset Hub upload.

    Args:
        cfg: Validated `PipelineConfig`.
        staging_dir: Target directory to populate.

    Returns:
        Dictionary mapping relative staged filenames to their absolute paths.
    """
    stage_root = Path(staging_dir)
    stage_root.mkdir(parents=True, exist_ok=True)
    staged_files: dict[str, str] = {}

    dataset_card_path = Path(cfg.paths.docs_dir) / "dataset_card.md"
    if dataset_card_path.exists():
        shutil.copy2(dataset_card_path, stage_root / "README.md")
        staged_files["README.md"] = str(stage_root / "README.md")

    split_meta_path = Path(cfg.data.processed_dir) / "split_metadata.json"
    if split_meta_path.exists():
        shutil.copy2(split_meta_path, stage_root / "split_metadata.json")
        staged_files["split_metadata.json"] = str(stage_root / "split_metadata.json")

    return staged_files


def upload_to_hf_hub(
    cfg: PipelineConfig,
    token: str | None = None,
    upload_dataset_card: bool = True,
    staging_root: str | Path | None = None,
) -> dict[str, Any]:
    """Upload staged model and dataset artifacts to the Hugging Face Hub.

    When `cfg.hf.dry_run=True`, stages all files locally and returns the manifest
    without performing any network calls.

    Args:
        cfg: Validated `PipelineConfig`.
        token: Optional Hugging Face API token (defaults to `HF_TOKEN` env var).
        upload_dataset_card: Whether to also stage/upload the dataset card repository.
        staging_root: Optional directory to persist staged payloads.

    Returns:
        Summary dictionary describing staged files and Hub repository URLs.
    """
    resolved_token = token or os.environ.get("HF_TOKEN")
    cleanup_dir: tempfile.TemporaryDirectory[str] | None = None

    if staging_root is not None:
        base_dir = Path(staging_root)
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        cleanup_dir = tempfile.TemporaryDirectory(prefix="mnist_hf_stage_")
        base_dir = Path(cleanup_dir.name)

    try:
        model_stage_dir = base_dir / "model_repo"
        dataset_stage_dir = base_dir / "dataset_repo"

        staged_model = stage_model_release(cfg, model_stage_dir)
        staged_dataset = (
            stage_dataset_release(cfg, dataset_stage_dir) if upload_dataset_card else {}
        )

        summary: dict[str, Any] = {
            "dry_run": cfg.hf.dry_run,
            "model_repo_id": cfg.hf.repo_id,
            "dataset_repo_id": cfg.hf.dataset_repo_id,
            "revision": cfg.hf.revision,
            "staged_model_files": sorted(staged_model.keys()),
            "staged_dataset_files": sorted(staged_dataset.keys()),
            "model_url": f"https://huggingface.co/{cfg.hf.repo_id}",
            "dataset_url": f"https://huggingface.co/datasets/{cfg.hf.dataset_repo_id}",
        }

        if cfg.hf.dry_run:
            LOGGER.info(
                "HF Hub dry_run=True: staged %d model files for '%s' and %d dataset files for '%s'.",
                len(staged_model),
                cfg.hf.repo_id,
                len(staged_dataset),
                cfg.hf.dataset_repo_id,
            )
            return summary

        if not resolved_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is required when hf.dry_run=False. "
                "Set HF_TOKEN or pass `hf.dry_run=true` to test staging locally."
            )

        api = HfApi(token=resolved_token)
        api.create_repo(
            repo_id=cfg.hf.repo_id,
            repo_type="model",
            private=cfg.hf.private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=cfg.hf.repo_id,
            repo_type="model",
            folder_path=str(model_stage_dir),
            revision=cfg.hf.revision,
            commit_message="Upload trained MNISTConvNet weights, metrics, plots, and Model Card",
        )
        LOGGER.info("Uploaded model artifacts to https://huggingface.co/%s", cfg.hf.repo_id)

        if upload_dataset_card and staged_dataset:
            api.create_repo(
                repo_id=cfg.hf.dataset_repo_id,
                repo_type="dataset",
                private=cfg.hf.private,
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=cfg.hf.dataset_repo_id,
                repo_type="dataset",
                folder_path=str(dataset_stage_dir),
                revision=cfg.hf.revision,
                commit_message="Upload MNIST stratified split metadata and Dataset Card",
            )
            LOGGER.info(
                "Uploaded dataset card to https://huggingface.co/datasets/%s",
                cfg.hf.dataset_repo_id,
            )

        return summary
    finally:
        if cleanup_dir is not None:
            cleanup_dir.cleanup()
