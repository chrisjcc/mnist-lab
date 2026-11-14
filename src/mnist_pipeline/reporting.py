"""Automated visualization, evaluation report, and Model/Dataset Card generation."""

from __future__ import annotations

import csv
import datetime
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from jinja2 import Template

from mnist_pipeline.config import PipelineConfig
from mnist_pipeline.data import load_processed_split
from mnist_pipeline.model import load_checkpoint
from mnist_pipeline.utils import load_json, save_json

LOGGER = logging.getLogger(__name__)


def _read_train_history(history_csv_path: str | Path) -> list[dict[str, float]]:
    """Read epoch-by-epoch training history from CSV.

    Args:
        history_csv_path: Path to `train_history.csv`.

    Returns:
        List of row dictionaries with float values.
    """
    path = Path(history_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Training history CSV not found at {path}")
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if k is not None and v is not None})
    return rows


def plot_training_curves(
    history_csv_path: str | Path,
    loss_plot_path: str | Path,
    acc_plot_path: str | Path,
) -> tuple[Path, Path]:
    """Generate publication-quality training and validation loss/accuracy curves.

    Args:
        history_csv_path: Path to `train_history.csv`.
        loss_plot_path: Output path for `loss_curve.png`.
        acc_plot_path: Output path for `accuracy_curve.png`.

    Returns:
        Tuple of `(loss_plot_path, acc_plot_path)` as resolved `Path` objects.
    """
    rows = _read_train_history(history_csv_path)
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_acc = [r.get("train_accuracy", r.get("train_acc", 0.0)) * 100.0 for r in rows]
    val_acc = [r.get("val_accuracy", r.get("val_acc", 0.0)) * 100.0 for r in rows]

    loss_out = Path(loss_plot_path)
    acc_out = Path(acc_plot_path)
    loss_out.parent.mkdir(parents=True, exist_ok=True)
    acc_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.plot(epochs, train_loss, marker="o", linewidth=2.2, color="#1f77b4", label="Train Loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2.2, color="#d62728", label="Validation Loss")
    ax.set_title("MNISTConvNet Training vs. Validation Cross-Entropy Loss", fontsize=12, pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(loss_out)
    plt.close(fig)

    # 2. Accuracy curve
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    ax.plot(
        epochs,
        train_acc,
        marker="o",
        linewidth=2.2,
        color="#2ca02c",
        label="Train Accuracy (%)",
    )
    ax.plot(
        epochs,
        val_acc,
        marker="s",
        linewidth=2.2,
        color="#ff7f0e",
        label="Validation Accuracy (%)",
    )
    ax.set_title("MNISTConvNet Training vs. Validation Accuracy", fontsize=12, pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(acc_out)
    plt.close(fig)

    return loss_out, acc_out


def plot_confusion_matrix(
    confusion_csv_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Render an annotated 10x10 heatmap of the test set confusion matrix.

    Args:
        confusion_csv_path: Path to `confusion_matrix.csv`.
        output_path: Destination PNG path (`confusion_matrix.png`).

    Returns:
        Path to the saved confusion matrix figure.
    """
    csv_path = Path(confusion_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Confusion matrix CSV not found at {csv_path}")

    matrix_rows: list[list[int]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        _ = next(reader, None)  # skip header
        for row in reader:
            if len(row) > 1:
                matrix_rows.append([int(x) for x in row[1:]])

    cm = np.asarray(matrix_rows, dtype=np.int64)
    num_classes = int(cm.shape[0])

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.8), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(num_classes)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_yticklabels([str(i) for i in range(num_classes)])
    ax.set_xlabel("Predicted Digit Label", fontsize=11)
    ax.set_ylabel("True Digit Label", fontsize=11)
    ax.set_title("MNIST Held-Out Test Set Confusion Matrix (10x10)", fontsize=12, pad=12)

    thresh = float(np.max(cm)) / 2.0 if cm.size > 0 else 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = int(cm[i, j])
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_sample_predictions(
    model: nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    output_path: str | Path,
    mean: float = 0.1307,
    std: float = 0.3081,
    num_samples: int = 16,
) -> Path:
    """Plot a 4x4 grid of test digits annotated with predicted label and confidence.

    Args:
        model: Trained PyTorch model in eval mode.
        test_images: Normalized test image tensor `(N, 1, 28, 28)`.
        test_labels: Ground-truth test label tensor `(N,)`.
        output_path: Destination PNG path (`sample_predictions.png`).
        mean: Normalization mean used to unnormalize display pixels.
        std: Normalization std used to unnormalize display pixels.
        num_samples: Number of samples to display (default 16 for 4x4 grid).

    Returns:
        Path to the saved sample prediction grid figure.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = min(num_samples, int(test_images.shape[0]))
    images_subset = test_images[:count]
    labels_subset = test_labels[:count]

    model_device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        logits = model(images_subset.to(model_device))
        probs = torch.softmax(logits, dim=1).cpu()
        confs, preds = probs.max(dim=1)

    grid_size = int(np.ceil(np.sqrt(count)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8.5, 8.5), dpi=150)
    axes_flat = np.asarray(axes).reshape(-1)

    for idx, ax in enumerate(axes_flat):
        if idx >= count:
            ax.axis("off")
            continue
        img_norm = images_subset[idx, 0].cpu().numpy()
        img_unnorm = np.clip(img_norm * std + mean, 0.0, 1.0)
        true_lbl = int(labels_subset[idx].item())
        pred_lbl = int(preds[idx].item())
        conf = float(confs[idx].item()) * 100.0

        ax.imshow(img_unnorm, cmap="gray", vmin=0.0, vmax=1.0)
        is_correct = true_lbl == pred_lbl
        title_color = "#1b7837" if is_correct else "#d73027"
        ax.set_title(
            f"Pred: {pred_lbl} ({conf:.1f}%)\nTrue: {true_lbl}",
            fontsize=8.5,
            color=title_color,
            fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle("MNIST Held-Out Test Set Sample Predictions", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_evaluation_report(
    cfg: PipelineConfig,
    train_history_path: str | Path,
    test_metrics_path: str | Path,
    benchmark_metrics_path: str | Path,
    env_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Compile a comprehensive Markdown evaluation and benchmark report.

    Args:
        cfg: Validated `PipelineConfig`.
        train_history_path: Path to `train_history.csv`.
        test_metrics_path: Path to `test_metrics.json`.
        benchmark_metrics_path: Path to `benchmark_metrics.json`.
        env_path: Path to `environment.json`.
        output_path: Destination path for `evaluation_report.md`.

    Returns:
        Resolved `Path` to the generated Markdown report.
    """
    history_rows = _read_train_history(train_history_path)
    test_metrics = load_json(test_metrics_path)
    benchmark_metrics = load_json(benchmark_metrics_path)
    env_meta = load_json(env_path) if Path(env_path).exists() else {}

    lines: list[str] = [
        "# MNIST Classification Pipeline — Evaluation & Benchmark Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Test Accuracy**: **{test_metrics['test_acc'] * 100:.2f}%** (`{test_metrics['test_acc']:.4f}`)",
        f"- **Macro F1-Score**: `{test_metrics['macro_f1']:.4f}`",
        f"- **Test Cross-Entropy Loss**: `{test_metrics['test_loss']:.4f}`",
        f"- **Evaluated Test Samples**: `{test_metrics['num_samples']:,}`",
        f"- **Trainable Parameters**: `{benchmark_metrics['trainable_params']:,}` (`{benchmark_metrics['checkpoint_size_mb']:.2f} MB`)",
        f"- **Execution Device**: `{benchmark_metrics['device']}`",
        f"- **Global Seed**: `{cfg.train.seed}` (`deterministic={cfg.train.deterministic}`)",
        "",
        "---",
        "",
        "## 2. Training Progression Summary",
        "",
        "| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate | Epoch Time (s) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for row in history_rows:
        t_acc = row.get("train_accuracy", row.get("train_acc", 0.0))
        v_acc = row.get("val_accuracy", row.get("val_acc", 0.0))
        lines.append(
            f"| {int(row['epoch'])} | {row['train_loss']:.4f} | {t_acc * 100:.2f}% | "
            f"{row['val_loss']:.4f} | {v_acc * 100:.2f}% | {row['lr']:.6f} | "
            f"{row['epoch_time_sec']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Training & Validation Curves",
            "",
            "| Training vs. Validation Loss | Training vs. Validation Accuracy |",
            "| :---: | :---: |",
            "| ![Loss Curve](plots/loss_curve.png) | ![Accuracy Curve](plots/accuracy_curve.png) |",
            "",
            "---",
            "",
            "## 3. Held-Out Test Set Classification Metrics",
            "",
            "| Metric | Value |",
            "| :--- | :---: |",
            f"| **Overall Accuracy** | `{test_metrics['test_acc']:.4f}` ({test_metrics['test_acc'] * 100:.2f}%) |",
            f"| **Cross-Entropy Loss** | `{test_metrics['test_loss']:.4f}` |",
            f"| **Macro Precision** | `{test_metrics['macro_precision']:.4f}` |",
            f"| **Macro Recall** | `{test_metrics['macro_recall']:.4f}` |",
            f"| **Macro F1-Score** | `{test_metrics['macro_f1']:.4f}` |",
            f"| **Weighted Precision** | `{test_metrics['weighted_precision']:.4f}` |",
            f"| **Weighted Recall** | `{test_metrics['weighted_recall']:.4f}` |",
            f"| **Weighted F1-Score** | `{test_metrics['weighted_f1']:.4f}` |",
            "",
            "### Per-Class Precision, Recall & F1-Score",
            "",
            "| Digit Class | Precision | Recall | F1-Score | Support |",
            "| :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for digit, stats in test_metrics["per_class"].items():
        lines.append(
            f"| **Digit {digit}** | {stats['precision']:.4f} | {stats['recall']:.4f} | "
            f"{stats['f1']:.4f} | {stats['support']} |"
        )

    lines.extend(
        [
            "",
            "### Confusion Matrix & Sample Predictions",
            "",
            "| Held-Out Test Set Confusion Matrix (10x10) | Sample Predictions on Test Digits (4x4) |",
            "| :---: | :---: |",
            "| ![Confusion Matrix](plots/confusion_matrix.png) | ![Sample Predictions](plots/sample_predictions.png) |",
            "",
            "---",
            "",
            "## 4. Inference Latency & Throughput Benchmarks",
            "",
            "| Batch Size | Mean Latency (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (samples/sec) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for bs, bm in benchmark_metrics["batch_metrics"].items():
        lines.append(
            f"| **{bs}** | {bm['mean_ms']:.3f} | {bm['std_ms']:.3f} | {bm['p50_ms']:.3f} | "
            f"{bm['p95_ms']:.3f} | {bm['p99_ms']:.3f} | {bm['samples_per_sec']:.1f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Runtime Environment Metadata",
            "",
            f"- **Python**: `{env_meta.get('python_version', 'N/A')}`",
            f"- **PyTorch**: `{env_meta.get('torch_version', 'N/A')}`",
            f"- **Platform**: `{env_meta.get('platform', 'N/A')}`",
            f"- **Timestamp (UTC)**: `{env_meta.get('timestamp_utc', 'N/A')}`",
            "",
        ]
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def render_model_card(
    cfg: PipelineConfig,
    test_metrics: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    env_meta: dict[str, Any],
    template_path: str | Path = "templates/model_card_template.md",
    output_path: str | Path = "docs/model_card.md",
) -> Path:
    """Render `docs/model_card.md` from `templates/model_card_template.md`.

    Args:
        cfg: Validated `PipelineConfig`.
        test_metrics: Loaded dictionary from `test_metrics.json`.
        benchmark_metrics: Loaded dictionary from `benchmark_metrics.json`.
        env_meta: Loaded dictionary from `environment.json`.
        template_path: Path to the Jinja2 model card template.
        output_path: Destination path for the rendered model card.

    Returns:
        Path to the rendered model card file.
    """
    tpl_text = Path(template_path).read_text(encoding="utf-8")
    template = Template(tpl_text)
    rendered = template.render(
        hf_model_repo_id=cfg.hf.repo_id,
        hf_dataset_repo_id=cfg.hf.dataset_repo_id,
        seed=cfg.train.seed,
        deterministic=cfg.train.deterministic,
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        evaluate_cfg=cfg.evaluate,
        test_metrics=test_metrics,
        benchmark_metrics=benchmark_metrics,
        env_meta=env_meta,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def render_dataset_card(
    cfg: PipelineConfig,
    split_meta: dict[str, Any],
    template_path: str | Path = "templates/dataset_card_template.md",
    output_path: str | Path = "docs/dataset_card.md",
) -> Path:
    """Render `docs/dataset_card.md` from `templates/dataset_card_template.md`.

    Args:
        cfg: Validated `PipelineConfig`.
        split_meta: Loaded dictionary from `data/processed/split_metadata.json`.
        template_path: Path to the Jinja2 dataset card template.
        output_path: Destination path for the rendered dataset card.

    Returns:
        Path to the rendered dataset card file.
    """
    tpl_text = Path(template_path).read_text(encoding="utf-8")
    template = Template(tpl_text)
    rendered = template.render(
        hf_dataset_repo_id=cfg.hf.dataset_repo_id,
        split_meta=split_meta,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


def generate_all_reports(
    cfg: PipelineConfig,
    template_dir: str | Path | None = None,
    docs_dir: str | Path | None = None,
) -> dict[str, str]:
    """Generate all diagnostic plots, evaluation report, and Model/Dataset cards.

    Args:
        cfg: Validated `PipelineConfig`.
        template_dir: Optional directory containing Jinja2 card templates.
        docs_dir: Optional directory where `model_card.md` and `dataset_card.md` are written.

    Returns:
        Dictionary mapping artifact names to their generated file paths.
    """
    report_dir = Path(cfg.paths.reports_dir)
    plots_dir = Path(cfg.paths.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    history_csv = report_dir / "train_history.csv"
    env_json = report_dir / "environment.json"
    eval_dir = Path(cfg.paths.evaluation_dir)
    test_metrics_json = eval_dir / "test_metrics.json"
    cm_csv = eval_dir / "confusion_matrix.csv"
    benchmark_json = Path(cfg.paths.benchmarks_dir) / "benchmark_metrics.json"
    split_meta_json = Path(cfg.data.processed_dir) / "split_metadata.json"

    loss_plot, acc_plot = plot_training_curves(
        history_csv_path=history_csv,
        loss_plot_path=plots_dir / "loss_curve.png",
        acc_plot_path=plots_dir / "accuracy_curve.png",
    )
    cm_plot = plot_confusion_matrix(
        confusion_csv_path=cm_csv,
        output_path=plots_dir / "confusion_matrix.png",
    )

    ckpt_best = Path(cfg.paths.checkpoints_dir) / "checkpoint_best.pt"
    ckpt_path = ckpt_best if ckpt_best.exists() else Path(cfg.evaluate.checkpoint_path)
    model, _ = load_checkpoint(ckpt_path, device="cpu")
    test_images, test_labels = load_processed_split(cfg.data.processed_dir, "test")

    sample_plot = plot_sample_predictions(
        model=model,
        test_images=test_images,
        test_labels=test_labels,
        output_path=plots_dir / "sample_predictions.png",
        mean=cfg.data.mean,
        std=cfg.data.std,
        num_samples=16,
    )

    eval_report = generate_evaluation_report(
        cfg=cfg,
        train_history_path=history_csv,
        test_metrics_path=test_metrics_json,
        benchmark_metrics_path=benchmark_json,
        env_path=env_json,
        output_path=report_dir / "evaluation_report.md",
    )

    test_metrics = load_json(test_metrics_json)
    benchmark_metrics = load_json(benchmark_json)
    env_meta = load_json(env_json) if env_json.exists() else {}
    split_meta = load_json(split_meta_json)

    tpl_root = Path(template_dir) if template_dir is not None else Path(cfg.paths.templates_dir)
    docs_root = Path(docs_dir) if docs_dir is not None else Path(cfg.paths.docs_dir)

    model_card = render_model_card(
        cfg=cfg,
        test_metrics=test_metrics,
        benchmark_metrics=benchmark_metrics,
        env_meta=env_meta,
        template_path=tpl_root / "model_card_template.md",
        output_path=docs_root / "model_card.md",
    )
    dataset_card = render_dataset_card(
        cfg=cfg,
        split_meta=split_meta,
        template_path=tpl_root / "dataset_card_template.md",
        output_path=docs_root / "dataset_card.md",
    )

    LOGGER.info("All evaluation plots, Markdown report, and Model/Dataset cards generated.")
    return {
        "loss_curve": str(loss_plot),
        "accuracy_curve": str(acc_plot),
        "confusion_matrix": str(cm_plot),
        "sample_predictions": str(sample_plot),
        "evaluation_report": str(eval_report),
        "model_card": str(model_card),
        "dataset_card": str(dataset_card),
    }


def generate_sweep_report(
    sweep_id: str | None = None,
    sweeps_dir: str | Path = "artifacts/sweeps",
    output_path: str | Path | None = None,
    project_name: str = "mnist-classification",
    entity: str | None = None,
) -> Path:
    """Generate a Markdown sweep evaluation report comparing all trials.

    Reads trial results from `<sweeps_dir>/<sweep_id>/trials.json`, ranks runs
    by validation accuracy (and validation loss), highlights the champion model,
    and formats a leaderboard table.

    Args:
        sweep_id: W&B sweep ID or local sweep identifier. If None, resolves the
            most recently modified sweep in `sweeps_dir`.
        sweeps_dir: Directory containing per-sweep trial runs and ledgers.
        output_path: Path where the report should be saved. Defaults to
            `artifacts/reports/sweep_report.md` (and also writes to
            `<sweeps_dir>/<sweep_id>/sweep_report.md`).
        project_name: W&B project name for URL attribution.
        entity: W&B entity/organization for URL attribution.

    Returns:
        Path to the generated report file.

    Raises:
        FileNotFoundError: If no sweep directories or trials are found.
        ValueError: If `trials.json` contains no trial records.
    """
    root_sweeps = Path(sweeps_dir)
    resolved_sweep_id: str | None = sweep_id

    if resolved_sweep_id is None:
        if not root_sweeps.exists():
            raise FileNotFoundError(
                f"Sweeps directory '{root_sweeps}' does not exist. Run `make sweep` first."
            )
        candidate_dirs = [
            d for d in root_sweeps.iterdir() if d.is_dir() and (d / "trials.json").exists()
        ]
        if not candidate_dirs:
            raise FileNotFoundError(
                f"No completed sweeps found in '{root_sweeps}'. Run `make sweep` first."
            )
        candidate_dirs.sort(key=lambda p: (p / "trials.json").stat().st_mtime, reverse=True)
        resolved_sweep_dir = candidate_dirs[0]
        resolved_sweep_id = resolved_sweep_dir.name
    else:
        resolved_sweep_dir = root_sweeps / resolved_sweep_id

    trials_file = resolved_sweep_dir / "trials.json"
    if not trials_file.exists():
        # Seamless W&B API synchronization fallback
        try:
            import wandb

            resolved_entity = entity or os.environ.get("WANDB_ENTITY", "rl4aa")
            resolved_project = project_name or os.environ.get(
                "WANDB_PROJECT", "mnist-classification"
            )
            api = wandb.Api()
            sweep_path = f"{resolved_entity}/{resolved_project}/{resolved_sweep_id}"
            LOGGER.info(
                "Local ledger missing for '%s'; syncing trials from W&B API (%s)...",
                resolved_sweep_id,
                sweep_path,
            )
            wb_sweep = api.sweep(sweep_path)  # type: ignore[no-untyped-call]
            synced_trials: list[dict[str, Any]] = []
            for r in wb_sweep.runs:
                if r.state in ("finished", "running"):
                    cfg_dict = dict(r.config) if r.config else {}
                    acc = float(r.summary.get("best_val_accuracy", 0.0) or 0.0)
                    loss = float(r.summary.get("best_val_loss", 999.0) or 999.0)
                    epoch = int(r.summary.get("best_epoch", 5) or 5)
                    synced_trials.append(
                        {
                            "run_id": str(r.id),
                            "run_name": str(r.name or r.id),
                            "sweep_id": str(resolved_sweep_id),
                            "params": cfg_dict,
                            "best_val_accuracy": acc,
                            "best_val_loss": loss,
                            "best_epoch": epoch,
                            "total_epochs": int(cfg_dict.get("epochs", 5)),
                            "duration_seconds": float(r.summary.get("_runtime", 0.0) or 0.0),
                            "checkpoint_path": str(
                                resolved_sweep_dir / r.id / "checkpoint_best.pt"
                            ),
                            "history_path": None,
                            "wandb_url": r.url,
                            "timestamp": str(r.created_at),
                        }
                    )
            if synced_trials:
                resolved_sweep_dir.mkdir(parents=True, exist_ok=True)
                save_json(
                    {
                        "sweep_id": str(resolved_sweep_id),
                        "project": str(resolved_project),
                        "entity": str(resolved_entity),
                        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                        "trials": synced_trials,
                    },
                    trials_file,
                )
                LOGGER.info(
                    "Synced %d trials from W&B API to '%s'.", len(synced_trials), trials_file
                )
        except Exception as exc:
            LOGGER.debug("Could not sync sweep from W&B API: %s", exc)

    if not trials_file.exists():
        raise FileNotFoundError(
            f"Sweep trial ledger not found at '{trials_file}'. "
            f"Ensure trials have executed under sweep '{resolved_sweep_id}'."
        )

    trials_raw = load_json(trials_file)
    trials_data = trials_raw.get("trials", []) if isinstance(trials_raw, dict) else trials_raw
    if not isinstance(trials_data, list) or not trials_data:
        raise ValueError(f"No trial records found in '{trials_file}'.")

    # Sort trials by validation accuracy descending, breaking ties with validation loss ascending
    def sort_key(t: dict[str, Any]) -> tuple[float, float]:
        val_acc = float(t.get("best_val_accuracy", 0.0))
        val_loss = float(t.get("best_val_loss", 999.0))
        return (-val_acc, val_loss)

    ranked_trials = sorted(trials_data, key=sort_key)
    champion = ranked_trials[0]

    # Compute statistics
    accs = [float(t.get("best_val_accuracy", 0.0)) for t in ranked_trials]
    losses = [float(t.get("best_val_loss", 0.0)) for t in ranked_trials]
    min_acc, max_acc, mean_acc = min(accs), max(accs), sum(accs) / len(accs)
    min_loss, max_loss, mean_loss = min(losses), max(losses), sum(losses) / len(losses)

    now_iso = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Construct W&B sweep link if entity provided or discoverable
    resolved_entity = entity or os.environ.get("WANDB_ENTITY", "rl4aa")
    resolved_project = project_name or os.environ.get("WANDB_PROJECT", "mnist-classification")
    wandb_sweep_url = (
        f"https://wandb.ai/{resolved_entity}/{resolved_project}/sweeps/{resolved_sweep_id}"
    )

    lines: list[str] = [
        f"# 🧹 Hyperparameter Sweep Report: `{resolved_sweep_id}`",
        "",
        f"> **Generated at**: {now_iso}  ",
        f"> **Project**: `{resolved_project}` | **Entity**: `{resolved_entity}` | **Total Trials**: {len(ranked_trials)}  ",
        f"> **W&B Dashboard**: [{wandb_sweep_url}]({wandb_sweep_url})",
        "",
        "---",
        "",
        "## 🏆 Champion Model Overview",
        "",
        f"The best performing model identified during this sweep is trial **`{champion.get('run_name', champion.get('run_id'))}`**.",
        "",
        "| Metric / Property | Champion Value |",
        "| :--- | :--- |",
        f"| **Run ID** | `{champion.get('run_id')}` |",
        f"| **Run Name** | `{champion.get('run_name')}` |",
        f"| **Best Validation Accuracy** | **`{float(champion.get('best_val_accuracy', 0.0)):.4%}`** |",
        f"| **Best Validation Loss** | **`{float(champion.get('best_val_loss', 0.0)):.4f}`** |",
        f"| **Best Epoch** | `{champion.get('best_epoch')}` of `{champion.get('total_epochs', 5)}` |",
        f"| **Learning Rate** | `{champion.get('params', {}).get('learning_rate', champion.get('params', {}).get('train.lr', 'N/A'))}` |",
        f"| **Optimizer** | `{champion.get('params', {}).get('optimizer', champion.get('params', {}).get('train.optimizer', 'N/A'))}` |",
        f"| **Dropout** | `{champion.get('params', {}).get('dropout', champion.get('params', {}).get('model.dropout', 'N/A'))}` |",
        f"| **Batch Size** | `{champion.get('params', {}).get('batch_size', champion.get('params', {}).get('data.batch_size', 'N/A'))}` |",
        f"| **Weight Decay** | `{champion.get('params', {}).get('weight_decay', champion.get('params', {}).get('train.weight_decay', 'N/A'))}` |",
        f"| **Checkpoint Path** | `{champion.get('checkpoint_path', 'N/A')}` |",
        "",
        "---",
        "",
        "## 📊 Sweep Leaderboard",
        "",
        "| Rank | Run Name | Run ID | Val Acc | Val Loss | Best Epoch | LR | Optimizer | Dropout | Batch Size | Weight Decay |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for rank, trial in enumerate(ranked_trials, start=1):
        p = trial.get("params", {})
        lr_val = p.get("learning_rate", p.get("train.lr", "N/A"))
        lr_str = f"{float(lr_val):.2e}" if isinstance(lr_val, (int, float)) else str(lr_val)
        opt_str = str(p.get("optimizer", p.get("train.optimizer", "N/A")))
        do_val = p.get("dropout", p.get("model.dropout", "N/A"))
        do_str = f"{float(do_val):.3f}" if isinstance(do_val, (int, float)) else str(do_val)
        bs_str = str(p.get("batch_size", p.get("data.batch_size", "N/A")))
        wd_val = p.get("weight_decay", p.get("train.weight_decay", "N/A"))
        wd_str = f"{float(wd_val):.2e}" if isinstance(wd_val, (int, float)) else str(wd_val)
        val_acc_fmt = f"{float(trial.get('best_val_accuracy', 0.0)):.2%}"
        val_loss_fmt = f"{float(trial.get('best_val_loss', 0.0)):.4f}"
        badge = " 🥇" if rank == 1 else (" 🥈" if rank == 2 else (" 🥉" if rank == 3 else ""))

        lines.append(
            f"| {rank}{badge} | `{trial.get('run_name')}` | `{trial.get('run_id')}` | "
            f"**{val_acc_fmt}** | {val_loss_fmt} | {trial.get('best_epoch')} | "
            f"{lr_str} | {opt_str} | {do_str} | {bs_str} | {wd_str} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 📈 Sweep Performance Statistics",
            "",
            "| Metric | Minimum | Mean | Maximum |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Validation Accuracy** | {min_acc:.2%} | {mean_acc:.2%} | {max_acc:.2%} |",
            f"| **Validation Loss** | {min_loss:.4f} | {mean_loss:.4f} | {max_loss:.4f} |",
            "",
            "---",
            "",
            "## 🚀 Promotion & Deployment",
            "",
            "To promote this sweep's champion model into canonical checkpoints and update `params.yaml`:",
            "",
            "```bash",
            f"make sweep-promote SWEEP_ID={resolved_sweep_id}",
            "```",
            "",
            "To promote the champion model and immediately release it to Hugging Face Hub:",
            "",
            "```bash",
            f"make sweep-publish-hf SWEEP_ID={resolved_sweep_id}",
            "```",
            "",
        ]
    )

    report_content = "\n".join(lines) + "\n"

    # Save to local sweep folder
    resolved_sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_folder_report = resolved_sweep_dir / "sweep_report.md"
    sweep_folder_report.write_text(report_content, encoding="utf-8")

    # Also save to canonical reports dir
    canon_report_path = (
        Path(output_path) if output_path is not None else Path("artifacts/reports/sweep_report.md")
    )
    canon_report_path.parent.mkdir(parents=True, exist_ok=True)
    canon_report_path.write_text(report_content, encoding="utf-8")

    LOGGER.info(
        "Generated sweep report for '%s' -> %s (mirror: %s)",
        resolved_sweep_id,
        sweep_folder_report,
        canon_report_path,
    )
    return canon_report_path
