"""Held-out test evaluation and latency/throughput benchmarking for MNIST."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mnist_pipeline.config import PipelineConfig
from mnist_pipeline.data import build_dataloaders
from mnist_pipeline.metrics import compute_classification_metrics
from mnist_pipeline.model import (
    MNISTConvNet,
    count_total_parameters,
    count_trainable_parameters,
    load_checkpoint,
)
from mnist_pipeline.tracker import ExperimentTracker
from mnist_pipeline.utils import (
    resolve_device,
    save_json,
    set_seed,
    synchronize_device,
)

LOGGER = logging.getLogger(__name__)


def _resolve_checkpoint_path(
    cfg: PipelineConfig,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Resolve the best checkpoint path from explicit path or config defaults.

    Args:
        cfg: Validated `PipelineConfig`.
        checkpoint_path: Optional override path to a `.pt` checkpoint file.

    Returns:
        Resolved `Path` to the checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path)
    else:
        default_ckpt = Path(cfg.paths.checkpoints_dir) / "checkpoint_best.pt"
        resolved = default_ckpt if default_ckpt.exists() else Path(cfg.evaluate.checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found at {resolved}. Run `make train` first.")
    return resolved


def evaluate_checkpoint(
    cfg: PipelineConfig,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a trained MNIST checkpoint on the held-out test dataset.

    Writes:
      - `<paths.evaluation_dir>/test_metrics.json`
      - `<paths.evaluation_dir>/classification_report.txt`
      - `<paths.evaluation_dir>/confusion_matrix.csv`

    Args:
        cfg: Validated `PipelineConfig`.
        checkpoint_path: Optional override path to the `.pt` checkpoint.

    Returns:
        JSON-serializable dictionary of test metrics.
    """
    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    ckpt_path = _resolve_checkpoint_path(cfg, checkpoint_path)

    bundle = build_dataloaders(cfg, device=device)
    model, payload = load_checkpoint(ckpt_path, device=device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for images, labels in bundle.test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss: torch.Tensor = criterion(logits, labels)

            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            all_preds.extend(int(x) for x in preds.cpu().tolist())
            all_labels.extend(int(x) for x in labels.cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        avg_loss=avg_loss,
        num_classes=cfg.model.num_classes,
    )

    output_dir = Path(cfg.paths.evaluation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = metrics.to_dict()
    metrics_dict["checkpoint_path"] = str(ckpt_path)
    metrics_dict["checkpoint_epoch"] = int(payload.get("epoch", 0))
    metrics_dict["device"] = str(device)

    metrics_json_path = output_dir / "test_metrics.json"
    report_txt_path = output_dir / "classification_report.txt"
    cm_csv_path = output_dir / "confusion_matrix.csv"

    save_json(metrics_dict, metrics_json_path)
    report_txt_path.write_text(metrics.classification_report_text, encoding="utf-8")

    with cm_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["true\\pred"] + [f"pred_{i}" for i in range(cfg.model.num_classes)]
        writer.writerow(header)
        for cls_idx, row in enumerate(metrics.confusion_matrix):
            writer.writerow([f"true_{cls_idx}", *row])

    tracker = ExperimentTracker(cfg.wandb, full_cfg=cfg, job_type="evaluate")
    tracker.start()
    try:
        tracker.log_summary(
            {
                "test/loss": metrics.test_loss,
                "test/acc": metrics.test_acc,
                "test/macro_f1": metrics.macro_f1,
                "test/weighted_f1": metrics.weighted_f1,
            }
        )
    finally:
        tracker.finish()

    LOGGER.info(
        "Held-out test evaluation complete: test_acc=%.4f | macro_f1=%.4f | test_loss=%.4f",
        metrics.test_acc,
        metrics.macro_f1,
        metrics.test_loss,
    )
    return metrics_dict


def benchmark_checkpoint(
    cfg: PipelineConfig,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark inference latency, throughput, and model footprint across batch sizes.

    Runs `cfg.evaluate.benchmark_warmup_runs` warm-up forward passes followed by
    `cfg.evaluate.benchmark_timed_runs` timed forward passes with explicit device
    synchronization (`torch.cuda.synchronize` / `torch.mps.synchronize`) for each
    batch size in `cfg.evaluate.benchmark_batch_sizes`.

    Args:
        cfg: Validated `PipelineConfig`.
        checkpoint_path: Optional override path to the `.pt` checkpoint.

    Returns:
        JSON-serializable dictionary saved to `<paths.benchmarks_dir>/benchmark_metrics.json`.
    """
    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    ckpt_path = _resolve_checkpoint_path(cfg, checkpoint_path)

    model: MNISTConvNet
    model, _ = load_checkpoint(ckpt_path, device=device)
    model.eval()

    trainable_params = count_trainable_parameters(model)
    total_params = count_total_parameters(model)
    checkpoint_size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 4)

    safetensors_path = ckpt_path.parent / "model.safetensors"
    safetensors_size_mb = (
        round(safetensors_path.stat().st_size / (1024 * 1024), 4)
        if safetensors_path.exists()
        else None
    )

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    batch_metrics: dict[str, dict[str, float]] = {}
    warmup_runs = cfg.evaluate.benchmark_warmup_runs
    timed_runs = cfg.evaluate.benchmark_timed_runs

    with torch.inference_mode():
        for batch_size in cfg.evaluate.benchmark_batch_sizes:
            dummy_input = torch.randn(
                batch_size,
                cfg.model.in_channels,
                28,
                28,
                dtype=torch.float32,
                device=device,
            )

            # Warm-up passes
            for _ in range(warmup_runs):
                _ = model(dummy_input)
            synchronize_device(device)

            # Timed passes
            latencies_ms: list[float] = []
            for _ in range(timed_runs):
                synchronize_device(device)
                start_ns = time.perf_counter_ns()
                _ = model(dummy_input)
                synchronize_device(device)
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                latencies_ms.append(elapsed_ms)

            arr = np.asarray(latencies_ms, dtype=np.float64)
            mean_ms = float(np.mean(arr))
            std_ms = float(np.std(arr))
            p50_ms = float(np.percentile(arr, 50))
            p95_ms = float(np.percentile(arr, 95))
            p99_ms = float(np.percentile(arr, 99))
            samples_per_sec = float((batch_size * 1000.0) / max(mean_ms, 1e-6))

            batch_metrics[str(batch_size)] = {
                "mean_ms": round(mean_ms, 4),
                "std_ms": round(std_ms, 4),
                "p50_ms": round(p50_ms, 4),
                "p95_ms": round(p95_ms, 4),
                "p99_ms": round(p99_ms, 4),
                "samples_per_sec": round(samples_per_sec, 2),
            }
            LOGGER.info(
                "Benchmark batch_size=%d: mean=%.3f ms | p95=%.3f ms | throughput=%.1f samples/s",
                batch_size,
                mean_ms,
                p95_ms,
                samples_per_sec,
            )

    peak_cuda_memory_mb: float | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        peak_cuda_memory_mb = round(
            float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024), 4
        )

    result: dict[str, Any] = {
        "device": str(device),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "checkpoint_size_mb": checkpoint_size_mb,
        "safetensors_size_mb": safetensors_size_mb,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "batch_metrics": batch_metrics,
    }

    benchmark_dir = Path(cfg.paths.benchmarks_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, benchmark_dir / "benchmark_metrics.json")
    return result
