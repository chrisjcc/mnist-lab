"""Training and validation loops, checkpoint management, and telemetry persistence."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mnist_pipeline.config import PipelineConfig, TrainConfig
from mnist_pipeline.data import build_dataloaders
from mnist_pipeline.model import (
    build_model,
    count_trainable_parameters,
    load_checkpoint,
    save_checkpoint,
)
from mnist_pipeline.tracker import ExperimentTracker
from mnist_pipeline.utils import (
    collect_environment_metadata,
    resolve_device,
    save_json,
    set_seed,
    synchronize_device,
)

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Epoch-level training and validation metrics.

    Attributes:
        epoch: 1-indexed epoch number.
        train_loss: Sample-weighted average training cross-entropy loss.
        train_accuracy: Training classification accuracy in [0.0, 1.0].
        val_loss: Sample-weighted average validation cross-entropy loss.
        val_accuracy: Validation classification accuracy in [0.0, 1.0].
        lr: Active optimizer learning rate.
        epoch_time_sec: Wall-clock duration of the epoch in seconds.
    """

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    lr: float
    epoch_time_sec: float


@dataclass
class TrainingSummary:
    """Summary of a completed training run.

    Attributes:
        best_epoch: 1-indexed epoch that achieved the highest validation accuracy.
        best_val_accuracy: Highest validation accuracy achieved.
        best_val_loss: Validation loss at the best epoch.
        final_train_loss: Training loss at the final completed epoch.
        final_train_accuracy: Training accuracy at the final completed epoch.
        total_training_time_sec: Total wall-clock training time in seconds.
        avg_epoch_time_sec: Mean wall-clock duration per epoch in seconds.
        trainable_parameters: Total trainable parameters in the model.
        device: String identifier of the compute device used.
        checkpoint_best_path: Path to checkpoint_best.pt.
        checkpoint_last_path: Path to checkpoint_last.pt.
        history: List of EpochMetrics across all executed epochs.
    """

    best_epoch: int
    best_val_accuracy: float
    best_val_loss: float
    final_train_loss: float
    final_train_accuracy: float
    total_training_time_sec: float
    avg_epoch_time_sec: float
    trainable_parameters: int
    device: str
    checkpoint_best_path: str
    checkpoint_last_path: str
    history: list[EpochMetrics]


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Instantiate the configured PyTorch optimizer for model parameters.

    Args:
        model: Neural network module whose trainable parameters will be optimized.
        cfg: TrainConfig specifying optimizer type, learning rate, and weight decay.

    Returns:
        Configured torch.optim.Optimizer instance.

    Raises:
        ValueError: If cfg.optimizer is unsupported.
    """
    opt_name = cfg.optimizer.strip().lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'")


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Run one training epoch and return sample-weighted average loss and accuracy.

    Args:
        model: PyTorch model instance.
        loader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer instance.
        device: Target compute device.
        max_batches: Optional maximum number of mini-batches to process.

    Returns:
        Tuple of (avg_train_loss, train_accuracy).
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    non_blocking = device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss: torch.Tensor = criterion(logits, labels)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training loader yielded 0 samples.")
    return total_loss / total_samples, total_correct / total_samples


def run_validation_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Run one gradient-free validation epoch and return average loss and accuracy.

    Args:
        model: PyTorch model instance.
        loader: Validation DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        device: Target compute device.
        max_batches: Optional maximum number of mini-batches to process.

    Returns:
        Tuple of (avg_val_loss, val_accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            logits = model(images)
            loss: torch.Tensor = criterion(logits, labels)

            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Validation loader yielded 0 samples.")
    return total_loss / total_samples, total_correct / total_samples


def write_train_history_csv(history: list[EpochMetrics], path: str | Path) -> Path:
    """Save epoch-level training and validation metrics to a CSV file.

    Args:
        history: List of EpochMetrics objects.
        path: Output CSV file path.

    Returns:
        Resolved Path to the written CSV file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "lr",
        "epoch_time_sec",
    ]
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(asdict(row))
    return target


def train_model(cfg: PipelineConfig) -> TrainingSummary:
    """Execute the full training and validation loop, saving checkpoints and history.

    Args:
        cfg: Validated PipelineConfig object.

    Returns:
        TrainingSummary describing the best checkpoint, metrics, and timing.
    """
    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    bundle = build_dataloaders(cfg, device=device)

    model = build_model(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.train)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 1

    if cfg.train.resume_from:
        resume_path = Path(cfg.train.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found at '{resume_path}'")
        model, ckpt_payload = load_checkpoint(resume_path, device=device, optimizer=optimizer)
        start_epoch = int(ckpt_payload.get("epoch", 0)) + 1
        best_val_accuracy = float(ckpt_payload.get("best_val_accuracy", -1.0))
        logger.info(
            "Resumed training from '%s' at epoch %d (best_val_accuracy=%.4f)",
            resume_path,
            start_epoch,
            best_val_accuracy,
        )

    num_params = count_trainable_parameters(model)
    logger.info(
        "Starting training on device=%s (trainable_params=%d, train_size=%d, val_size=%d, epochs=%d)",
        device,
        num_params,
        bundle.train_size,
        bundle.val_size,
        cfg.train.epochs,
    )

    checkpoints_dir = Path(cfg.paths.checkpoints_dir)
    reports_dir = Path(cfg.paths.reports_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt_best_path = checkpoints_dir / "checkpoint_best.pt"
    ckpt_last_path = checkpoints_dir / "checkpoint_last.pt"

    tracker = ExperimentTracker(cfg.wandb, full_cfg=cfg, job_type="train")
    tracker.start()

    history: list[EpochMetrics] = []
    epochs_without_improvement = 0
    synchronize_device(device)
    train_start_time = time.perf_counter()

    try:
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            synchronize_device(device)
            epoch_start = time.perf_counter()

            train_loss, train_acc = run_train_epoch(
                model=model,
                loader=bundle.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                max_batches=cfg.train.max_train_batches,
            )
            val_loss, val_acc = run_validation_epoch(
                model=model,
                loader=bundle.val_loader,
                criterion=criterion,
                device=device,
                max_batches=cfg.train.max_val_batches,
            )

            synchronize_device(device)
            epoch_elapsed = time.perf_counter() - epoch_start
            current_lr = float(optimizer.param_groups[0]["lr"])

            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                lr=current_lr,
                epoch_time_sec=epoch_elapsed,
            )
            history.append(epoch_metrics)

            improved = val_acc > best_val_accuracy
            if improved:
                best_val_accuracy = val_acc
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    path=ckpt_best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_accuracy=best_val_accuracy,
                    cfg=cfg,
                )
                model.save_pretrained(
                    checkpoints_dir,
                    extra_metadata={
                        "best_epoch": best_epoch,
                        "best_val_accuracy": best_val_accuracy,
                        "best_val_loss": best_val_loss,
                        "trainable_parameters": num_params,
                    },
                )
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                path=ckpt_last_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                cfg=cfg,
            )

            tracker.log_metrics(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "train/lr": current_lr,
                    "train/epoch_time_sec": epoch_elapsed,
                },
                step=epoch,
            )

            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | %.2fs%s",
                epoch,
                cfg.train.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                epoch_elapsed,
                " [BEST]" if improved else "",
            )

            if (
                cfg.train.early_stopping_patience is not None
                and epochs_without_improvement >= cfg.train.early_stopping_patience
            ):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch,
                    cfg.train.early_stopping_patience,
                )
                break

        synchronize_device(device)
        total_time = time.perf_counter() - train_start_time
        avg_epoch_time = total_time / max(1, len(history))

        write_train_history_csv(history, reports_dir / "train_history.csv")

        env_meta: dict[str, Any] = collect_environment_metadata(device=device, seed=cfg.train.seed)
        env_meta["training_summary"] = {
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "total_training_time_sec": total_time,
            "avg_epoch_time_sec": avg_epoch_time,
            "trainable_parameters": num_params,
            "epochs_completed": len(history),
        }
        save_json(env_meta, reports_dir / "environment.json")

        tracker.log_summary(
            {
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,
                "total_training_time_sec": total_time,
                "trainable_parameters": num_params,
            }
        )
        if ckpt_best_path.exists():
            tracker.log_artifact(
                name="mnist-cnn-best",
                artifact_type="model",
                file_path=ckpt_best_path,
                metadata={"best_epoch": best_epoch, "best_val_accuracy": best_val_accuracy},
            )
    finally:
        tracker.finish()

    final_metrics = (
        history[-1] if history else EpochMetrics(0, 0.0, 0.0, 0.0, 0.0, cfg.train.lr, 0.0)
    )
    return TrainingSummary(
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
        best_val_loss=best_val_loss,
        final_train_loss=final_metrics.train_loss,
        final_train_accuracy=final_metrics.train_accuracy,
        total_training_time_sec=total_time,
        avg_epoch_time_sec=avg_epoch_time,
        trainable_parameters=num_params,
        device=str(device),
        checkpoint_best_path=str(ckpt_best_path),
        checkpoint_last_path=str(ckpt_last_path),
        history=history,
    )
