"""MNIST dataset downloading, deterministic stratified splitting, preprocessing, and DataLoaders."""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from mnist_pipeline.config import DataConfig, PipelineConfig
from mnist_pipeline.utils import save_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataLoadersBundle:
    """Container holding train, validation, and test PyTorch DataLoaders and split sizes.

    Attributes:
        train_loader: DataLoader for the training partition (shuffled).
        val_loader: DataLoader for the validation partition (unshuffled).
        test_loader: DataLoader for the held-out test partition (unshuffled).
        train_size: Number of examples in the training split.
        val_size: Number of examples in the validation split.
        test_size: Number of examples in the test split.
    """

    train_loader: DataLoader[tuple[torch.Tensor, ...]]
    val_loader: DataLoader[tuple[torch.Tensor, ...]]
    test_loader: DataLoader[tuple[torch.Tensor, ...]]
    train_size: int
    val_size: int
    test_size: int


def _seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker processes deterministically.

    Args:
        worker_id: Index of the worker process assigned by PyTorch DataLoader.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Compute a deterministic SHA-256 hex digest of a CPU tensor's raw bytes.

    Args:
        tensor: Input CPU tensor.

    Returns:
        64-character hexadecimal SHA-256 digest.
    """
    contiguous = tensor.detach().cpu().contiguous()
    raw_bytes = contiguous.numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def _class_distribution(labels: torch.Tensor, num_classes: int = 10) -> dict[str, int]:
    """Count the number of examples per class label.

    Args:
        labels: 1D integer tensor of class labels.
        num_classes: Total number of classes.

    Returns:
        Dictionary mapping string class label ('0'..'9') to sample count.
    """
    counts = torch.bincount(labels.to(torch.int64), minlength=num_classes)
    return {str(idx): int(counts[idx].item()) for idx in range(num_classes)}


def stratified_train_val_split(
    labels: torch.Tensor,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split dataset indices into stratified train and validation sets.

    Args:
        labels: 1D integer tensor of training set class labels.
        val_fraction: Fraction of examples per class assigned to validation.
        seed: Random seed controlling per-class index permutation.

    Returns:
        Tuple of (train_indices, val_indices) as 1D int64 tensors.

    Raises:
        ValueError: If val_fraction is outside (0, 1) or labels is not 1D.
    """
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels tensor, got shape {tuple(labels.shape)}")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    labels_np = labels.detach().cpu().numpy()
    unique_classes = np.unique(labels_np)

    train_idx_parts: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    val_idx_parts: list[np.ndarray[Any, np.dtype[np.int64]]] = []

    total_n = len(labels_np)
    target_val_total = int(round(total_n * val_fraction))

    allocated_val = 0
    class_indices_list: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    for cls in unique_classes:
        cls_indices = np.where(labels_np == cls)[0].astype(np.int64)
        rng.shuffle(cls_indices)
        class_indices_list.append(cls_indices)

    val_counts: list[int] = []
    for cls_indices in class_indices_list:
        n_cls = len(cls_indices)
        n_val = max(1, int(round(n_cls * val_fraction)))
        val_counts.append(n_val)
        allocated_val += n_val

    # Adjust largest classes by +/- 1 so total validation size matches exact target_val_total
    diff = target_val_total - allocated_val
    if diff != 0 and len(val_counts) > 0:
        order = np.argsort([-len(arr) for arr in class_indices_list])
        step = 1 if diff > 0 else -1
        for idx in range(abs(diff)):
            target_bucket = int(order[idx % len(order)])
            proposed = val_counts[target_bucket] + step
            if 1 <= proposed < len(class_indices_list[target_bucket]):
                val_counts[target_bucket] = proposed

    for cls_indices, n_val in zip(class_indices_list, val_counts, strict=True):
        val_idx_parts.append(cls_indices[:n_val])
        train_idx_parts.append(cls_indices[n_val:])

    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return torch.from_numpy(train_idx).to(torch.int64), torch.from_numpy(val_idx).to(torch.int64)


def generate_synthetic_digit_template(digit: int) -> torch.Tensor:
    """Create a canonical 28x28 grayscale pattern for a given digit (0-9).

    Args:
        digit: Integer digit in [0, 9].

    Returns:
        Float32 tensor of shape [1, 28, 28] with pixel intensities in [0.0, 1.0].
    """
    canvas = torch.zeros((1, 28, 28), dtype=torch.float32)
    d = digit % 10
    if d == 0:
        canvas[:, 5:23, 7:10] = 1.0
        canvas[:, 5:23, 18:21] = 1.0
        canvas[:, 5:8, 7:21] = 1.0
        canvas[:, 20:23, 7:21] = 1.0
    elif d == 1:
        canvas[:, 5:23, 13:16] = 1.0
        canvas[:, 20:23, 9:20] = 1.0
    elif d == 2:
        canvas[:, 5:8, 7:21] = 1.0
        canvas[:, 5:15, 18:21] = 1.0
        canvas[:, 12:15, 7:21] = 1.0
        canvas[:, 12:23, 7:10] = 1.0
        canvas[:, 20:23, 7:21] = 1.0
    elif d == 3:
        canvas[:, 5:8, 7:21] = 1.0
        canvas[:, 12:15, 9:21] = 1.0
        canvas[:, 20:23, 7:21] = 1.0
        canvas[:, 5:23, 18:21] = 1.0
    elif d == 4:
        canvas[:, 5:15, 7:10] = 1.0
        canvas[:, 12:15, 7:21] = 1.0
        canvas[:, 5:23, 17:20] = 1.0
    elif d == 5:
        canvas[:, 5:8, 7:21] = 1.0
        canvas[:, 5:15, 7:10] = 1.0
        canvas[:, 12:15, 7:21] = 1.0
        canvas[:, 12:23, 18:21] = 1.0
        canvas[:, 20:23, 7:21] = 1.0
    elif d == 6:
        canvas[:, 5:23, 7:10] = 1.0
        canvas[:, 5:8, 7:21] = 1.0
        canvas[:, 12:15, 7:21] = 1.0
        canvas[:, 12:23, 18:21] = 1.0
        canvas[:, 20:23, 7:21] = 1.0
    elif d == 7:
        canvas[:, 5:8, 7:21] = 1.0
        canvas[:, 5:23, 17:20] = 1.0
    elif d == 8:
        canvas[:, 5:23, 7:10] = 1.0
        canvas[:, 5:23, 18:21] = 1.0
        canvas[:, 5:8, 7:21] = 1.0
        canvas[:, 12:15, 7:21] = 1.0
        canvas[:, 20:23, 7:21] = 1.0
    else:  # 9
        canvas[:, 5:15, 7:10] = 1.0
        canvas[:, 5:23, 18:21] = 1.0
        canvas[:, 5:8, 7:21] = 1.0
        canvas[:, 12:15, 7:21] = 1.0
        canvas[:, 20:23, 7:21] = 1.0
    return canvas


def generate_synthetic_mnist_tensors(
    num_samples: int,
    mean: float = 0.1307,
    std: float = 0.3081,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic, class-balanced synthetic digit tensors for testing.

    Args:
        num_samples: Total number of samples to generate.
        mean: Normalization mean to apply.
        std: Normalization standard deviation to apply.
        seed: Random seed for noise generation.

    Returns:
        Tuple of (images [N, 1, 28, 28] float32, labels [N] int64).
    """
    gen = torch.Generator().manual_seed(seed)
    templates = torch.stack([generate_synthetic_digit_template(d) for d in range(10)], dim=0)
    labels = torch.arange(num_samples, dtype=torch.int64) % 10
    perm = torch.randperm(num_samples, generator=gen)
    labels = labels[perm]

    base_images = templates[labels]
    noise = torch.randn(base_images.shape, generator=gen, dtype=torch.float32) * 0.08
    raw_images = torch.clamp(base_images + noise, 0.0, 1.0)
    normalized = (raw_images - mean) / std
    return normalized.to(torch.float32), labels.to(torch.int64)


def normalize_uint8_images(
    raw_images: torch.Tensor,
    mean: float = 0.1307,
    std: float = 0.3081,
) -> torch.Tensor:
    """Convert uint8 [N, 28, 28] images to normalized float32 [N, 1, 28, 28] tensors.

    Args:
        raw_images: Uint8 tensor of shape [N, 28, 28] or [N, 1, 28, 28] with values in [0, 255].
        mean: Normalization mean.
        std: Normalization standard deviation.

    Returns:
        Normalized float32 tensor of shape [N, 1, 28, 28].
    """
    if raw_images.ndim == 3:
        raw_images = raw_images.unsqueeze(1)
    if raw_images.ndim != 4 or raw_images.shape[1:] != (1, 28, 28):
        raise ValueError(
            f"Expected image tensor shape [N, 1, 28, 28], got {tuple(raw_images.shape)}"
        )
    scaled = raw_images.to(torch.float32) / 255.0
    return (scaled - mean) / std


def validate_batch_contract(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
) -> None:
    """Assert that a mini-batch of images and labels conforms to the MNIST tensor contract.

    Args:
        images: Image batch tensor, expected float32 of shape [N, 1, 28, 28].
        labels: Label batch tensor, expected int64 of shape [N] with values in [0, num_classes - 1].
        num_classes: Number of valid class indices.

    Raises:
        ValueError: If shape, dtype, finiteness, or label range checks fail.
    """
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError(f"Expected images shape [N, 1, 28, 28], got {tuple(images.shape)}")
    if images.dtype != torch.float32:
        raise ValueError(f"Expected images dtype torch.float32, got {images.dtype}")
    if not torch.isfinite(images).all():
        raise ValueError("Images tensor contains non-finite values (NaN or Inf)")

    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError(f"Expected labels shape [{images.shape[0]}], got {tuple(labels.shape)}")
    if labels.dtype != torch.int64:
        raise ValueError(f"Expected labels dtype torch.int64, got {labels.dtype}")
    min_lbl = int(labels.min().item())
    max_lbl = int(labels.max().item())
    if min_lbl < 0 or max_lbl >= num_classes:
        raise ValueError(
            f"Labels out of valid range [0, {num_classes - 1}]: [{min_lbl}, {max_lbl}]"
        )


def preprocess_mnist_data(cfg: DataConfig) -> dict[str, Any]:
    """Download, normalize, deterministically split, and persist MNIST tensors to disk.

    Args:
        cfg: DataConfig instance specifying paths, normalization constants, and split seed.

    Returns:
        Dictionary of dataset split metadata (also saved to split_metadata.json).
    """
    raw_dir = Path(cfg.raw_dir)
    processed_dir = Path(cfg.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    source_type = "torchvision.datasets.MNIST"
    try:
        train_ds = datasets.MNIST(root=str(raw_dir), train=True, download=cfg.download)
        test_ds = datasets.MNIST(root=str(raw_dir), train=False, download=cfg.download)
        full_train_images = normalize_uint8_images(train_ds.data, mean=cfg.mean, std=cfg.std)
        full_train_labels = train_ds.targets.to(torch.int64)
        test_images = normalize_uint8_images(test_ds.data, mean=cfg.mean, std=cfg.std)
        test_labels = test_ds.targets.to(torch.int64)
    except Exception as exc:
        if not cfg.allow_synthetic_fallback:
            raise RuntimeError(
                f"Failed to load or download MNIST dataset in '{raw_dir}': {exc}"
            ) from exc
        logger.warning(
            "MNIST download unavailable (%s); falling back to deterministic synthetic dataset.",
            exc,
        )
        source_type = "synthetic_fallback"
        full_train_images, full_train_labels = generate_synthetic_mnist_tensors(
            num_samples=cfg.num_synthetic_train,
            mean=cfg.mean,
            std=cfg.std,
            seed=cfg.seed,
        )
        test_images, test_labels = generate_synthetic_mnist_tensors(
            num_samples=cfg.num_synthetic_test,
            mean=cfg.mean,
            std=cfg.std,
            seed=cfg.seed + 1,
        )

    train_idx, val_idx = stratified_train_val_split(
        labels=full_train_labels,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )

    train_images = full_train_images[train_idx].contiguous()
    train_labels = full_train_labels[train_idx].contiguous()
    val_images = full_train_images[val_idx].contiguous()
    val_labels = full_train_labels[val_idx].contiguous()
    test_images = test_images.contiguous()
    test_labels = test_labels.contiguous()

    validate_batch_contract(train_images, train_labels)
    validate_batch_contract(val_images, val_labels)
    validate_batch_contract(test_images, test_labels)

    train_path = processed_dir / "train.pt"
    val_path = processed_dir / "val.pt"
    test_path = processed_dir / "test.pt"

    torch.save({"images": train_images, "labels": train_labels, "indices": train_idx}, train_path)
    torch.save({"images": val_images, "labels": val_labels, "indices": val_idx}, val_path)
    torch.save({"images": test_images, "labels": test_labels}, test_path)

    metadata: dict[str, Any] = {
        "source": source_type,
        "seed": cfg.seed,
        "val_fraction": cfg.val_fraction,
        "normalization": {"mean": cfg.mean, "std": cfg.std},
        "splits": {
            "train": {
                "num_examples": int(train_images.shape[0]),
                "shape": list(train_images.shape),
                "images_sha256": _tensor_sha256(train_images),
                "labels_sha256": _tensor_sha256(train_labels),
                "class_distribution": _class_distribution(train_labels),
            },
            "val": {
                "num_examples": int(val_images.shape[0]),
                "shape": list(val_images.shape),
                "images_sha256": _tensor_sha256(val_images),
                "labels_sha256": _tensor_sha256(val_labels),
                "class_distribution": _class_distribution(val_labels),
            },
            "test": {
                "num_examples": int(test_images.shape[0]),
                "shape": list(test_images.shape),
                "images_sha256": _tensor_sha256(test_images),
                "labels_sha256": _tensor_sha256(test_labels),
                "class_distribution": _class_distribution(test_labels),
            },
        },
    }
    save_json(metadata, processed_dir / "split_metadata.json")
    logger.info(
        "Saved processed splits to %s (train=%d, val=%d, test=%d)",
        processed_dir,
        train_images.shape[0],
        val_images.shape[0],
        test_images.shape[0],
    )
    return metadata


def load_processed_split(
    processed_dir: str | Path,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a preprocessed split ('train', 'val', or 'test') from disk.

    Args:
        processed_dir: Directory containing train.pt, val.pt, and test.pt.
        split: Split identifier ('train', 'val', or 'test').

    Returns:
        Tuple of (images [N, 1, 28, 28], labels [N]).

    Raises:
        FileNotFoundError: If the requested split file does not exist.
        ValueError: If split name is invalid or loaded tensors fail contract validation.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
    split_path = Path(processed_dir) / f"{split}.pt"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Processed split file not found at '{split_path}'. Run `make data` first."
        )
    payload = torch.load(split_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "images" not in payload or "labels" not in payload:
        raise ValueError(f"Invalid split payload in '{split_path}'")
    images = payload["images"]
    labels = payload["labels"]
    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError(f"Expected Tensor objects in '{split_path}'")
    validate_batch_contract(images, labels)
    return images, labels


def build_dataloaders(
    cfg: PipelineConfig,
    device: torch.device | None = None,
) -> DataLoadersBundle:
    """Build deterministic train, validation, and test DataLoaders from processed splits.

    Automatically triggers `preprocess_mnist_data` if the processed files do not yet exist.

    Args:
        cfg: Full PipelineConfig object.
        device: Target compute device (used to configure pin_memory).

    Returns:
        DataLoadersBundle containing train, val, and test DataLoaders and sample counts.
    """
    processed_dir = Path(cfg.data.processed_dir)
    if not (processed_dir / "train.pt").exists():
        preprocess_mnist_data(cfg.data)

    train_images, train_labels = load_processed_split(processed_dir, "train")
    val_images, val_labels = load_processed_split(processed_dir, "val")
    test_images, test_labels = load_processed_split(processed_dir, "test")

    train_ds = TensorDataset(train_images, train_labels)
    val_ds = TensorDataset(val_images, val_labels)
    test_ds = TensorDataset(test_images, test_labels)

    use_pin_memory = bool(cfg.data.pin_memory and device is not None and device.type == "cuda")
    generator = torch.Generator()
    generator.manual_seed(cfg.data.seed)

    train_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker if cfg.data.num_workers > 0 else None,
        generator=generator,
    )
    val_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        val_ds,
        batch_size=cfg.evaluate.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    test_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        test_ds,
        batch_size=cfg.evaluate.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return DataLoadersBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=int(train_images.shape[0]),
        val_size=int(val_images.shape[0]),
        test_size=int(test_images.shape[0]),
    )
