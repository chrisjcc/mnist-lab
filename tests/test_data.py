"""Unit tests for MNIST data preprocessing, stratified splitting, and DataLoaders."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mnist_pipeline.config import DataConfig, PipelineConfig
from mnist_pipeline.data import (
    build_dataloaders,
    generate_synthetic_mnist_tensors,
    load_processed_split,
    normalize_uint8_images,
    preprocess_mnist_data,
    stratified_train_val_split,
    validate_batch_contract,
)


def test_stratified_split_is_deterministic_and_balanced() -> None:
    _, labels = generate_synthetic_mnist_tensors(num_samples=200, seed=42)
    train_idx_1, val_idx_1 = stratified_train_val_split(labels, val_fraction=0.1, seed=42)
    train_idx_2, val_idx_2 = stratified_train_val_split(labels, val_fraction=0.1, seed=42)

    assert len(train_idx_1) == 180
    assert len(val_idx_1) == 20
    assert torch.equal(train_idx_1, train_idx_2)
    assert torch.equal(val_idx_1, val_idx_2)

    overlap = set(train_idx_1.tolist()).intersection(set(val_idx_1.tolist()))
    assert len(overlap) == 0

    val_labels = labels[val_idx_1]
    counts = torch.bincount(val_labels, minlength=10)
    assert torch.all(counts == 2)


def test_normalize_and_batch_contract_validation() -> None:
    raw = torch.randint(0, 256, (16, 28, 28), dtype=torch.uint8)
    norm = normalize_uint8_images(raw, mean=0.1307, std=0.3081)
    labels = torch.arange(16, dtype=torch.int64) % 10
    validate_batch_contract(norm, labels)

    with pytest.raises(ValueError, match="Expected images shape"):
        validate_batch_contract(torch.zeros((16, 28, 28)), labels)

    with pytest.raises(ValueError, match="Labels out of valid range"):
        validate_batch_contract(norm, torch.full((16,), 12, dtype=torch.int64))


def test_preprocess_and_build_dataloaders_with_synthetic_fallback(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            download=False,
            val_fraction=0.2,
            batch_size=16,
            seed=42,
            allow_synthetic_fallback=True,
            num_synthetic_train=100,
            num_synthetic_test=30,
        )
    )
    meta = preprocess_mnist_data(cfg.data)
    assert meta["splits"]["train"]["num_examples"] == 80
    assert meta["splits"]["val"]["num_examples"] == 20
    assert meta["splits"]["test"]["num_examples"] == 30

    train_images, train_labels = load_processed_split(cfg.data.processed_dir, "train")
    assert train_images.shape == (80, 1, 28, 28)
    assert train_labels.shape == (80,)

    bundle = build_dataloaders(cfg, device=torch.device("cpu"))
    assert bundle.train_size == 80
    assert bundle.val_size == 20
    assert bundle.test_size == 30

    batch_images, batch_labels = next(iter(bundle.train_loader))
    validate_batch_contract(batch_images, batch_labels)
    assert batch_images.shape == (16, 1, 28, 28)
