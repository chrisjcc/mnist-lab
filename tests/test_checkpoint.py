"""Unit tests for model checkpointing and Hugging Face safetensors round-trip serialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mnist_pipeline.config import PipelineConfig
from mnist_pipeline.model import MNISTConvNet, load_checkpoint, save_checkpoint
from mnist_pipeline.utils import set_seed


def test_checkpoint_round_trip_matches_logits(tmp_path: Path) -> None:
    set_seed(42)
    cfg = PipelineConfig()
    model = MNISTConvNet(cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    ckpt_path = save_checkpoint(
        path=tmp_path / "checkpoint_best.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_val_accuracy=0.989,
        cfg=cfg,
    )
    assert ckpt_path.exists()

    new_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    reloaded_model, payload = load_checkpoint(ckpt_path, device="cpu", optimizer=new_optimizer)
    assert payload["epoch"] == 3
    assert payload["best_val_accuracy"] == pytest.approx(0.989)

    model.eval()
    reloaded_model.eval()
    inputs = torch.randn(6, 1, 28, 28)
    with torch.inference_mode():
        assert torch.allclose(model(inputs), reloaded_model(inputs), atol=1e-6)


def test_safetensors_pretrained_round_trip(tmp_path: Path) -> None:
    set_seed(99)
    cfg = PipelineConfig()
    model = MNISTConvNet(cfg.model)
    model.eval()

    save_dir = model.save_pretrained(
        tmp_path / "hf_export",
        extra_metadata={"test_accuracy": 0.991},
    )
    assert (save_dir / "model.safetensors").exists()
    assert (save_dir / "config.json").exists()

    reloaded = MNISTConvNet.from_pretrained(save_dir, map_location="cpu")
    inputs = torch.randn(4, 1, 28, 28)
    with torch.inference_mode():
        assert torch.allclose(model(inputs), reloaded(inputs), atol=1e-6)
