"""Unit tests for PIL image preprocessing, inference model loading, and prediction output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mnist_pipeline.config import PipelineConfig
from mnist_pipeline.inference import (
    load_inference_model,
    predict_pil_image,
    predict_tensor,
    preprocess_pil_image,
)
from mnist_pipeline.model import MNISTConvNet, save_checkpoint


def test_preprocess_pil_image_rgb_and_inversion() -> None:
    """Verify arbitrary RGB and RGBA images are normalized to (1, 1, 28, 28) float32."""
    # Dark stroke on white background (simulating pen on paper)
    arr = np.full((64, 80, 3), 255, dtype=np.uint8)
    arr[16:48, 34:46, :] = 10
    rgb_img = Image.fromarray(arr, mode="RGB")

    tensor, display_img = preprocess_pil_image(
        rgb_img,
        invert=False,
        auto_invert=True,
        center_crop_foreground=True,
    )
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert display_img.shape == (28, 28)
    # Center of image should have positive foreground intensity after auto-invert
    assert float(display_img[10:18, 10:18].max()) > 0.5


def test_predict_pil_image_and_load_inference_model(tmp_path: Path) -> None:
    """Verify load_inference_model from local pretrained/checkpoint and probability calibration."""
    cfg = PipelineConfig()
    model = MNISTConvNet(cfg.model)

    # Save both pretrained safetensors and .pt checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    model.save_pretrained(ckpt_dir)
    ckpt_file = save_checkpoint(
        path=ckpt_dir / "checkpoint_best.pt",
        model=model,
        optimizer=None,
        epoch=3,
        best_val_accuracy=0.991,
        cfg=cfg,
    )

    loaded_model, source = load_inference_model(
        repo_id_or_path=ckpt_dir,
        checkpoint_fallback=ckpt_file,
        device="cpu",
    )
    assert source.startswith("local_pretrained:")

    canvas = np.zeros((40, 40), dtype=np.uint8)
    canvas[8:32, 18:22] = 250
    pil_img = Image.fromarray(canvas, mode="L")

    result = predict_pil_image(loaded_model, pil_img, device="cpu")
    assert 0 <= result.predicted_digit <= 9
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 10
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_tensor_invalid_shape_raises() -> None:
    """Verify predict_tensor rejects tensors with invalid spatial dimensions."""
    model = MNISTConvNet()
    bad_tensor = torch.zeros(1, 1, 16, 16)
    with pytest.raises(ValueError, match="Expected image_tensor shape"):
        predict_tensor(model, bad_tensor)
