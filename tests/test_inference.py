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
    assert float(np.max(display_img[10:18, 10:18])) > 0.5


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


def test_preprocess_canvas_for_mnist() -> None:
    """Verify canonical MNIST bounding box crop, 20x20 scale, and Center-of-Mass alignment."""
    from mnist_pipeline.inference import preprocess_canvas_for_mnist

    # 1. Off-center drawing in top-left of a 280x280 canvas
    arr = np.zeros((280, 280, 4), dtype=np.uint8)
    arr[20:40, 20:180, :3] = 255
    arr[40:180, 140:160, :3] = 255
    arr[:, :, 3] = 255

    res = preprocess_canvas_for_mnist(arr)
    assert res.shape == (1, 1, 28, 28)
    assert res.dtype == np.float32
    assert np.isfinite(res).all()

    # Reconstruct unnormalized intensities to compute center of mass
    unnorm = np.clip(res[0, 0] * 0.3081 + 0.1307, 0.0, 1.0)
    assert float(np.max(unnorm)) > 0.8
    total_mass = float(unnorm.sum())
    assert total_mass > 0.0
    grid_y, grid_x = np.indices((28, 28))
    com_y = float((grid_y * unnorm).sum() / total_mass)
    com_x = float((grid_x * unnorm).sum() / total_mass)
    # Centered near (13.5, 13.5) despite being drawn in the top-left
    assert abs(com_y - 13.5) <= 1.5
    assert abs(com_x - 13.5) <= 1.5

    # 2. Blank canvas returns zeros
    blank = np.zeros((100, 100, 4), dtype=np.uint8)
    res_blank = preprocess_canvas_for_mnist(blank)
    assert res_blank.shape == (1, 1, 28, 28)
    assert (res_blank == 0.0).all()

    # 3. Transparent RGBA canvas with stroke in alpha channel
    alpha_arr = np.zeros((120, 120, 4), dtype=np.uint8)
    alpha_arr[30:90, 50:70, 3] = 255
    res_alpha = preprocess_canvas_for_mnist(alpha_arr)
    assert res_alpha.shape == (1, 1, 28, 28)
    assert float(np.max(res_alpha)) > 0.0


def test_load_inference_model_vit_fallback(tmp_path: Path) -> None:
    """Verify load_inference_model defaults to MNISTViT when ViT repo is specified without Hub access."""
    from mnist_pipeline.model import MNISTViT

    nonexistent_ckpt = tmp_path / "nonexistent" / "checkpoint_best.pt"
    model, source = load_inference_model(
        repo_id_or_path="chrisjcc/mnist-vit",
        checkpoint_fallback=nonexistent_ckpt,
        device="cpu",
    )
    assert isinstance(model, MNISTViT)
    assert source == "untrained_fallback"


def test_app_model_presets() -> None:
    """Verify Streamlit application MODEL_PRESETS catalog contains valid architectures and paths."""
    from app.app import MODEL_PRESETS

    assert "CNN (Convolutional)" in MODEL_PRESETS
    assert "ViT (Vision Transformer)" in MODEL_PRESETS
    assert "Custom Model / Repo" in MODEL_PRESETS

    cnn = MODEL_PRESETS["CNN (Convolutional)"]
    assert cnn["repo_id"] == "chrisjcc/mnist-cnn"
    assert "mnist_cnn" in cnn["checkpoint_fallback"]

    vit = MODEL_PRESETS["ViT (Vision Transformer)"]
    assert vit["repo_id"] == "chrisjcc/mnist-vit"
    assert "mnist_vit" in vit["checkpoint_fallback"]
