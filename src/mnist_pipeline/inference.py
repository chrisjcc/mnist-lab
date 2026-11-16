"""Image preprocessing, Hugging Face Hub model loading, and single/batch digit inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

from mnist_pipeline.model import MNISTConvNet, load_checkpoint, load_model_from_pretrained

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Structured output from single-image digit classification inference.

    Attributes:
        predicted_digit: Predicted integer digit in `[0, 9]`.
        confidence: Calibrated softmax probability of the top predicted digit in `[0.0, 1.0]`.
        probabilities: Mapping from digit string (`"0"`–`"9"`) to softmax probability.
        preprocessed_tensor: Normalized tensor of shape `(1, 1, 28, 28)` fed to the model.
        display_image: Unnormalized `28x28` float32 numpy array in `[0.0, 1.0]` for UI preview.
    """

    predicted_digit: int
    confidence: float
    probabilities: dict[str, float]
    preprocessed_tensor: torch.Tensor
    display_image: np.ndarray[tuple[int, int], np.dtype[np.float32]]


def preprocess_canvas_for_mnist(
    canvas_img: np.ndarray[Any, Any] | Image.Image,
    mnist_mean: float = 0.1307,
    mnist_std: float = 0.3081,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Convert a web canvas drawing into an MNIST-normalized (1, 1, 28, 28) float32 tensor.

    Replicates the canonical NIST -> MNIST geometric normalization pipeline:
      1. RGBA transparency & polarity resolution (white stroke on black background).
      2. Threshold background noise (> 25) and crop tight foreground bounding box.
      3. Aspect-ratio preserving Lanczos resize so longest side is exactly 20 pixels.
      4. Normalize peak stroke brightness to 255.0 (recovers from downsampling blur).
      5. Placement inside 28x28 frame and shift Center-of-Mass (intensity centroid) to (13.5, 13.5).
      6. Standard MNIST z-score normalization ((x / 255.0 - mean) / std).

    Args:
        canvas_img: Raw canvas input as a numpy array (RGB, RGBA, or uint8) or PIL Image.
        mnist_mean: Normalization mean (default: 0.1307).
        mnist_std: Normalization standard deviation (default: 0.3081).

    Returns:
        Float32 numpy array of shape (1, 1, 28, 28).
    """
    pil_img = Image.fromarray(canvas_img) if isinstance(canvas_img, np.ndarray) else canvas_img

    # 1. Handle RGBA transparency or RGB background polarity -> white stroke on black (0..255)
    if pil_img.mode == "RGBA":
        arr_rgba = np.asarray(pil_img, dtype=np.float32)
        # If background is transparent, use alpha channel; otherwise convert RGB
        if float(np.min(arr_rgba[..., 3])) < 250.0 and float(np.max(arr_rgba[..., :3])) < 10.0:
            gray = arr_rgba[..., 3]
        else:
            bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(bg, pil_img).convert("L")
            gray = np.asarray(composited, dtype=np.float32)
            if float(np.median(gray)) > 127.0:  # Black ink on white background -> invert
                gray = 255.0 - gray
    else:
        gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
        if float(np.median(gray)) > 127.0:
            gray = 255.0 - gray

    # 2. Threshold background noise and crop tight bounding box
    mask = gray > 25.0
    if not bool(np.any(mask)):
        return np.zeros((1, 1, 28, 28), dtype=np.float32)

    ys, xs = np.where(mask)
    y_min, y_max = int(np.min(ys)), int(np.max(ys)) + 1
    x_min, x_max = int(np.min(xs)), int(np.max(xs)) + 1
    cropped = gray[y_min:y_max, x_min:x_max]

    # 3. Aspect-ratio preserving resize so the longest side is 20 pixels
    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized_pil = Image.fromarray(cropped.astype(np.uint8)).resize(
        (new_w, new_h), resample=Image.Resampling.LANCZOS
    )
    resized = np.asarray(resized_pil, dtype=np.float32)

    # Normalize peak stroke brightness to 255 (matches MNIST stroke intensity)
    max_val = float(np.max(resized))
    if max_val > 0.0:
        resized = (resized / max_val) * 255.0

    # 4. Place inside 28x28 frame and shift Center-of-Mass (CoM) to (13.5, 13.5)
    canvas_28 = np.zeros((28, 28), dtype=np.float32)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas_28[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    total_mass = float(canvas_28.sum())
    if total_mass > 0:
        grid_y, grid_x = np.indices((28, 28))
        com_y = float((grid_y * canvas_28).sum() / total_mass)
        com_x = float((grid_x * canvas_28).sum() / total_mass)
        shift_y = int(round(13.5 - com_y))
        shift_x = int(round(13.5 - com_x))
        canvas_28 = np.roll(canvas_28, shift=(shift_y, shift_x), axis=(0, 1))

    # 5. Scale to [0, 1] and apply standard MNIST normalization
    norm = (canvas_28 / 255.0 - mnist_mean) / mnist_std
    return norm[None, None, :, :].astype(np.float32)


def preprocess_pil_image(
    image: Image.Image,
    invert: bool = False,
    auto_invert: bool = False,
    center_crop_foreground: bool = True,
    mean: float = 0.1307,
    std: float = 0.3081,
) -> tuple[torch.Tensor, np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
    """Convert an arbitrary PIL image into a normalized `(1, 1, 28, 28)` MNIST tensor.

    Replicates the canonical NIST -> MNIST preprocessing pipeline:
      1. RGBA transparency & polarity resolution (white stroke on black background).
      2. Tight bounding-box foreground crop.
      3. Aspect-ratio preserving Lanczos resize to 20x20 box.
      4. Peak stroke brightness normalization to 255.0.
      5. Center-of-Mass (CoM) translation onto (13.5, 13.5) in 28x28 frame.
      6. Standard MNIST z-score normalization ((x / 255.0 - mean) / std).

    Args:
        image: Input PIL image (RGB, RGBA, or L).
        invert: Explicitly invert pixel intensities (`255 - x`).
        auto_invert: Automatically invert if the border/median background is bright (`> 127`).
        center_crop_foreground: If True and non-zero foreground strokes exist, crop
            around the tight foreground bounding box, resize the longest side to 20px,
            and center-pad into a `28x28` canvas with Center-of-Mass alignment.
        mean: Normalization mean.
        std: Normalization standard deviation.

    Returns:
        Tuple of:
          - Normalized `torch.float32` tensor of shape `(1, 1, 28, 28)`.
          - Unnormalized `28x28` `np.float32` array in `[0.0, 1.0]` for visualization.
    """
    # Handle RGBA transparency by compositing onto a black or white background
    if image.mode == "RGBA":
        arr_rgba = np.asarray(image, dtype=np.float32)
        if float(np.min(arr_rgba[..., 3])) < 250.0 and float(np.max(arr_rgba[..., :3])) < 10.0:
            gray = arr_rgba[..., 3]
        else:
            bg_val = (
                0
                if (not invert and (not auto_invert or float(arr_rgba[..., :3].mean()) >= 128.0))
                else 255
            )
            bg = Image.new("RGBA", image.size, (bg_val, bg_val, bg_val, 255))
            composited = Image.alpha_composite(bg, image).convert("L")
            gray = np.asarray(composited, dtype=np.float32)
            if invert or (auto_invert and float(np.median(gray)) > 127.0):
                gray = 255.0 - gray
    else:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        if invert or (auto_invert and float(np.median(gray)) > 127.0):
            gray = 255.0 - gray

    if center_crop_foreground:
        mask = gray > 25.0
        if bool(np.any(mask)):
            ys, xs = np.where(mask)
            y_min, y_max = int(np.min(ys)), int(np.max(ys)) + 1
            x_min, x_max = int(np.min(xs)), int(np.max(xs)) + 1
            cropped = gray[y_min:y_max, x_min:x_max]
            h, w = cropped.shape
            scale = 20.0 / max(h, w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            resized_pil = Image.fromarray(cropped.astype(np.uint8)).resize(
                (new_w, new_h), resample=Image.Resampling.LANCZOS
            )
            resized = np.asarray(resized_pil, dtype=np.float32)
            max_val = float(np.max(resized))
            if max_val > 0.0:
                resized = (resized / max_val) * 255.0

            canvas_28 = np.zeros((28, 28), dtype=np.float32)
            y0 = (28 - new_h) // 2
            x0 = (28 - new_w) // 2
            canvas_28[y0 : y0 + new_h, x0 : x0 + new_w] = resized

            total_mass = float(canvas_28.sum())
            if total_mass > 0:
                grid_y, grid_x = np.indices((28, 28))
                com_y = float((grid_y * canvas_28).sum() / total_mass)
                com_x = float((grid_x * canvas_28).sum() / total_mass)
                shift_y = int(round(13.5 - com_y))
                shift_x = int(round(13.5 - com_x))
                canvas_28 = np.roll(canvas_28, shift=(shift_y, shift_x), axis=(0, 1))
        else:
            canvas_28 = np.zeros((28, 28), dtype=np.float32)
    else:
        resized_pil = Image.fromarray(gray.astype(np.uint8)).resize(
            (28, 28), resample=Image.Resampling.LANCZOS
        )
        canvas_28 = np.asarray(resized_pil, dtype=np.float32)

    unit_np: np.ndarray[tuple[int, int], np.dtype[np.float32]] = (canvas_28 / 255.0).astype(
        np.float32
    )
    norm_np = (unit_np - float(mean)) / float(std)
    tensor = torch.from_numpy(norm_np).to(torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor, unit_np


def load_inference_model(
    repo_id_or_path: str | Path = "chrisjcc/mnist-cnn",
    checkpoint_fallback: str | Path = "artifacts/checkpoints/checkpoint_best.pt",
    revision: str = "main",
    cache_dir: str | Path | None = "artifacts/hf_cache",
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, str]:
    """Load an `MNISTConvNet` for inference with resilient local and Hub fallback.

    Resolution order:
      1. If `repo_id_or_path` is an existing local directory with `model.safetensors`
         and `config.json`, load via `MNISTConvNet.from_pretrained`.
      2. If `checkpoint_fallback` exists locally (`artifacts/checkpoints/checkpoint_best.pt`),
         or its parent directory contains `model.safetensors`, load locally when Hub
         download is unavailable or when running offline.
      3. Attempt to download `model.safetensors` + `config.json` from Hugging Face Hub
         via `MNISTConvNet.from_pretrained(repo_id_or_path, ...)`.
      4. Fall back to an initialized default `MNISTConvNet` so interactive demos remain
         functional even in completely clean offline environments.

    Args:
        repo_id_or_path: Hugging Face Hub repo ID (`"username/repo"`) or local directory.
        checkpoint_fallback: Path to local `.pt` checkpoint fallback.
        revision: Hub git branch, tag, or commit.
        cache_dir: Local directory for caching Hub files.
        device: Target PyTorch device.

    Returns:
        Tuple of `(model_in_eval_mode, source_description)`.
    """
    local_candidate = Path(repo_id_or_path)
    if (
        local_candidate.exists()
        and local_candidate.is_dir()
        and (local_candidate / "model.safetensors").exists()
        and (local_candidate / "config.json").exists()
    ):
        model = load_model_from_pretrained(
            local_candidate,
            map_location=device,
        )
        return model, f"local_pretrained:{local_candidate}"

    if local_candidate.exists() and local_candidate.is_file() and local_candidate.suffix == ".pt":
        model, _ = load_checkpoint(local_candidate, device=device)
        model.eval()
        return model, f"local_checkpoint:{local_candidate}"

    ckpt_path = Path(checkpoint_fallback)
    if ckpt_path.parent.exists() and (ckpt_path.parent / "model.safetensors").exists():
        try:
            model = load_model_from_pretrained(ckpt_path.parent, map_location=device)
            return model, f"local_safetensors:{ckpt_path.parent}"
        except Exception as exc:
            LOGGER.debug("Local safetensors load failed (%s); trying .pt checkpoint.", exc)

    if ckpt_path.exists():
        try:
            model, _ = load_checkpoint(ckpt_path, device=device)
            model.eval()
            return model, f"local_checkpoint:{ckpt_path}"
        except Exception as exc:
            LOGGER.warning("Failed to load local checkpoint '%s': %s", ckpt_path, exc)

    try:
        model = load_model_from_pretrained(
            str(repo_id_or_path),
            revision=revision,
            cache_dir=cache_dir,
            map_location=device,
        )
        return model, f"hf_hub:{repo_id_or_path}@{revision}"
    except Exception as exc:
        LOGGER.warning(
            "Could not download model from Hugging Face Hub '%s' (%s); "
            "using freshly initialized MNISTConvNet fallback.",
            repo_id_or_path,
            exc,
        )
        fallback_model = MNISTConvNet().to(device)
        fallback_model.eval()
        return fallback_model, "untrained_fallback"


def predict_tensor(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: str | torch.device = "cpu",
    mean: float = 0.1307,
    std: float = 0.3081,
) -> PredictionResult:
    """Run digit classification inference on a normalized `(1, 1, 28, 28)` or `(1, 28, 28)` tensor.

    Args:
        model: Loaded PyTorch model instance.
        image_tensor: Normalized float32 tensor of shape `(1, 1, 28, 28)` or `(1, 28, 28)`.
        device: Target compute device.
        mean: Normalization mean (used to reconstruct `display_image`).
        std: Normalization std (used to reconstruct `display_image`).

    Returns:
        Populated `PredictionResult`.
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4 or image_tensor.shape[1:] != (1, 28, 28):
        raise ValueError(
            f"Expected image_tensor shape (1, 1, 28, 28), got {tuple(image_tensor.shape)}"
        )

    model.eval()
    target_device = torch.device(device)
    model.to(target_device)

    with torch.inference_mode():
        logits = model(image_tensor.to(target_device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()

    prob_dict = {str(i): round(float(probs[i].item()), 6) for i in range(int(probs.numel()))}
    top_conf, top_idx = probs.max(dim=0)

    raw_slice = image_tensor[0, 0].detach().cpu().numpy()
    display_img: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.clip(
        raw_slice * float(std) + float(mean), 0.0, 1.0
    ).astype(np.float32)

    return PredictionResult(
        predicted_digit=int(top_idx.item()),
        confidence=round(float(top_conf.item()), 6),
        probabilities=prob_dict,
        preprocessed_tensor=image_tensor.detach().cpu(),
        display_image=display_img,
    )


def predict_pil_image(
    model: nn.Module,
    image: Image.Image,
    invert: bool = False,
    auto_invert: bool = False,
    center_crop_foreground: bool = True,
    device: str | torch.device = "cpu",
    mean: float = 0.1307,
    std: float = 0.3081,
) -> PredictionResult:
    """Preprocess a PIL image and run digit classification inference.

    Args:
        model: Loaded `MNISTConvNet` instance.
        image: Input PIL image.
        invert: Explicitly invert foreground/background polarity.
        auto_invert: Automatically invert if background is bright.
        center_crop_foreground: Center-crop and scale foreground bounding box to MNIST canonical layout.
        device: Target compute device.
        mean: Normalization mean.
        std: Normalization std.

    Returns:
        Populated `PredictionResult`.
    """
    tensor, display_img = preprocess_pil_image(
        image=image,
        invert=invert,
        auto_invert=auto_invert,
        center_crop_foreground=center_crop_foreground,
        mean=mean,
        std=std,
    )
    res = predict_tensor(model=model, image_tensor=tensor, device=device, mean=mean, std=std)
    return PredictionResult(
        predicted_digit=res.predicted_digit,
        confidence=res.confidence,
        probabilities=res.probabilities,
        preprocessed_tensor=res.preprocessed_tensor,
        display_image=display_img,
    )
