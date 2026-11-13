"""Image preprocessing, Hugging Face Hub model loading, and single/batch digit inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from mnist_pipeline.model import MNISTConvNet, load_checkpoint

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


def preprocess_pil_image(
    image: Image.Image,
    invert: bool = False,
    auto_invert: bool = False,
    center_crop_foreground: bool = True,
    mean: float = 0.1307,
    std: float = 0.3081,
) -> tuple[torch.Tensor, np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
    """Convert an arbitrary PIL image into a normalized `(1, 1, 28, 28)` MNIST tensor.

    MNIST digits are white foreground strokes on a black background, centered inside
    a `20x20` bounding box padded to `28x28`.

    Args:
        image: Input PIL image (RGB, RGBA, or L).
        invert: Explicitly invert pixel intensities (`255 - x`).
        auto_invert: Automatically invert if the border/median background is bright (`> 127`).
        center_crop_foreground: If True and non-zero foreground strokes exist, crop
            around the tight foreground bounding box, resize the longest side to 20px,
            and center-pad into a `28x28` canvas matching canonical MNIST formatting.
        mean: Normalization mean.
        std: Normalization standard deviation.

    Returns:
        Tuple of:
          - Normalized `torch.float32` tensor of shape `(1, 1, 28, 28)`.
          - Unnormalized `28x28` `np.float32` array in `[0.0, 1.0]` for visualization.
    """
    # Handle RGBA transparency by compositing onto a black or white background
    if image.mode == "RGBA":
        arr_rgba = np.asarray(image, dtype=np.uint8)
        alpha = arr_rgba[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr_rgba[:, :, :3].astype(np.float32)
        # Estimate whether RGB strokes are dark on transparent or white on dark
        gray_est = rgb.mean(axis=2, keepdims=True)
        bg_val = 255.0 if (invert or (auto_invert and float(gray_est.mean()) < 128.0)) else 0.0
        composited = rgb * alpha + bg_val * (1.0 - alpha)
        gray_img = Image.fromarray(composited.astype(np.uint8)).convert("L")
    else:
        gray_img = image.convert("L")

    gray_np = np.asarray(gray_img, dtype=np.uint8)
    if invert or (auto_invert and float(np.median(gray_np)) > 127.0):
        gray_img = ImageOps.invert(gray_img)
        gray_np = np.asarray(gray_img, dtype=np.uint8)

    if center_crop_foreground:
        # Locate foreground pixels above noise threshold
        ys, xs = np.where(gray_np > 25)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = int(np.min(xs)), int(np.max(xs)) + 1
            y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
            cropped = gray_img.crop((x0, y0, x1, y1))
            w, h = cropped.size
            max_side = max(w, h, 1)
            new_w = max(1, int(round(w * 20.0 / max_side)))
            new_h = max(1, int(round(h * 20.0 / max_side)))
            resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)
            canvas = Image.new("L", (28, 28), color=0)
            paste_x = (28 - new_w) // 2
            paste_y = (28 - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
            gray_img = canvas
        else:
            gray_img = gray_img.resize((28, 28), Image.Resampling.BILINEAR)
    else:
        gray_img = gray_img.resize((28, 28), Image.Resampling.BILINEAR)

    unit_np: np.ndarray[tuple[int, int], np.dtype[np.float32]] = (
        np.asarray(gray_img, dtype=np.float32) / 255.0
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
) -> tuple[MNISTConvNet, str]:
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
        model = MNISTConvNet.from_pretrained(
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
            model = MNISTConvNet.from_pretrained(ckpt_path.parent, map_location=device)
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
        model = MNISTConvNet.from_pretrained(
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
    model: MNISTConvNet,
    image_tensor: torch.Tensor,
    device: str | torch.device = "cpu",
    mean: float = 0.1307,
    std: float = 0.3081,
) -> PredictionResult:
    """Run digit classification inference on a normalized `(1, 1, 28, 28)` or `(1, 28, 28)` tensor.

    Args:
        model: Loaded `MNISTConvNet` instance.
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
    model: MNISTConvNet,
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
