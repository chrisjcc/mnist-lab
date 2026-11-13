"""Hydra CLI entrypoint for downloading or loading a trained MNIST model and running inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image

from mnist_pipeline.config import to_pipeline_config
from mnist_pipeline.data import generate_synthetic_digit_template, load_processed_split
from mnist_pipeline.inference import (
    load_inference_model,
    predict_pil_image,
    predict_tensor,
)
from mnist_pipeline.utils import resolve_device

LOGGER = logging.getLogger(__name__)
_CONFIG_DIR = str((Path(__file__).resolve().parent.parent / "configs").resolve())


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Load the trained model from HF Hub (or local fallback) and run digit inference.

    Args:
        cfg: Composed Hydra `DictConfig`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipeline_cfg = to_pipeline_config(cfg)
    device = resolve_device(pipeline_cfg.train.device)

    model, source = load_inference_model(
        repo_id_or_path=pipeline_cfg.hf.repo_id,
        checkpoint_fallback=Path(pipeline_cfg.paths.checkpoints_dir) / "checkpoint_best.pt",
        revision=pipeline_cfg.hf.revision,
        cache_dir=pipeline_cfg.hf.cache_dir,
        device=device,
    )
    LOGGER.info("Loaded inference model from source='%s' on device='%s'", source, device)

    true_label: int | None = None
    if pipeline_cfg.hf.image_path:
        img_path = Path(pipeline_cfg.hf.image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Inference image not found at '{img_path}'")
        pil_img = Image.open(img_path)
        result = predict_pil_image(
            model=model,
            image=pil_img,
            auto_invert=True,
            device=device,
            mean=pipeline_cfg.data.mean,
            std=pipeline_cfg.data.std,
        )
    else:
        processed_test = Path(pipeline_cfg.data.processed_dir) / "test.pt"
        idx = max(0, int(pipeline_cfg.hf.sample_index))
        if processed_test.exists():
            test_images, test_labels = load_processed_split(pipeline_cfg.data.processed_dir, "test")
            sample_idx = idx % int(test_images.shape[0])
            sample_tensor = test_images[sample_idx : sample_idx + 1]
            true_label = int(test_labels[sample_idx].item())
        else:
            digit = idx % 10
            raw_tpl = generate_synthetic_digit_template(digit)
            sample_tensor = ((raw_tpl - pipeline_cfg.data.mean) / pipeline_cfg.data.std).unsqueeze(
                0
            )
            true_label = digit

        result = predict_tensor(
            model=model,
            image_tensor=sample_tensor,
            device=device,
            mean=pipeline_cfg.data.mean,
            std=pipeline_cfg.data.std,
        )

    output_payload = {
        "model_source": source,
        "predicted_digit": result.predicted_digit,
        "true_digit": true_label,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
    }
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
