"""Hydra CLI entrypoint for uploading trained MNIST model and cards to Hugging Face Hub."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mnist_pipeline.config import to_pipeline_config
from mnist_pipeline.hf_hub import upload_to_hf_hub

LOGGER = logging.getLogger(__name__)
_CONFIG_DIR = str((Path(__file__).resolve().parent.parent / "configs").resolve())


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Validate configuration and upload model + dataset cards to Hugging Face Hub.

    Args:
        cfg: Composed Hydra `DictConfig`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipeline_cfg = to_pipeline_config(cfg)
    summary = upload_to_hf_hub(pipeline_cfg)
    LOGGER.info(
        "HF Hub upload complete (dry_run=%s): model_files=%s | dataset_files=%s",
        summary["dry_run"],
        summary["staged_model_files"],
        summary["staged_dataset_files"],
    )


if __name__ == "__main__":
    main()
