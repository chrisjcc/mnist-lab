#!/usr/bin/env python3
"""Stage 2 CLI entrypoint: Train the MNIST CNN model, save checkpoints, and log to W&B."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from mnist_pipeline.config import to_pipeline_config
from mnist_pipeline.train import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Execute the MNIST model training and validation pipeline stage.

    Args:
        cfg: Hydra DictConfig composed from configs/config.yaml.
    """
    pipeline_cfg = to_pipeline_config(cfg)
    summary = train_model(pipeline_cfg)
    logger.info(
        "Training finished: best_epoch=%d, best_val_accuracy=%.4f, total_time=%.2fs, best_ckpt=%s",
        summary.best_epoch,
        summary.best_val_accuracy,
        summary.total_training_time_sec,
        summary.checkpoint_best_path,
    )


if __name__ == "__main__":
    main()
