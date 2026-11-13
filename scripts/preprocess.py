#!/usr/bin/env python3
"""Stage 1 CLI entrypoint: Download, normalize, and deterministically split MNIST data."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from mnist_pipeline.config import to_pipeline_config
from mnist_pipeline.data import preprocess_mnist_data
from mnist_pipeline.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Execute the MNIST data download and deterministic preprocessing stage.

    Args:
        cfg: Hydra DictConfig composed from configs/config.yaml.
    """
    pipeline_cfg = to_pipeline_config(cfg)
    set_seed(pipeline_cfg.data.seed, deterministic=pipeline_cfg.train.deterministic)
    metadata = preprocess_mnist_data(pipeline_cfg.data)
    splits = metadata["splits"]
    logger.info(
        "Preprocessing complete: train=%d, val=%d, test=%d",
        splits["train"]["num_examples"],
        splits["val"]["num_examples"],
        splits["test"]["num_examples"],
    )


if __name__ == "__main__":
    main()
