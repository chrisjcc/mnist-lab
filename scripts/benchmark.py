"""Hydra CLI entrypoint for inference latency, throughput, and footprint benchmarking."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mnist_pipeline.config import to_pipeline_config
from mnist_pipeline.evaluate import benchmark_checkpoint

LOGGER = logging.getLogger(__name__)
_CONFIG_DIR = str((Path(__file__).resolve().parent.parent / "configs").resolve())


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Validate configuration and benchmark `checkpoint_best.pt` across batch sizes.

    Args:
        cfg: Composed Hydra `DictConfig`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipeline_cfg = to_pipeline_config(cfg)
    summary = benchmark_checkpoint(pipeline_cfg)
    LOGGER.info(
        "Benchmark finished on device=%s | trainable_params=%d | checkpoint_size_mb=%.3f",
        summary["device"],
        summary["trainable_params"],
        summary["checkpoint_size_mb"],
    )


if __name__ == "__main__":
    main()
