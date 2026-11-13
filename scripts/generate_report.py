"""Hydra CLI entrypoint for generating evaluation plots, Markdown report, and Hub cards."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mnist_pipeline.config import to_pipeline_config
from mnist_pipeline.reporting import generate_all_reports

LOGGER = logging.getLogger(__name__)
_CONFIG_DIR = str((Path(__file__).resolve().parent.parent / "configs").resolve())


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate plots, Markdown evaluation report, and Model/Dataset cards.

    Args:
        cfg: Composed Hydra `DictConfig`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipeline_cfg = to_pipeline_config(cfg)
    repo_root = Path(__file__).resolve().parent.parent
    outputs = generate_all_reports(
        pipeline_cfg,
        template_dir=repo_root / "templates",
        docs_dir=repo_root / "docs",
    )
    for key, val in outputs.items():
        LOGGER.info("Generated %s -> %s", key, val)


if __name__ == "__main__":
    main()
