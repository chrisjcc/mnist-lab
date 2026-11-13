"""Hydra CLI entrypoint for held-out test set evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from mnist_pipeline.config import to_pipeline_config
from mnist_pipeline.evaluate import evaluate_checkpoint

LOGGER = logging.getLogger(__name__)
_CONFIG_DIR = str((Path(__file__).resolve().parent.parent / "configs").resolve())


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Validate configuration and evaluate `checkpoint_best.pt` on the held-out test set.

    Args:
        cfg: Composed Hydra `DictConfig`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    pipeline_cfg = to_pipeline_config(cfg)
    metrics = evaluate_checkpoint(pipeline_cfg)
    LOGGER.info(
        "Evaluation finished: test_acc=%.4f | macro_f1=%.4f | test_loss=%.4f",
        metrics["test_acc"],
        metrics["macro_f1"],
        metrics["test_loss"],
    )


if __name__ == "__main__":
    main()
