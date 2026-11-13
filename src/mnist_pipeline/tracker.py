"""Weights & Biases experiment tracking wrapper supporting online, offline, and disabled modes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import wandb

from mnist_pipeline.config import PipelineConfig, WandbConfig, config_to_dict

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Wrapper around Weights & Biases for metric, artifact, and plot logging.

    Gracefully becomes a no-op when `wandb.enabled=False` or `wandb.mode='disabled'`.
    """

    def __init__(
        self,
        wandb_cfg: WandbConfig,
        full_cfg: PipelineConfig | None = None,
        job_type: str = "train",
    ) -> None:
        """Initialize the ExperimentTracker without starting a W&B run until start() is called.

        Args:
            wandb_cfg: WandbConfig specifying project, entity, mode, and tags.
            full_cfg: Optional full PipelineConfig to log as run configuration.
            job_type: Logical stage identifier ('train', 'evaluate', 'benchmark').
        """
        self.cfg = wandb_cfg
        self.full_cfg = full_cfg
        self.job_type = job_type
        self._run: Any = None
        self.active: bool = False

    def start(self) -> None:
        """Initialize the underlying W&B run if enabled."""
        mode = self.cfg.mode.lower()
        if not self.cfg.enabled or mode == "disabled":
            logger.info("W&B tracking disabled (enabled=%s, mode=%s)", self.cfg.enabled, mode)
            self.active = False
            return

        config_payload = config_to_dict(self.full_cfg) if self.full_cfg is not None else {}
        try:
            self._run = wandb.init(
                project=self.cfg.project,
                entity=self.cfg.entity,
                name=self.cfg.run_name,
                tags=list(self.cfg.tags),
                mode=mode,  # type: ignore[arg-type]
                job_type=self.job_type,
                config=config_payload,
                reinit=True,
            )
            self.active = self._run is not None
            logger.info("Initialized W&B run (project=%s, mode=%s)", self.cfg.project, mode)
        except Exception as exc:
            logger.warning("Failed to initialize W&B (%s); continuing with tracking disabled.", exc)
            self._run = None
            self.active = False

    def log_metrics(self, metrics: dict[str, float | int], step: int | None = None) -> None:
        """Log scalar metrics to the active W&B run.

        Args:
            metrics: Dictionary mapping metric names to scalar float/int values.
            step: Optional global step or epoch index.
        """
        if not self.active or self._run is None:
            return
        try:
            wandb.log(metrics, step=step)
        except Exception as exc:
            logger.warning("Failed to log metrics to W&B: %s", exc)

    def log_summary(self, summary_metrics: dict[str, Any]) -> None:
        """Record summary key-value pairs on the active W&B run.

        Args:
            summary_metrics: Dictionary of summary metrics to persist on the run.
        """
        if not self.active or self._run is None:
            return
        try:
            for key, value in summary_metrics.items():
                self._run.summary[key] = value
        except Exception as exc:
            logger.warning("Failed to update W&B summary: %s", exc)

    def log_image(self, key: str, image_path: str | Path, caption: str | None = None) -> None:
        """Log a PNG/JPEG plot file as a W&B Image.

        Args:
            key: W&B media key (e.g. 'plots/confusion_matrix').
            image_path: Path to the image file on disk.
            caption: Optional image caption.
        """
        if not self.active or self._run is None:
            return
        path_obj = Path(image_path)
        if not path_obj.exists():
            return
        try:
            wandb.log({key: wandb.Image(str(path_obj), caption=caption)})
        except Exception as exc:
            logger.warning("Failed to log image '%s' to W&B: %s", path_obj, exc)

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Upload a checkpoint or report file as a versioned W&B Artifact.

        Args:
            name: Artifact identifier string.
            artifact_type: Type category (e.g., 'model', 'evaluation').
            file_path: Local file path to attach.
            metadata: Optional metadata dictionary.
        """
        if not self.active or self._run is None or not self.cfg.log_model:
            return
        path_obj = Path(file_path)
        if not path_obj.exists():
            return
        try:
            artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
            artifact.add_file(str(path_obj))
            self._run.log_artifact(artifact)
        except Exception as exc:
            logger.warning("Failed to log artifact '%s' to W&B: %s", name, exc)

    def finish(self) -> None:
        """Close the active W&B run cleanly."""
        if not self.active or self._run is None:
            return
        try:
            wandb.finish()
        except Exception as exc:
            logger.warning("Error while finishing W&B run: %s", exc)
        finally:
            self._run = None
            self.active = False
