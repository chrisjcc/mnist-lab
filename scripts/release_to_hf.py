"""CLI script for rendering and optionally publishing Hugging Face Model and Dataset Cards."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mnist_pipeline.config import load_hydra_config
from mnist_pipeline.hf_hub import upload_to_hf_hub
from mnist_pipeline.reporting import generate_all_reports

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Render Model/Dataset cards and optionally publish the release to Hugging Face Hub."""
    parser = argparse.ArgumentParser(
        description="Render and optionally publish Hugging Face Model and Dataset Cards."
    )
    parser.add_argument(
        "--card-only",
        action="store_true",
        help="Render docs/model_card.md and docs/dataset_card.md locally without uploading.",
    )
    parser.add_argument(
        "--publish-cards",
        action="store_true",
        help="Render cards and upload model + dataset repositories to Hugging Face Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage upload payloads locally without making network requests to HF Hub.",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default=None,
        help="Optional override for hf.repo_id.",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=None,
        help="Optional override for hf.dataset_repo_id.",
    )
    parser.add_argument(
        "--release-tag",
        type=str,
        default=None,
        help="Optional release version tag for logging provenance.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parent.parent
    overrides: list[str] = []
    if args.dry_run or (args.card_only and not args.publish_cards):
        overrides.append("hf.dry_run=true")
    if args.model_repo:
        overrides.append(f"hf.repo_id={args.model_repo}")
    if args.dataset_repo:
        overrides.append(f"hf.dataset_repo_id={args.dataset_repo}")

    cfg = load_hydra_config(config_dir=repo_root / "configs", overrides=overrides)
    outputs = generate_all_reports(
        cfg,
        template_dir=repo_root / "templates",
        docs_dir=repo_root / "docs",
    )
    LOGGER.info(
        "Rendered cards (release_tag=%s): model_card=%s | dataset_card=%s",
        args.release_tag or "HEAD",
        outputs["model_card"],
        outputs["dataset_card"],
    )

    if args.publish_cards:
        summary = upload_to_hf_hub(cfg)
        LOGGER.info("Published release to Hugging Face Hub: %s", summary)


if __name__ == "__main__":
    main()
