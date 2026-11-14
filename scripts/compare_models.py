#!/usr/bin/env python3
"""CLI utility to compare evaluation and benchmark metrics across multiple model architectures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mnist_pipeline.comparison import generate_model_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("compare_models")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare trained models across evaluation and benchmark metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "CNN (MNISTConvNet)=artifacts/models/mnist_cnn",
            "Transformer (MNISTViT)=artifacts/models/mnist_vit",
        ],
        help="List of model specifications in 'Label=path/to/artifacts_dir' format.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/comparisons",
        help="Directory to save the generated comparison report and plots.",
    )
    parser.add_argument(
        "--copy-to-docs",
        default="docs/experiments/cnn_vs_transformer_study.md",
        help="Optional destination path to copy the generated report to in docs/.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute model comparison CLI workflow."""
    args = parse_args()

    models_dict: dict[str, Path] = {}
    for spec in args.models:
        if "=" in spec:
            label, path_str = spec.split("=", 1)
        else:
            p_val = Path(spec)
            label = p_val.name
            path_str = spec
        models_dict[label.strip()] = Path(path_str.strip())

    LOGGER.info("Starting systematic model comparison for %d models:", len(models_dict))
    for label, path in models_dict.items():
        LOGGER.info("  - %s: %s", label, path)

    try:
        report_path, plot_paths = generate_model_comparison(
            models_dict=models_dict,
            output_dir=args.output_dir,
            copy_to_docs=args.copy_to_docs,
        )
        LOGGER.info("Successfully generated model comparison report at: %s", report_path)
        for p in plot_paths:
            LOGGER.info("  Plot: %s", p)
        if args.copy_to_docs:
            LOGGER.info("Successfully mirrored study to: %s", args.copy_to_docs)
    except Exception as exc:
        LOGGER.error("Failed to generate model comparison: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
