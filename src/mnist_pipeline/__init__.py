"""MNIST-Lab: Reproducible, configuration-driven PyTorch MNIST classification pipeline."""

from __future__ import annotations

__version__ = "0.1.0"

from mnist_pipeline.inference import preprocess_canvas_for_mnist

__all__ = ["__version__", "preprocess_canvas_for_mnist"]
