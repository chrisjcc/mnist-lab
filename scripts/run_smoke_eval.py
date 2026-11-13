#!/usr/bin/env python3
"""Fast (< 2.0s) zero-budget in-memory smoke evaluation for CI and pre-commit hooks."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from mnist_pipeline.config import PipelineConfig, validate_config
from mnist_pipeline.data import (
    generate_synthetic_mnist_tensors,
    stratified_train_val_split,
    validate_batch_contract,
)
from mnist_pipeline.model import (
    MNISTConvNet,
    count_trainable_parameters,
    load_checkpoint,
    save_checkpoint,
)
from mnist_pipeline.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_smoke_verification() -> dict[str, float | int]:
    """Run in-memory smoke verification across config, data, model, and serialization.

    Returns:
        Dictionary of smoke evaluation timing and parameter metrics.
    """
    start = time.perf_counter()
    cfg = PipelineConfig()
    validate_config(cfg)
    set_seed(cfg.train.seed, deterministic=True)

    images, labels = generate_synthetic_mnist_tensors(num_samples=100, seed=cfg.data.seed)
    validate_batch_contract(images, labels)
    train_idx, val_idx = stratified_train_val_split(labels, val_fraction=0.2, seed=cfg.data.seed)
    if len(train_idx) != 80 or len(val_idx) != 20:
        raise RuntimeError("Stratified split size mismatch in smoke evaluation.")

    model = MNISTConvNet(cfg.model)
    num_params = count_trainable_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images[:16])
    loss = criterion(logits, labels[:16])
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss encountered during smoke evaluation.")
    loss.backward()
    optimizer.step()

    model.eval()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_file = save_checkpoint(
            path=tmp_path / "ckpt.pt",
            model=model,
            optimizer=optimizer,
            epoch=1,
            best_val_accuracy=0.95,
            cfg=cfg,
        )
        reloaded_model, _ = load_checkpoint(ckpt_file, device="cpu")
        reloaded_model.eval()

        hf_dir = model.save_pretrained(tmp_path / "hf_model")
        hf_model = MNISTConvNet.from_pretrained(hf_dir, map_location="cpu")

        with torch.inference_mode():
            ref_out = model(images[:8])
            ckpt_out = reloaded_model(images[:8])
            hf_out = hf_model(images[:8])

        if not torch.allclose(ref_out, ckpt_out, atol=1e-6):
            raise RuntimeError("Checkpoint round-trip logit mismatch in smoke evaluation.")
        if not torch.allclose(ref_out, hf_out, atol=1e-6):
            raise RuntimeError("Safetensors round-trip logit mismatch in smoke evaluation.")

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "Zero-budget smoke evaluation PASSED in %.1f ms (trainable_params=%d)",
        elapsed_ms,
        num_params,
    )
    return {"elapsed_ms": elapsed_ms, "trainable_params": num_params}


if __name__ == "__main__":
    run_smoke_verification()
