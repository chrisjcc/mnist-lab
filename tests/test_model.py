"""Unit tests for MNISTConvNet forward pass, backward gradient flow, and parameter counting."""

from __future__ import annotations

import torch
import torch.nn as nn

from mnist_pipeline.config import ModelConfig
from mnist_pipeline.model import (
    MNISTConvNet,
    count_total_parameters,
    count_trainable_parameters,
)
from mnist_pipeline.utils import set_seed


def test_model_forward_shape_and_finite_logits() -> None:
    set_seed(42)
    model = MNISTConvNet(ModelConfig())
    model.eval()
    inputs = torch.randn(8, 1, 28, 28)
    with torch.inference_mode():
        logits = model(inputs)
    assert logits.shape == (8, 10)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_model_backward_produces_finite_gradients() -> None:
    set_seed(42)
    model = MNISTConvNet(ModelConfig())
    model.train()
    inputs = torch.randn(8, 1, 28, 28)
    targets = torch.randint(0, 10, (8,), dtype=torch.int64)
    loss = nn.CrossEntropyLoss()(model(inputs), targets)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_parameter_counts_are_consistent() -> None:
    model = MNISTConvNet(ModelConfig(conv_channels=[16, 32], hidden_dim=64))
    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    assert trainable > 0
    assert trainable == total
