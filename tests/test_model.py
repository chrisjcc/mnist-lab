"""Unit tests for MNISTConvNet forward pass, backward gradient flow, and parameter counting."""

from pathlib import Path

import torch
import torch.nn as nn

from mnist_pipeline.config import ModelConfig
from mnist_pipeline.model import (
    MNISTConvNet,
    MNISTViT,
    build_model,
    count_total_parameters,
    count_trainable_parameters,
    load_checkpoint,
    load_model_from_pretrained,
    save_checkpoint,
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


def test_vit_forward_shape_and_finite_logits() -> None:
    set_seed(42)
    cfg = ModelConfig(
        name="mnist_vit",
        architecture="transformer",
        patch_size=4,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
    )
    vit = MNISTViT(cfg)
    vit.eval()
    inputs = torch.randn(4, 1, 28, 28)
    with torch.inference_mode():
        logits = vit(inputs)
    assert logits.shape == (4, 10)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_vit_backward_produces_finite_gradients() -> None:
    set_seed(42)
    cfg = ModelConfig(
        name="mnist_vit",
        architecture="transformer",
        patch_size=4,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
    )
    vit = MNISTViT(cfg)
    vit.train()
    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,), dtype=torch.int64)
    loss = nn.CrossEntropyLoss()(vit(inputs), targets)
    loss.backward()

    for name, param in vit.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for ViT parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


def test_build_model_factory() -> None:
    cnn_cfg = ModelConfig(name="mnist_cnn", architecture="cnn")
    cnn = build_model(cnn_cfg)
    assert isinstance(cnn, MNISTConvNet)

    vit_cfg = ModelConfig(name="mnist_vit", architecture="transformer")
    vit = build_model(vit_cfg)
    assert isinstance(vit, MNISTViT)


def test_vit_save_and_from_pretrained_roundtrip(tmp_path: Path) -> None:
    cfg = ModelConfig(
        name="mnist_vit",
        architecture="transformer",
        patch_size=4,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
    )
    vit = MNISTViT(cfg)
    save_dir = tmp_path / "vit_pretrained"
    vit.save_pretrained(save_dir)

    assert (save_dir / "config.json").exists()
    assert (save_dir / "model.safetensors").exists()

    loaded = MNISTViT.from_pretrained(save_dir)
    assert isinstance(loaded, MNISTViT)
    assert loaded.cfg.patch_size == 4
    assert loaded.cfg.embed_dim == 64

    # Test dispatch loader
    dispatched = load_model_from_pretrained(save_dir)
    assert isinstance(dispatched, MNISTViT)

    inputs = torch.randn(2, 1, 28, 28)
    vit.eval()
    with torch.inference_mode():
        out_orig = vit(inputs)
        out_loaded = loaded(inputs)
        out_disp = dispatched(inputs)
    assert torch.allclose(out_orig, out_loaded, atol=1e-5)
    assert torch.allclose(out_orig, out_disp, atol=1e-5)


def test_save_and_load_checkpoint_vit(tmp_path: Path) -> None:
    from mnist_pipeline.config import PipelineConfig

    cfg = PipelineConfig()
    cfg.model = ModelConfig(
        name="mnist_vit",
        architecture="transformer",
        patch_size=4,
        embed_dim=64,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
    )
    vit = MNISTViT(cfg.model)
    opt = torch.optim.AdamW(vit.parameters(), lr=1e-3)
    ckpt_file = tmp_path / "vit_best.pt"

    save_checkpoint(ckpt_file, vit, opt, epoch=3, best_val_accuracy=0.98, cfg=cfg)
    assert ckpt_file.exists()

    loaded_model, payload = load_checkpoint(ckpt_file)
    assert isinstance(loaded_model, MNISTViT)
    assert payload["epoch"] == 3
    assert payload["best_val_accuracy"] == 0.98
