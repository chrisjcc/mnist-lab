"""Convolutional neural network architecture, checkpointing, and Hugging Face Hub serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from mnist_pipeline.config import ModelConfig, PipelineConfig, config_to_dict
from mnist_pipeline.utils import load_json, save_json


class MNISTConvNet(nn.Module):
    """Compact two-block convolutional neural network for 28x28 MNIST digit classification.

    Architecture:
        - Block 1: Conv2d(in_channels -> c1, k=3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        - Block 2: Conv2d(c1 -> c2, k=3, pad=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        - Head: Flatten -> Dropout -> Linear(c2 * 7 * 7 -> hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim -> num_classes)
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        """Initialize the MNISTConvNet layers from a ModelConfig.

        Args:
            cfg: Optional ModelConfig specifying channel dimensions, dropout, and classes.
        """
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c1 = self.cfg.conv_channels[0]
        c2 = self.cfg.conv_channels[1] if len(self.cfg.conv_channels) > 1 else c1 * 2
        padding = self.cfg.kernel_size // 2

        self.features = nn.Sequential(
            nn.Conv2d(
                self.cfg.in_channels,
                c1,
                kernel_size=self.cfg.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                c1,
                c2,
                kernel_size=self.cfg.kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Compute flattened feature map size dynamically using a dummy 28x28 tensor
        with torch.no_grad():
            dummy = torch.zeros(1, self.cfg.in_channels, 28, 28)
            flat_dim = int(self.features(dummy).numel())

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.cfg.dropout),
            nn.Linear(flat_dim, self.cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize convolutional and linear layer weights using Kaiming normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of normalized 28x28 grayscale images.

        Args:
            x: Input tensor of shape [N, 1, 28, 28].

        Returns:
            Unnormalized logit tensor of shape [N, num_classes].
        """
        feat: torch.Tensor = self.features(x)
        logits: torch.Tensor = self.classifier(feat)
        return logits

    def save_pretrained(
        self,
        save_directory: str | Path,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save model weights in safetensors format alongside config.json for Hugging Face Hub.

        Args:
            save_directory: Directory where model.safetensors and config.json will be written.
            extra_metadata: Optional metadata dictionary (e.g., normalization stats, metrics).

        Returns:
            Resolved Path to save_directory.
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_safetensors(state_dict, str(out_dir / "model.safetensors"))

        config_payload: dict[str, Any] = {
            "architectures": ["MNISTConvNet"],
            "model_type": self.cfg.name,
            "in_channels": self.cfg.in_channels,
            "conv_channels": list(self.cfg.conv_channels),
            "kernel_size": self.cfg.kernel_size,
            "dropout": self.cfg.dropout,
            "hidden_dim": self.cfg.hidden_dim,
            "num_classes": self.cfg.num_classes,
            "normalization": {"mean": 0.1307, "std": 0.3081},
        }
        if extra_metadata:
            config_payload.update(extra_metadata)

        save_json(config_payload, out_dir / "config.json")
        return out_dir

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        revision: str = "main",
        cache_dir: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> MNISTConvNet:
        """Load an MNISTConvNet from a local directory or a Hugging Face Model Hub repository.

        Args:
            pretrained_model_name_or_path: Local path or Hugging Face Hub repo_id ('user/repo').
            revision: Git revision (branch, tag, or commit) when downloading from the Hub.
            cache_dir: Optional local cache directory for Hugging Face Hub downloads.
            map_location: Target device mapping for loaded tensors.

        Returns:
            Initialized MNISTConvNet in evaluation mode with loaded weights.
        """
        local_path = Path(pretrained_model_name_or_path)
        if local_path.exists() and local_path.is_dir():
            config_file = local_path / "config.json"
            safetensors_file = local_path / "model.safetensors"
            if not config_file.exists() or not safetensors_file.exists():
                raise FileNotFoundError(
                    f"Directory '{local_path}' must contain both config.json and model.safetensors"
                )
        else:
            repo_id = str(pretrained_model_name_or_path)
            cache_str = str(cache_dir) if cache_dir is not None else None
            config_file = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename="config.json",
                    revision=revision,
                    cache_dir=cache_str,
                )
            )
            safetensors_file = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename="model.safetensors",
                    revision=revision,
                    cache_dir=cache_str,
                )
            )

        raw_cfg = load_json(config_file)
        model_cfg = ModelConfig(
            name=str(raw_cfg.get("model_type", "mnist_cnn")),
            in_channels=int(raw_cfg.get("in_channels", 1)),
            conv_channels=[int(c) for c in raw_cfg.get("conv_channels", [32, 64])],
            kernel_size=int(raw_cfg.get("kernel_size", 3)),
            dropout=float(raw_cfg.get("dropout", 0.25)),
            hidden_dim=int(raw_cfg.get("hidden_dim", 128)),
            num_classes=int(raw_cfg.get("num_classes", 10)),
        )
        model = cls(model_cfg)
        device_str = str(map_location)
        state_dict = load_safetensors(str(safetensors_file), device=device_str)
        model.load_state_dict(state_dict)
        model.to(map_location)
        model.eval()
        return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch module.

    Args:
        model: PyTorch module to inspect.

    Returns:
        Total number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count the total number of parameters (trainable and frozen) in a PyTorch module.

    Args:
        model: PyTorch module to inspect.

    Returns:
        Total parameter count.
    """
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    path: str | Path,
    model: MNISTConvNet,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_accuracy: float,
    cfg: PipelineConfig,
) -> Path:
    """Save a training checkpoint containing model, optimizer, epoch, metric, and config state.

    Args:
        path: Target file path for the .pt checkpoint.
        model: Trained MNISTConvNet instance.
        optimizer: Optional optimizer whose state_dict should be saved.
        epoch: Completed epoch number (1-indexed).
        best_val_accuracy: Best validation accuracy achieved up to this epoch.
        cfg: Resolved PipelineConfig for reproducibility.

    Returns:
        Resolved Path to the written checkpoint file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_val_accuracy": best_val_accuracy,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_config": {
            "name": model.cfg.name,
            "in_channels": model.cfg.in_channels,
            "conv_channels": list(model.cfg.conv_channels),
            "kernel_size": model.cfg.kernel_size,
            "dropout": model.cfg.dropout,
            "hidden_dim": model.cfg.hidden_dim,
            "num_classes": model.cfg.num_classes,
        },
        "config": config_to_dict(cfg),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[MNISTConvNet, dict[str, Any]]:
    """Load an MNISTConvNet (and optional optimizer state) from a saved .pt checkpoint.

    Args:
        path: Path to the .pt checkpoint file.
        device: Target device onto which the model should be loaded.
        optimizer: Optional optimizer instance to restore state into.

    Returns:
        Tuple of (loaded MNISTConvNet, full checkpoint dictionary).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at '{ckpt_path}'")

    checkpoint: dict[str, Any] = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )
    raw_model_cfg = checkpoint.get("model_config", {})
    model_cfg = ModelConfig(
        name=str(raw_model_cfg.get("name", "mnist_cnn")),
        in_channels=int(raw_model_cfg.get("in_channels", 1)),
        conv_channels=[int(c) for c in raw_model_cfg.get("conv_channels", [32, 64])],
        kernel_size=int(raw_model_cfg.get("kernel_size", 3)),
        dropout=float(raw_model_cfg.get("dropout", 0.25)),
        hidden_dim=int(raw_model_cfg.get("hidden_dim", 128)),
        num_classes=int(raw_model_cfg.get("num_classes", 10)),
    )
    model = MNISTConvNet(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, checkpoint
