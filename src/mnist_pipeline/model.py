"""Convolutional and Vision Transformer neural network architectures, checkpointing, and serialization."""

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
        self.cfg = cfg or ModelConfig(name="mnist_cnn", architecture="cnn")
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
                if module.bias is not None:
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
            "architecture": "cnn",
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
            architecture="cnn",
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


class MNISTViT(nn.Module):
    """Vision Transformer (ViT) tailored for 28x28 grayscale MNIST digit classification.

    Architecture:
        - Patch Projection: Conv2d(in_channels -> embed_dim, kernel_size=patch_size, stride=patch_size)
        - Token Sequence: Prepend learnable [CLS] token -> Add 1D learnable positional embeddings -> Dropout
        - Transformer Encoder: Stack of Pre-LN TransformerEncoderLayers (norm_first=True, activation=gelu)
        - Classification Head: LayerNorm -> Extract [CLS] token -> Linear(embed_dim -> num_classes)
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        """Initialize the MNISTViT layers from a ModelConfig.

        Args:
            cfg: Optional ModelConfig specifying patch_size, embed_dim, depth, heads, etc.
        """
        super().__init__()
        self.cfg = cfg or ModelConfig(name="mnist_vit", architecture="transformer")
        self.patch_size = self.cfg.patch_size
        self.embed_dim = self.cfg.embed_dim
        self.num_classes = self.cfg.num_classes

        if 28 % self.patch_size != 0:
            raise ValueError(
                f"Image dimension 28 must be divisible by patch_size {self.patch_size}"
            )
        self.grid_size = 28 // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Patch projection via Conv2d (single-kernel 2D patch extraction)
        self.patch_embed = nn.Conv2d(
            in_channels=self.cfg.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Learnable [CLS] token and 1D positional encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.cfg.dropout)

        # Standard Pre-LN Transformer Encoder layers
        mlp_dim = int(self.embed_dim * self.cfg.mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.cfg.num_heads,
            dim_feedforward=mlp_dim,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.depth,
            enable_nested_tensor=False,
        )

        # Final LayerNorm and classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

        # Truncated normal initialization
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights with standard truncated normal and zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of normalized 28x28 grayscale images.

        Args:
            x: Input tensor of shape [B, 1, 28, 28].

        Returns:
            Unnormalized logit tensor of shape [B, num_classes].
        """
        b = x.shape[0]
        # 1. Project patches: [B, 1, 28, 28] -> [B, D, grid, grid] -> [B, num_patches, D]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)

        # 2. Prepend learnable [CLS] token: [B, num_patches + 1, D]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1)

        # 3. Add positional embeddings and apply dropout
        tokens = self.pos_drop(tokens + self.pos_embed)

        # 4. Pass through Transformer encoder blocks
        encoded = self.transformer(tokens)

        # 5. Extract [CLS] token, apply LayerNorm, and project through classification head
        cls_out = self.norm(encoded[:, 0])
        logits: torch.Tensor = self.classifier(cls_out)
        return logits

    def save_pretrained(
        self,
        save_directory: str | Path,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save model weights in safetensors format alongside config.json for Hugging Face Hub.

        Args:
            save_directory: Directory where model.safetensors and config.json will be written.
            extra_metadata: Optional metadata dictionary.

        Returns:
            Resolved Path to save_directory.
        """
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_safetensors(state_dict, str(out_dir / "model.safetensors"))

        config_payload: dict[str, Any] = {
            "architectures": ["MNISTViT"],
            "model_type": self.cfg.name,
            "architecture": "transformer",
            "in_channels": self.cfg.in_channels,
            "num_classes": self.cfg.num_classes,
            "patch_size": self.cfg.patch_size,
            "embed_dim": self.cfg.embed_dim,
            "depth": self.cfg.depth,
            "num_heads": self.cfg.num_heads,
            "mlp_ratio": self.cfg.mlp_ratio,
            "dropout": self.cfg.dropout,
            "attention_dropout": self.cfg.attention_dropout,
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
    ) -> MNISTViT:
        """Load an MNISTViT from a local directory or a Hugging Face Model Hub repository.

        Args:
            pretrained_model_name_or_path: Local path or Hugging Face Hub repo_id ('user/repo').
            revision: Git revision (branch, tag, or commit) when downloading from the Hub.
            cache_dir: Optional local cache directory for Hugging Face Hub downloads.
            map_location: Target device mapping for loaded tensors.

        Returns:
            Initialized MNISTViT in evaluation mode with loaded weights.
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
            name=str(raw_cfg.get("model_type", "mnist_vit")),
            architecture="transformer",
            in_channels=int(raw_cfg.get("in_channels", 1)),
            num_classes=int(raw_cfg.get("num_classes", 10)),
            patch_size=int(raw_cfg.get("patch_size", 4)),
            embed_dim=int(raw_cfg.get("embed_dim", 128)),
            depth=int(raw_cfg.get("depth", 4)),
            num_heads=int(raw_cfg.get("num_heads", 4)),
            mlp_ratio=float(raw_cfg.get("mlp_ratio", 2.0)),
            dropout=float(raw_cfg.get("dropout", 0.1)),
            attention_dropout=float(raw_cfg.get("attention_dropout", 0.0)),
        )
        model = cls(model_cfg)
        device_str = str(map_location)
        state_dict = load_safetensors(str(safetensors_file), device=device_str)
        model.load_state_dict(state_dict)
        model.to(map_location)
        model.eval()
        return model


def build_model(cfg: ModelConfig) -> MNISTConvNet | MNISTViT:
    """Instantiate a neural network model matching the requested architecture in ModelConfig.

    Args:
        cfg: Validated ModelConfig specification.

    Returns:
        Initialized model (MNISTConvNet or MNISTViT).
    """
    arch = getattr(cfg, "architecture", "cnn").lower()
    if arch in ("cnn", "convnet"):
        return MNISTConvNet(cfg)
    elif arch in ("transformer", "vit"):
        return MNISTViT(cfg)
    else:
        raise ValueError(
            f"Unsupported architecture '{cfg.architecture}'. Must be one of ['cnn', 'transformer']"
        )


def load_model_from_pretrained(
    pretrained_model_name_or_path: str | Path,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> MNISTConvNet | MNISTViT:
    """Load an arbitrary pretrained model (CNN or ViT) from a local directory or Hub repository.

    Args:
        pretrained_model_name_or_path: Local directory path or Hugging Face repository ID.
        revision: Git revision tag or commit hash.
        cache_dir: Optional cache directory.
        map_location: Target device mapping.

    Returns:
        Initialized evaluation-mode model instance (MNISTConvNet or MNISTViT).
    """
    local_path = Path(pretrained_model_name_or_path)
    if local_path.exists() and local_path.is_dir():
        config_file = local_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found in '{local_path}'")
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

    raw_cfg = load_json(config_file)
    arch = str(raw_cfg.get("architecture", "")).lower()
    architectures = raw_cfg.get("architectures", [])
    if arch in ("transformer", "vit") or "MNISTViT" in architectures:
        return MNISTViT.from_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            map_location=map_location,
        )
    return MNISTConvNet.from_pretrained(
        pretrained_model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
        map_location=map_location,
    )


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
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_accuracy: float,
    cfg: PipelineConfig,
) -> Path:
    """Save a training checkpoint containing model, optimizer, epoch, metric, and config state.

    Args:
        path: Target file path for the .pt checkpoint.
        model: Trained model instance (MNISTConvNet or MNISTViT).
        optimizer: Optional optimizer whose state_dict should be saved.
        epoch: Completed epoch number (1-indexed).
        best_val_accuracy: Best validation accuracy achieved up to this epoch.
        cfg: Resolved PipelineConfig for reproducibility.

    Returns:
        Resolved Path to the written checkpoint file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    m_cfg: ModelConfig = getattr(model, "cfg", cfg.model)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_val_accuracy": best_val_accuracy,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_config": {
            "name": m_cfg.name,
            "architecture": getattr(m_cfg, "architecture", "cnn"),
            "in_channels": m_cfg.in_channels,
            "num_classes": m_cfg.num_classes,
            "conv_channels": list(getattr(m_cfg, "conv_channels", [32, 64])),
            "kernel_size": getattr(m_cfg, "kernel_size", 3),
            "dropout": m_cfg.dropout,
            "hidden_dim": getattr(m_cfg, "hidden_dim", 128),
            "patch_size": getattr(m_cfg, "patch_size", 4),
            "embed_dim": getattr(m_cfg, "embed_dim", 128),
            "depth": getattr(m_cfg, "depth", 4),
            "num_heads": getattr(m_cfg, "num_heads", 4),
            "mlp_ratio": getattr(m_cfg, "mlp_ratio", 2.0),
            "attention_dropout": getattr(m_cfg, "attention_dropout", 0.0),
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
) -> tuple[MNISTConvNet | MNISTViT, dict[str, Any]]:
    """Load a trained model (and optional optimizer state) from a saved .pt checkpoint.

    Args:
        path: Path to the .pt checkpoint file.
        device: Target device onto which the model should be loaded.
        optimizer: Optional optimizer instance to restore state into.

    Returns:
        Tuple of (loaded model nn.Module, full checkpoint dictionary).

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
    arch = str(raw_model_cfg.get("architecture", "cnn")).lower()
    if arch not in ("cnn", "convnet", "transformer", "vit"):
        # Infer from presence of transformer-specific keys
        arch = "transformer" if "patch_size" in raw_model_cfg else "cnn"

    model_cfg = ModelConfig(
        name=str(
            raw_model_cfg.get(
                "name", "mnist_vit" if arch in ("transformer", "vit") else "mnist_cnn"
            )
        ),
        architecture=arch,
        in_channels=int(raw_model_cfg.get("in_channels", 1)),
        num_classes=int(raw_model_cfg.get("num_classes", 10)),
        conv_channels=[int(c) for c in raw_model_cfg.get("conv_channels", [32, 64])],
        kernel_size=int(raw_model_cfg.get("kernel_size", 3)),
        dropout=float(raw_model_cfg.get("dropout", 0.25 if arch in ("cnn", "convnet") else 0.1)),
        hidden_dim=int(raw_model_cfg.get("hidden_dim", 128)),
        patch_size=int(raw_model_cfg.get("patch_size", 4)),
        embed_dim=int(raw_model_cfg.get("embed_dim", 128)),
        depth=int(raw_model_cfg.get("depth", 4)),
        num_heads=int(raw_model_cfg.get("num_heads", 4)),
        mlp_ratio=float(raw_model_cfg.get("mlp_ratio", 2.0)),
        attention_dropout=float(raw_model_cfg.get("attention_dropout", 0.0)),
    )
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, checkpoint
