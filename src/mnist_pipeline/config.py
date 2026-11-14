"""Structured configuration schemas, Hydra composition helpers, and validation utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class _ParamsCacheEntry:
    mtime_ns: int
    size: int
    data: dict[str, Any]


_PARAMS_CACHE: dict[Path, _ParamsCacheEntry] = {}


def get_params_file_path(params_path: str | Path | None = None) -> Path | None:
    """Locate the DVC params.yaml file across custom paths, environment variables, or repo roots.

    Args:
        params_path: Optional explicit path to the parameters file.

    Returns:
        Resolved Path to the parameters file if found, otherwise None.
    """
    if params_path is not None:
        p = Path(params_path)
        return p if p.is_file() else None

    env_path = os.environ.get("MNIST_PARAMS_PATH") or os.environ.get("DVC_PARAMS_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    cwd_path = Path.cwd() / "params.yaml"
    if cwd_path.is_file():
        return cwd_path

    repo_root_path = Path(__file__).resolve().parents[2] / "params.yaml"
    if repo_root_path.is_file():
        return repo_root_path

    return None


def load_params_dict(
    params_path: str | Path | None = None,
    require_file: bool | None = None,
) -> dict[str, Any]:
    """Load parameter dictionary from params.yaml with nanosecond mtime and size caching.

    Args:
        params_path: Optional explicit path to the parameters file.
        require_file: Whether the parameters file is strictly required. If None, checks the
            `MNIST_REQUIRE_PARAMS` environment variable (default: False).

    Returns:
        Nested dictionary of parameters, or empty dict if file is not found and not required.

    Raises:
        FileNotFoundError: If the parameter file is required but cannot be found.
        ValueError: If the file contains invalid YAML syntax or does not parse into a mapping.
        OSError: If an operating system error occurs while reading the file.
    """
    if require_file is None:
        require_file = os.environ.get("MNIST_REQUIRE_PARAMS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    target = get_params_file_path(params_path)
    if target is None or not target.is_file():
        if require_file:
            target_desc = str(params_path) if params_path is not None else "params.yaml"
            raise FileNotFoundError(
                f"Canonical parameter file '{target_desc}' was not found. "
                "Set MNIST_REQUIRE_PARAMS=0 to allow fallback to schema defaults."
            )
        return {}

    stat_res = target.stat()
    mtime_ns = stat_res.st_mtime_ns
    size = stat_res.st_size

    cached = _PARAMS_CACHE.get(target)
    if cached is not None and cached.mtime_ns == mtime_ns and cached.size == size:
        return cached.data

    try:
        with open(target, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse parameter YAML file at '{target}': {exc}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read parameter file at '{target}': {exc}") from exc

    if data is None:
        data = {}
    elif not isinstance(data, dict):
        raise ValueError(
            f"Parameter file at '{target}' must be a mapping/dictionary, got {type(data).__name__}."
        )

    _PARAMS_CACHE[target] = _ParamsCacheEntry(mtime_ns=mtime_ns, size=size, data=data)
    return data


def get_dvc_param(key: str, default: Any = None, params_path: str | Path | None = None) -> Any:
    """Retrieve a dot-separated parameter value from params.yaml with fallback to default.

    Args:
        key: Dot-separated parameter path (e.g., 'train.lr', 'data.batch_size').
        default: Fallback default value if parameter is not present.
        params_path: Optional explicit path to params.yaml.

    Returns:
        Value from params.yaml if present, otherwise default.

    Raises:
        ValueError: If params.yaml contains invalid YAML syntax or is not a mapping.
        FileNotFoundError: If params.yaml is missing and strictly required.
    """
    params = load_params_dict(params_path)
    cur: Any = params
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


# Register custom resolvers for OmegaConf and Hydra
try:
    OmegaConf.register_new_resolver("param", get_dvc_param, replace=True)
    OmegaConf.register_new_resolver("dvc_param", get_dvc_param, replace=True)
except Exception:
    pass


@dataclass
class DataConfig:
    """Configuration for MNIST dataset downloading, splitting, and loading.

    Attributes:
        raw_dir: Directory where raw torchvision MNIST files are stored.
        processed_dir: Directory where deterministic processed tensors are saved.
        download: Whether to download MNIST if not already present locally.
        val_fraction: Fraction of the 60,000 training examples reserved for validation.
        mean: Normalization mean for MNIST grayscale images.
        std: Normalization standard deviation for MNIST grayscale images.
        batch_size: Mini-batch size for training and validation DataLoaders.
        num_workers: Number of subprocesses for DataLoader workers.
        pin_memory: Whether to pin host memory for faster CUDA transfers.
        seed: Random seed for stratified train/validation splitting.
        allow_synthetic_fallback: Whether to generate synthetic data if download fails.
        num_synthetic_train: Number of synthetic training samples when fallback is enabled.
        num_synthetic_test: Number of synthetic test samples when fallback is enabled.
    """

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    download: bool = True
    val_fraction: float = 0.1
    mean: float = 0.1307
    std: float = 0.3081
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    allow_synthetic_fallback: bool = False
    num_synthetic_train: int = 600
    num_synthetic_test: int = 100


@dataclass
class ModelConfig:
    """Configuration for MNIST neural network architectures (CNN and Vision Transformer).

    Attributes:
        name: Architecture identifier string.
        architecture: Model architecture family ('cnn' or 'transformer' / 'vit').
        in_channels: Number of input image channels (1 for grayscale MNIST).
        conv_channels: Output channels for each convolutional block (CNN).
        kernel_size: Convolutional kernel size (CNN).
        dropout: Dropout probability before classification head (CNN) or encoder dropout (ViT).
        hidden_dim: Dimension of intermediate fully connected layer (CNN).
        num_classes: Number of target digit classes (10 for digits 0-9).
        patch_size: Square patch side dimension in pixels (ViT).
        embed_dim: Latent token embedding dimension (ViT).
        depth: Number of Transformer encoder layers (ViT).
        num_heads: Number of multi-head self-attention heads (ViT).
        mlp_ratio: Ratio of feedforward MLP hidden dimension to embed_dim (ViT).
        attention_dropout: Dropout probability applied to attention weights (ViT).
    """

    name: str = "mnist_cnn"
    architecture: str = "cnn"
    in_channels: int = 1
    conv_channels: list[int] = field(default_factory=lambda: [32, 64])
    kernel_size: int = 3
    dropout: float = 0.25
    hidden_dim: int = 128
    num_classes: int = 10
    patch_size: int = 4
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    attention_dropout: float = 0.0


@dataclass
class TrainConfig:
    """Configuration for model training, optimization, and checkpointing.

    Attributes:
        epochs: Number of full passes over the training dataset.
        lr: Optimizer learning rate.
        weight_decay: L2 weight decay coefficient.
        optimizer: Optimizer type ('adam', 'adamw', or 'sgd').
        seed: Global random seed for deterministic training.
        deterministic: Whether to enforce deterministic PyTorch algorithms.
        device: Target compute device ('auto', 'cpu', 'cuda', or 'mps').
        resume_from: Optional path to a checkpoint file for resuming training.
        early_stopping_patience: Optional number of epochs before early stopping.
        max_train_batches: Optional cap on training batches per epoch (for smoke tests).
        max_val_batches: Optional cap on validation batches per epoch (for smoke tests).
    """

    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    resume_from: str | None = None
    early_stopping_patience: int | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None


@dataclass
class EvaluateConfig:
    """Configuration for held-out test evaluation, benchmarking, and MVE gates.

    Attributes:
        batch_size: Mini-batch size for test-set evaluation.
        checkpoint_path: Path to the selected model checkpoint to evaluate.
        benchmark_warmup_runs: Number of untimed warmup forward passes.
        benchmark_timed_runs: Number of timed forward passes per batch size.
        benchmark_batch_sizes: List of batch sizes to benchmark.
        mve_min_val_accuracy: Minimum acceptable validation accuracy for MVE gate.
        mve_min_test_accuracy: Minimum acceptable test accuracy for MVE gate.
        mve_min_macro_f1: Minimum acceptable macro-averaged F1 score for MVE gate.
        mve_max_p95_latency_ms: Maximum acceptable P95 single-sample latency in ms.
        mve_min_throughput_sps: Minimum acceptable batched throughput in samples/sec.
    """

    batch_size: int = 256
    checkpoint_path: str = "artifacts/checkpoints/checkpoint_best.pt"
    benchmark_warmup_runs: int = 10
    benchmark_timed_runs: int = 50
    benchmark_batch_sizes: list[int] = field(default_factory=lambda: [1, 32, 128])
    mve_min_val_accuracy: float = 0.985
    mve_min_test_accuracy: float = 0.985
    mve_min_macro_f1: float = 0.985
    mve_max_p95_latency_ms: float = 25.0
    mve_min_throughput_sps: float = 1000.0


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases experiment tracking.

    Attributes:
        enabled: Whether W&B logging is enabled.
        project: W&B project name.
        entity: Optional W&B user or team entity.
        mode: W&B execution mode ('online', 'offline', or 'disabled').
        run_name: Optional custom display name for the run.
        tags: List of descriptive tags attached to the W&B run.
        log_model: Whether to upload model checkpoints as W&B artifacts.
    """

    enabled: bool = True
    project: str = "mnist-classification"
    entity: str | None = None
    mode: str = "disabled"
    run_name: str | None = None
    tags: list[str] = field(default_factory=lambda: ["mnist", "cnn", "baseline"])
    log_model: bool = False


@dataclass
class HFConfig:
    """Configuration for Hugging Face Hub model and dataset integration.

    Attributes:
        repo_id: Hugging Face Model Hub repository ID ('username/repo').
        dataset_repo_id: Hugging Face Dataset Hub repository ID ('username/dataset').
        revision: Git branch, tag, or commit hash on the Hub.
        private: Whether created Hub repositories should be private.
        cache_dir: Local directory for caching downloaded Hub artifacts.
        dry_run: Whether to stage upload payloads locally without network calls.
        image_path: Optional path to an image file for CLI inference.
        sample_index: Index of the test sample to run inference on if no image_path is given.
    """

    repo_id: str = "chrisjcc/mnist-cnn"
    dataset_repo_id: str = "chrisjcc/mnist-dataset"
    revision: str = "main"
    private: bool = False
    cache_dir: str = "artifacts/hf_cache"
    dry_run: bool = False
    image_path: str | None = None
    sample_index: int = 0


@dataclass
class PathsConfig:
    """File system paths for pipeline artifacts, reports, templates, and documentation.

    Attributes:
        artifacts_dir: Root directory for generated pipeline artifacts.
        checkpoints_dir: Directory storing model checkpoints and safetensors weights.
        evaluation_dir: Directory storing held-out test metrics and confusion matrices.
        benchmarks_dir: Directory storing latency and throughput benchmark outputs.
        reports_dir: Directory storing training history and Markdown evaluation reports.
        plots_dir: Directory storing diagnostic PNG plots.
        templates_dir: Directory containing Markdown templates for HF cards.
        docs_dir: Directory storing user documentation and rendered cards.
    """

    artifacts_dir: str = "artifacts"
    checkpoints_dir: str = "artifacts/checkpoints"
    evaluation_dir: str = "artifacts/evaluation"
    benchmarks_dir: str = "artifacts/benchmarks"
    reports_dir: str = "artifacts/reports"
    plots_dir: str = "artifacts/reports/plots"
    templates_dir: str = "templates"
    docs_dir: str = "docs"


@dataclass
class PipelineConfig:
    """Top-level configuration object combining all pipeline sub-configurations.

    Attributes:
        data: Dataset and preprocessing settings.
        model: Neural network architecture settings.
        train: Training loop and optimizer settings.
        evaluate: Evaluation, benchmarking, and MVE threshold settings.
        wandb: Weights & Biases tracking settings.
        hf: Hugging Face Hub upload and inference settings.
        paths: Artifact and documentation directory paths.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hf: HFConfig = field(default_factory=HFConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def validate_config(cfg: PipelineConfig | DictConfig) -> None:
    """Validate configuration values and raise ValueError on invalid settings.

    Args:
        cfg: A PipelineConfig instance or Hydra DictConfig to validate.

    Raises:
        ValueError: If any configuration field violates domain constraints.
    """
    if cfg.data.batch_size <= 0:
        raise ValueError(f"data.batch_size must be positive, got {cfg.data.batch_size}")
    if not (0.0 < cfg.data.val_fraction < 1.0):
        raise ValueError(f"data.val_fraction must be in (0, 1), got {cfg.data.val_fraction}")
    if cfg.data.std <= 0.0:
        raise ValueError(f"data.std must be positive, got {cfg.data.std}")
    if cfg.data.num_workers < 0:
        raise ValueError(f"data.num_workers must be >= 0, got {cfg.data.num_workers}")

    valid_architectures = {"cnn", "convnet", "transformer", "vit"}
    arch = cfg.model.architecture.lower()
    if arch not in valid_architectures:
        raise ValueError(
            f"model.architecture must be one of {sorted(valid_architectures)}, got '{cfg.model.architecture}'"
        )
    if cfg.model.in_channels <= 0:
        raise ValueError(f"model.in_channels must be positive, got {cfg.model.in_channels}")
    if cfg.model.num_classes < 2:
        raise ValueError(f"model.num_classes must be >= 2, got {cfg.model.num_classes}")

    if arch in {"cnn", "convnet"}:
        if len(cfg.model.conv_channels) < 1 or any(c <= 0 for c in cfg.model.conv_channels):
            raise ValueError(
                f"model.conv_channels must contain positive integers, got {cfg.model.conv_channels}"
            )
        if cfg.model.kernel_size <= 0:
            raise ValueError(f"model.kernel_size must be positive, got {cfg.model.kernel_size}")
        if not (0.0 <= cfg.model.dropout < 1.0):
            raise ValueError(f"model.dropout must be in [0, 1), got {cfg.model.dropout}")
        if cfg.model.hidden_dim <= 0:
            raise ValueError(f"model.hidden_dim must be positive, got {cfg.model.hidden_dim}")
    else:  # transformer / vit
        if cfg.model.patch_size <= 0 or 28 % cfg.model.patch_size != 0:
            raise ValueError(f"model.patch_size must divide 28 evenly, got {cfg.model.patch_size}")
        if cfg.model.num_heads <= 0:
            raise ValueError(f"model.num_heads must be positive, got {cfg.model.num_heads}")
        if cfg.model.embed_dim <= 0 or cfg.model.embed_dim % cfg.model.num_heads != 0:
            raise ValueError(
                f"model.embed_dim ({cfg.model.embed_dim}) must be divisible by num_heads ({cfg.model.num_heads})"
            )
        if cfg.model.depth <= 0:
            raise ValueError(f"model.depth must be positive, got {cfg.model.depth}")
        if cfg.model.mlp_ratio <= 0:
            raise ValueError(f"model.mlp_ratio must be positive, got {cfg.model.mlp_ratio}")
        if not (0.0 <= cfg.model.dropout < 1.0):
            raise ValueError(f"model.dropout must be in [0, 1), got {cfg.model.dropout}")
        if not (0.0 <= cfg.model.attention_dropout < 1.0):
            raise ValueError(
                f"model.attention_dropout must be in [0, 1), got {cfg.model.attention_dropout}"
            )

    if cfg.train.epochs <= 0:
        raise ValueError(f"train.epochs must be positive, got {cfg.train.epochs}")
    if cfg.train.lr <= 0.0:
        raise ValueError(f"train.lr must be positive, got {cfg.train.lr}")
    if cfg.train.weight_decay < 0.0:
        raise ValueError(f"train.weight_decay must be >= 0, got {cfg.train.weight_decay}")
    valid_optimizers = {"adam", "adamw", "sgd"}
    if cfg.train.optimizer.lower() not in valid_optimizers:
        raise ValueError(
            f"train.optimizer must be one of {sorted(valid_optimizers)}, got '{cfg.train.optimizer}'"
        )
    valid_devices = {"auto", "cpu", "cuda", "mps"}
    if cfg.train.device.lower() not in valid_devices:
        raise ValueError(
            f"train.device must be one of {sorted(valid_devices)}, got '{cfg.train.device}'"
        )

    if cfg.evaluate.batch_size <= 0:
        raise ValueError(f"evaluate.batch_size must be positive, got {cfg.evaluate.batch_size}")
    if cfg.evaluate.benchmark_warmup_runs < 0:
        raise ValueError(
            f"evaluate.benchmark_warmup_runs must be >= 0, got {cfg.evaluate.benchmark_warmup_runs}"
        )
    if cfg.evaluate.benchmark_timed_runs <= 0:
        raise ValueError(
            f"evaluate.benchmark_timed_runs must be > 0, got {cfg.evaluate.benchmark_timed_runs}"
        )
    if not cfg.evaluate.benchmark_batch_sizes or any(
        b <= 0 for b in cfg.evaluate.benchmark_batch_sizes
    ):
        raise ValueError("evaluate.benchmark_batch_sizes must contain positive batch sizes")

    valid_wandb_modes = {"online", "offline", "disabled"}
    if cfg.wandb.mode.lower() not in valid_wandb_modes:
        raise ValueError(
            f"wandb.mode must be one of {sorted(valid_wandb_modes)}, got '{cfg.wandb.mode}'"
        )


def to_pipeline_config(
    cfg: DictConfig | dict[str, Any] | PipelineConfig,
    params_path: str | Path | None = None,
    require_params: bool | None = None,
) -> PipelineConfig:
    """Convert a Hydra DictConfig or raw dictionary into a validated PipelineConfig.

    Integrates params.yaml as the Single Source of Truth: schema defaults are populated,
    params.yaml values are layered, and explicit configuration dictionary overrides
    take highest precedence.

    Args:
        cfg: Input configuration as DictConfig, dict, or PipelineConfig.
        params_path: Optional explicit path to params.yaml.
        require_params: Whether params.yaml is strictly required to exist.

    Returns:
        Validated PipelineConfig dataclass instance.
    """
    if isinstance(cfg, PipelineConfig):
        validate_config(cfg)
        return cfg

    raw_dict: dict[str, Any]
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            raise ValueError("Resolved OmegaConf container must be a mapping.")
        raw_dict = {str(k): v for k, v in container.items()}
    else:
        raw_dict = dict(cfg)

    raw_dict.pop("hydra", None)

    structured_schema = OmegaConf.structured(PipelineConfig)
    params_dict = load_params_dict(params_path, require_file=require_params)
    if params_dict:
        params_omega = OmegaConf.create(params_dict)
        merged = OmegaConf.merge(structured_schema, params_omega, raw_dict)
    else:
        merged = OmegaConf.merge(structured_schema, raw_dict)

    obj = OmegaConf.to_object(merged)
    if not isinstance(obj, PipelineConfig):
        raise ValueError("Failed to materialize PipelineConfig from configuration.")
    validate_config(obj)
    return obj


def load_hydra_config(
    config_dir: str | Path = "configs",
    config_name: str = "config",
    overrides: list[str] | None = None,
    params_path: str | Path | None = None,
    require_params: bool | None = None,
) -> PipelineConfig:
    """Load and validate a PipelineConfig using Hydra's Compose API.

    Args:
        config_dir: Path to the directory containing Hydra YAML files.
        config_name: Name of the root configuration file (without .yaml).
        overrides: Optional list of Hydra CLI-style override strings.
        params_path: Optional explicit path to params.yaml.
        require_params: Whether params.yaml is strictly required to exist.

    Returns:
        Validated PipelineConfig instance.
    """
    abs_config_dir = str(Path(config_dir).resolve())
    with initialize_config_dir(version_base="1.3", config_dir=abs_config_dir):
        hydra_cfg = compose(config_name=config_name, overrides=overrides or [])
    return to_pipeline_config(hydra_cfg, params_path=params_path, require_params=require_params)


def config_to_dict(cfg: PipelineConfig | DictConfig) -> dict[str, Any]:
    """Serialize a PipelineConfig or DictConfig into a plain JSON-serializable dictionary.

    Args:
        cfg: Configuration object to convert.

    Returns:
        Nested dictionary representation of the configuration.
    """
    if isinstance(cfg, DictConfig):
        res = OmegaConf.to_container(cfg, resolve=True)
    else:
        res = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
    if not isinstance(res, dict):
        raise ValueError("Failed to convert config to dictionary.")
    return {str(k): v for k, v in res.items()}
