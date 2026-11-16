"""Reproducibility seeding, device selection, timing, and environment metadata utilities."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random number generators for reproducibility.

    Args:
        seed: Integer seed value applied across all RNG backends.
        deterministic: Whether to enable deterministic cuDNN and PyTorch algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the PyTorch compute device from a configuration string.

    Args:
        requested: Device specification ('auto', 'cpu', 'cuda', or 'mps').

    Returns:
        Resolved torch.device instance.
    """
    req = requested.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if req == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerator queues before/after wall-clock timing.

    Args:
        device: Active torch.device being timed.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_git_commit_sha() -> str | None:
    """Return the current Git HEAD commit SHA if running inside a Git repository.

    Returns:
        40-character commit SHA string, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment_metadata(device: torch.device, seed: int) -> dict[str, Any]:
    """Collect software versions, hardware metadata, and reproducibility settings.

    Args:
        device: Active torch.device used by the run.
        seed: Configured global random seed.

    Returns:
        Dictionary of environment and hardware metadata.
    """
    cuda_available = torch.cuda.is_available()
    gpu_name: str | None = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown_cuda_device"

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "seed": seed,
        "git_commit": get_git_commit_sha(),
    }


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary to disk as formatted JSON, creating parent directories.

    Args:
        data: JSON-serializable dictionary.
        path: Target file path.

    Returns:
        Resolved Path to the written JSON file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file from disk into a dictionary.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed dictionary content.
    """
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        content = json.load(fh)
    if not isinstance(content, dict):
        raise ValueError(f"Expected JSON object in {target}, got {type(content).__name__}")
    return content


def setup_logging(
    log_file: str | Path | None = None,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> logging.Logger:
    """Configure root logging to both console (stdout) and a dedicated log file.

    Args:
        log_file: Optional log filename or path (e.g. 'train.log' or Path('logs/train.log')).
        log_dir: Directory where logs should be stored if log_file is a relative filename.
        level: Logging level (default: logging.INFO).
        log_format: Format string for log messages.

    Returns:
        Root logger instance.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = logging.Formatter(log_format)

    # Ensure console stream handler exists
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    else:
        for h in root_logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(level)
                h.setFormatter(formatter)

    if log_file is not None:
        file_path = Path(log_file)
        if not file_path.is_absolute() and len(file_path.parts) == 1:
            file_path = Path(log_dir) / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        already_has_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(file_path.resolve())
            for h in root_logger.handlers
        )
        if not already_has_file:
            file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    return root_logger
