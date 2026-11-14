"""Unit tests for Hydra configuration composition, schema validation, and utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mnist_pipeline.config import (
    PipelineConfig,
    config_to_dict,
    get_dvc_param,
    load_hydra_config,
    load_params_dict,
    to_pipeline_config,
    validate_config,
)
from mnist_pipeline.utils import (
    collect_environment_metadata,
    load_json,
    resolve_device,
    save_json,
    set_seed,
)


def test_default_hydra_config_loads() -> None:
    cfg = load_hydra_config(config_dir="configs", config_name="config")
    assert isinstance(cfg, PipelineConfig)
    assert cfg.data.batch_size == 128
    assert cfg.data.val_fraction == pytest.approx(0.1)
    assert cfg.model.num_classes == 10
    assert cfg.model.conv_channels == [32, 64]
    assert cfg.train.optimizer == "adam"


def test_hydra_config_overrides() -> None:
    cfg = load_hydra_config(
        config_dir="configs",
        config_name="config",
        overrides=["train.epochs=3", "train.lr=0.0005", "data.batch_size=64"],
    )
    assert cfg.train.epochs == 3
    assert cfg.train.lr == pytest.approx(0.0005)
    assert cfg.data.batch_size == 64


@pytest.mark.parametrize(
    "override",
    [
        "data.batch_size=0",
        "data.val_fraction=1.5",
        "data.std=-0.1",
        "model.dropout=1.0",
        "model.num_classes=1",
        "train.epochs=0",
        "train.lr=-0.01",
        "train.optimizer=invalid_opt",
        "train.device=tpu",
        "evaluate.batch_size=-1",
        "wandb.mode=invalid_mode",
    ],
)
def test_invalid_config_raises_value_error(override: str) -> None:
    with pytest.raises(ValueError):
        load_hydra_config(
            config_dir="configs",
            config_name="config",
            overrides=[override],
        )


def test_to_pipeline_config_and_serialization() -> None:
    cfg = PipelineConfig()
    validate_config(cfg)
    as_dict = config_to_dict(cfg)
    round_trip = to_pipeline_config(as_dict)
    assert round_trip.model.hidden_dim == cfg.model.hidden_dim
    assert round_trip.train.seed == cfg.train.seed


def test_seed_and_device_utilities(tmp_path: Path) -> None:
    set_seed(123, deterministic=True)
    t1 = torch.randn(4, 4)
    set_seed(123, deterministic=True)
    t2 = torch.randn(4, 4)
    assert torch.allclose(t1, t2)

    device = resolve_device("cpu")
    assert device.type == "cpu"

    meta = collect_environment_metadata(device=device, seed=123)
    assert meta["seed"] == 123
    assert meta["device"] == "cpu"

    json_path = save_json(meta, tmp_path / "env.json")
    loaded = load_json(json_path)
    assert loaded["seed"] == 123


def test_params_yaml_ssot_resolver_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_params = tmp_path / "custom_params.yaml"
    custom_params.write_text(
        "train:\n  lr: 0.0003\n  epochs: 8\ndata:\n  batch_size: 64\nmodel:\n  conv_channels: [16, 32]\n"
    )
    monkeypatch.setenv("MNIST_PARAMS_PATH", str(custom_params))

    cfg = load_hydra_config(config_dir="configs", config_name="config")
    assert cfg.train.lr == pytest.approx(0.0003)
    assert cfg.train.epochs == 8
    assert cfg.data.batch_size == 64
    assert cfg.model.conv_channels == [16, 32]

    # CLI overrides must still take precedence over params.yaml
    cfg_override = load_hydra_config(
        config_dir="configs",
        config_name="config",
        overrides=["train.lr=0.01", "train.epochs=12"],
    )
    assert cfg_override.train.lr == pytest.approx(0.01)
    assert cfg_override.train.epochs == 12
    # Non-overridden params must still come from custom_params.yaml
    assert cfg_override.data.batch_size == 64


def test_to_pipeline_config_merges_params_yaml(tmp_path: Path) -> None:
    custom_params = tmp_path / "params.yaml"
    custom_params.write_text("train:\n  lr: 0.0007\n  weight_decay: 0.0001\n")

    # When given a partial dictionary, params.yaml provides defaults
    cfg = to_pipeline_config({"train": {"epochs": 7}}, params_path=custom_params)
    assert cfg.train.epochs == 7
    assert cfg.train.lr == pytest.approx(0.0007)
    assert cfg.train.weight_decay == pytest.approx(0.0001)


def test_get_dvc_param_fallback_and_missing(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.yaml"
    val = get_dvc_param("train.non_existent", default=42, params_path=non_existent)
    assert val == 42


def test_invalid_yaml_syntax_raises_informative_error(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "corrupted_params.yaml"
    bad_yaml.write_text("train:\n  lr: [unclosed list\n")

    with pytest.raises(ValueError, match="Failed to parse parameter YAML file"):
        load_params_dict(bad_yaml)


def test_non_mapping_yaml_raises_error(tmp_path: Path) -> None:
    scalar_yaml = tmp_path / "scalar_params.yaml"
    scalar_yaml.write_text("- item1\n- item2\n")

    with pytest.raises(ValueError, match="must be a mapping/dictionary, got list"):
        load_params_dict(scalar_yaml)


def test_missing_params_file_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_file = tmp_path / "does_not_exist.yaml"

    # Default policy allows fallback
    assert load_params_dict(missing_file, require_file=False) == {}

    # Strict policy raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Canonical parameter file .* was not found"):
        load_params_dict(missing_file, require_file=True)

    # Environment variable policy
    monkeypatch.setenv("MNIST_REQUIRE_PARAMS", "1")
    with pytest.raises(FileNotFoundError):
        load_params_dict(missing_file)


def test_cache_invalidation_on_content_change(tmp_path: Path) -> None:
    params_file = tmp_path / "dynamic_params.yaml"
    params_file.write_text("train:\n  lr: 0.001\n")

    d1 = load_params_dict(params_file)
    assert d1["train"]["lr"] == 0.001

    # Modify file content (changes size and/or mtime_ns)
    params_file.write_text("train:\n  lr: 0.0005\n")
    d2 = load_params_dict(params_file)
    assert d2["train"]["lr"] == 0.0005


def test_transformer_hydra_config_loads() -> None:
    cfg = load_hydra_config(
        config_dir="configs",
        config_name="config",
        overrides=["model=transformer"],
    )
    assert cfg.model.name == "mnist_vit"
    assert cfg.model.architecture == "transformer"
    assert cfg.model.patch_size == 4
    assert cfg.model.embed_dim == 128
    assert cfg.model.depth == 4
    assert cfg.model.num_heads == 4
    assert cfg.model.mlp_ratio == 2.0


def test_artifact_path_isolation_interpolation() -> None:
    custom_art = "artifacts/models/mnist_vit"
    cfg = load_hydra_config(
        config_dir="configs",
        config_name="config",
        overrides=[f"paths.artifacts_dir={custom_art}"],
    )
    assert cfg.paths.artifacts_dir == custom_art
    assert cfg.paths.checkpoints_dir == f"{custom_art}/checkpoints"
    assert cfg.paths.evaluation_dir == f"{custom_art}/evaluation"
    assert cfg.paths.benchmarks_dir == f"{custom_art}/benchmarks"
    assert cfg.paths.reports_dir == f"{custom_art}/reports"
    assert cfg.paths.plots_dir == f"{custom_art}/reports/plots"
    assert cfg.evaluate.checkpoint_path == f"{custom_art}/checkpoints/checkpoint_best.pt"


@pytest.mark.parametrize(
    "override",
    [
        "model=transformer model.patch_size=5",
        "model=transformer model.embed_dim=127",
        "model=transformer model.depth=0",
        "model=transformer model.num_heads=0",
        "model=transformer model.mlp_ratio=0",
        "model.architecture=unsupported_arch",
    ],
)
def test_invalid_transformer_config_raises(override: str) -> None:
    with pytest.raises(ValueError):
        load_hydra_config(
            config_dir="configs",
            config_name="config",
            overrides=override.split(),
        )
