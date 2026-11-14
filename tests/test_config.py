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
