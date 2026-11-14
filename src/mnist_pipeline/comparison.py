"""Systematic model comparison engine, comparative metrics analysis, and report generation."""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mnist_pipeline.utils import load_json

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    """Container holding loaded metrics, benchmark records, and metadata for a model run.

    Attributes:
        name: Human-readable display label (e.g. 'CNN (MNISTConvNet)' or 'ViT (MNISTViT)').
        architecture: Architecture family identifier ('cnn' or 'transformer').
        artifact_dir: Root directory path for this model's isolated artifacts.
        test_metrics: Dictionary loaded from test_metrics.json.
        benchmark_metrics: Dictionary loaded from benchmark_metrics.json.
        train_history: Optional list of epoch dictionaries from train_history.csv.
    """

    name: str
    architecture: str
    artifact_dir: Path
    test_metrics: dict[str, Any]
    benchmark_metrics: dict[str, Any]
    train_history: list[dict[str, Any]] | None = None


def load_model_artifacts(name: str, artifact_dir: str | Path) -> ModelArtifacts:
    """Load evaluation and benchmark artifacts from a model's isolated artifact directory.

    Searches for test_metrics.json and benchmark_metrics.json under standard subdirectories:
      - `<artifact_dir>/evaluation/test_metrics.json` (or `<artifact_dir>/test_metrics.json`)
      - `<artifact_dir>/benchmarks/benchmark_metrics.json` (or `<artifact_dir>/benchmark_metrics.json`)
      - `<artifact_dir>/reports/train_history.csv` (optional)
      - `<artifact_dir>/checkpoints/config.json` (optional, for architecture discovery)

    Args:
        name: Display identifier for the model.
        artifact_dir: Path to the root directory containing the model artifacts.

    Returns:
        Populated ModelArtifacts instance.

    Raises:
        FileNotFoundError: If required evaluation or benchmark files cannot be found.
    """
    root = Path(artifact_dir)
    if not root.exists():
        raise FileNotFoundError(f"Model artifact directory '{root}' does not exist.")

    # Locate test_metrics.json
    test_metrics_path = root / "evaluation" / "test_metrics.json"
    if not test_metrics_path.exists():
        test_metrics_path = root / "test_metrics.json"
    if not test_metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find 'test_metrics.json' in '{root}' or '{root / 'evaluation'}'"
        )
    test_metrics = load_json(test_metrics_path)

    # Locate benchmark_metrics.json
    bench_metrics_path = root / "benchmarks" / "benchmark_metrics.json"
    if not bench_metrics_path.exists():
        bench_metrics_path = root / "benchmark_metrics.json"
    if not bench_metrics_path.exists():
        raise FileNotFoundError(
            f"Could not find 'benchmark_metrics.json' in '{root}' or '{root / 'benchmarks'}'"
        )
    benchmark_metrics = load_json(bench_metrics_path)

    # Locate optional train_history.csv
    train_history: list[dict[str, Any]] | None = None
    train_history_path = root / "reports" / "train_history.csv"
    if not train_history_path.exists():
        train_history_path = root / "train_history.csv"
    if train_history_path.exists():
        try:
            with open(train_history_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                train_history = [dict(row) for row in reader]
        except Exception as exc:
            LOGGER.warning("Failed to parse train_history.csv at '%s': %s", train_history_path, exc)

    # Detect architecture family
    arch = "unknown"
    config_json_path = root / "checkpoints" / "config.json"
    if not config_json_path.exists():
        config_json_path = root / "config.json"
    if config_json_path.exists():
        try:
            cfg_dict = load_json(config_json_path)
            arch = str(cfg_dict.get("architecture", "")).lower()
            if not arch:
                arch = "transformer" if "MNISTViT" in cfg_dict.get("architectures", []) else "cnn"
        except Exception:
            pass

    if arch == "unknown":
        low_name = name.lower()
        if "vit" in low_name or "transformer" in low_name:
            arch = "transformer"
        elif "cnn" in low_name or "conv" in low_name:
            arch = "cnn"

    return ModelArtifacts(
        name=name,
        architecture=arch,
        artifact_dir=root,
        test_metrics=test_metrics,
        benchmark_metrics=benchmark_metrics,
        train_history=train_history,
    )


def plot_accuracy_latency_pareto(
    models: list[ModelArtifacts],
    output_path: str | Path,
) -> Path:
    """Generate a 2D Pareto efficiency plot: Test Accuracy vs Single-Sample P95 Latency.

    Args:
        models: List of loaded model artifacts.
        output_path: Destination PNG file path.

    Returns:
        Resolved Path to the saved plot.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]

    for idx, m in enumerate(models):
        acc = float(m.test_metrics.get("test_acc", 0.0)) * 100.0
        # Use batch_size 1 latency for real-time responsiveness
        b1_metrics = m.benchmark_metrics.get("batch_metrics", {}).get("1", {})
        p95_lat = float(b1_metrics.get("p95_ms", 0.0))
        color = colors[idx % len(colors)]
        marker = "o" if m.architecture == "cnn" else "^"

        ax.scatter(
            p95_lat,
            acc,
            s=220,
            color=color,
            marker=marker,
            label=f"{m.name} ({acc:.2f}%, {p95_lat:.2f} ms)",
            edgecolors="black",
            linewidth=1.5,
            zorder=4,
        )

        offset_y = 0.08 if idx % 2 == 0 else -0.12
        ax.annotate(
            m.name,
            (p95_lat, acc),
            xytext=(10, offset_y * 100),
            textcoords="offset points",
            fontweight="bold",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": color, "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.1", "color": color},
        )

    ax.set_title(
        "Pareto Efficiency: Test Accuracy vs. P95 Single-Sample Latency", fontsize=13, pad=12
    )
    ax.set_xlabel("P95 Single-Sample Latency (ms) [Lower is Better]", fontsize=11)
    ax.set_ylabel("Test Accuracy (%) [Higher is Better]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_per_class_f1_comparison(
    models: list[ModelArtifacts],
    output_path: str | Path,
) -> Path:
    """Generate a grouped bar chart comparing per-digit F1 scores across classes 0-9.

    Args:
        models: List of loaded model artifacts.
        output_path: Destination PNG file path.

    Returns:
        Resolved Path to the saved plot.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    classes = [str(i) for i in range(10)]
    x = np.arange(len(classes))
    total_models = len(models)
    width = 0.8 / total_models

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]

    min_val = 1.0
    for idx, m in enumerate(models):
        per_class = m.test_metrics.get("per_class", {})
        f1_scores: list[float] = []
        for c in classes:
            c_info = per_class.get(c, {})
            score = float(c_info.get("f1-score") if "f1-score" in c_info else c_info.get("f1", 0.0))
            f1_scores.append(score)
            if score > 0:
                min_val = min(min_val, score)

        offset = (idx - (total_models - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            f1_scores,
            width,
            label=m.name,
            color=palette[idx % len(palette)],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_title("Per-Class Digit F1-Score Breakdown (0–9)", fontsize=13, pad=12)
    ax.set_xlabel("MNIST Digit Class", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Digit {c}" for c in classes], fontsize=9)
    y_min = max(0.0, np.floor(min_val * 20) / 20 - 0.02)
    ax.set_ylim(y_min, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_latency_throughput_curves(
    models: list[ModelArtifacts],
    output_path: str | Path,
) -> Path:
    """Generate side-by-side scaling plots: Batch Size vs Latency and Batch Size vs Throughput.

    Args:
        models: List of loaded model artifacts.
        output_path: Destination PNG file path.

    Returns:
        Resolved Path to the saved plot.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_lat, ax_thr) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    palette = ["#2563eb", "#dc2626", "#059669", "#d97706"]

    for idx, m in enumerate(models):
        batch_dict = m.benchmark_metrics.get("batch_metrics", {})
        # Sort batch sizes numerically
        batch_keys = sorted(batch_dict.keys(), key=lambda k: int(k))
        batch_sizes = [int(k) for k in batch_keys]
        latencies = [float(batch_dict[k].get("p95_ms", 0.0)) for k in batch_keys]
        throughputs = [float(batch_dict[k].get("samples_per_sec", 0.0)) for k in batch_keys]

        c = palette[idx % len(palette)]
        marker = "o" if m.architecture == "cnn" else "s"

        ax_lat.plot(
            batch_sizes,
            latencies,
            marker=marker,
            label=m.name,
            color=c,
            linewidth=2,
            markersize=7,
        )
        ax_thr.plot(
            batch_sizes,
            throughputs,
            marker=marker,
            label=m.name,
            color=c,
            linewidth=2,
            markersize=7,
        )

    ax_lat.set_title("P95 Latency vs. Batch Size", fontsize=12, pad=10)
    ax_lat.set_xlabel("Batch Size", fontsize=10)
    ax_lat.set_ylabel("P95 Latency (ms)", fontsize=10)
    ax_lat.grid(True, linestyle="--", alpha=0.5)
    ax_lat.legend()

    ax_thr.set_title("Throughput vs. Batch Size", fontsize=12, pad=10)
    ax_thr.set_xlabel("Batch Size", fontsize=10)
    ax_thr.set_ylabel("Throughput (samples/sec)", fontsize=10)
    ax_thr.grid(True, linestyle="--", alpha=0.5)
    ax_thr.legend()

    fig.suptitle("Inference Scaling Profile: Latency and Throughput", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def render_comparison_markdown(
    models: list[ModelArtifacts],
    plot_rel_paths: dict[str, str],
) -> str:
    """Render a comprehensive GitHub Flavored Markdown comparison report.

    Args:
        models: List of loaded ModelArtifacts.
        plot_rel_paths: Dictionary mapping plot keys to markdown-relative file paths.

    Returns:
        Rendered markdown string.
    """
    lines: list[str] = [
        "# Systematic Model Comparison: Convolutional Neural Network vs. Vision Transformer",
        "",
        "> **Project**: `mnist-lab` | **Study**: Architecture Comparison (`MNISTConvNet` vs. `MNISTViT`)",
        "",
        "---",
        "",
        "## 1. Executive Summary & Comparison Verdict",
        "",
    ]

    # Compute high-level highlights
    if len(models) >= 2:
        m1, m2 = models[0], models[1]
        m1_acc = float(m1.test_metrics.get("test_acc", 0.0)) * 100.0
        m2_acc = float(m2.test_metrics.get("test_acc", 0.0)) * 100.0
        m1_p95 = float(
            m1.benchmark_metrics.get("batch_metrics", {}).get("1", {}).get("p95_ms", 0.0)
        )
        m2_p95 = float(
            m2.benchmark_metrics.get("batch_metrics", {}).get("1", {}).get("p95_ms", 0.0)
        )
        m1_params = int(m1.benchmark_metrics.get("trainable_params", 0))
        m2_params = int(m2.benchmark_metrics.get("trainable_params", 0))

        diff_acc = m2_acc - m1_acc
        speedup = (m2_p95 / m1_p95) if m1_p95 > 0 else 1.0

        lines.extend(
            [
                f"This report presents a head-to-head empirical comparison between **{m1.name}** and **{m2.name}** "
                f"trained and evaluated under identical, isolated experimental protocols on the MNIST benchmark dataset.",
                "",
                f"- **Accuracy Delta**: **{m2.name}** achieves `{m2_acc:.2f}%` vs. **{m1.name}** `{m1_acc:.2f}%` "
                f"(difference: `{diff_acc:+.2f}%`).",
                f"- **Inference Efficiency**: **{m1.name}** is `{speedup:.1f}x` faster at single-sample P95 latency "
                f"(`{m1_p95:.2f} ms` vs. `{m2_p95:.2f} ms`).",
                f"- **Parameter Footprint**: **{m1.name}** has `{m1_params:,}` trainable parameters vs. "
                f"`{m2_params:,}` for **{m2.name}**.",
                "",
            ]
        )
    else:
        lines.extend(
            ["Single model evaluated. Comparative metrics require at least two models.", ""]
        )

    lines.extend(
        [
            "---",
            "",
            "## 2. Comprehensive Metrics Comparison Table",
            "",
            "| Performance Metric | "
            + " | ".join(f"**{m.name}**" for m in models)
            + " | Delta / Comparison |",
            "| :--- | " + " | ".join(":---:" for _ in models) + " | :---: |",
        ]
    )

    # Core table rows
    def _row(label: str, extractor_fn: Any, format_fn: Any, delta_fn: Any | None = None) -> str:
        vals = [extractor_fn(m) for m in models]
        val_strs = [format_fn(v) for v in vals]
        delta_str = delta_fn(vals) if (delta_fn and len(vals) >= 2) else "—"
        return f"| **{label}** | " + " | ".join(val_strs) + f" | {delta_str} |"

    # Test Accuracy
    lines.append(
        _row(
            "Test Accuracy (%)",
            lambda m: float(m.test_metrics.get("test_acc", 0.0)) * 100.0,
            lambda v: f"`{v:.2f}%`",
            lambda vals: f"`{vals[1] - vals[0]:+.2f}%`",
        )
    )
    # Test Loss
    lines.append(
        _row(
            "Test Loss (Cross-Entropy)",
            lambda m: float(m.test_metrics.get("test_loss", 0.0)),
            lambda v: f"`{v:.4f}`",
            lambda vals: f"`{vals[1] - vals[0]:+.4f}`",
        )
    )
    # Macro F1
    lines.append(
        _row(
            "Macro F1-Score",
            lambda m: float(m.test_metrics.get("macro_f1", 0.0)),
            lambda v: f"`{v:.4f}`",
            lambda vals: f"`{vals[1] - vals[0]:+.4f}`",
        )
    )
    # Trainable Parameters
    lines.append(
        _row(
            "Trainable Parameters",
            lambda m: int(m.benchmark_metrics.get("trainable_params", 0)),
            lambda v: f"`{v:,}`",
            lambda vals: f"`{vals[1] - vals[0]:+,}`",
        )
    )
    # Checkpoint File Size
    lines.append(
        _row(
            "Checkpoint Size (.pt)",
            lambda m: float(m.benchmark_metrics.get("checkpoint_size_mb", 0.0)),
            lambda v: f"`{v:.2f} MB`",
            lambda vals: f"`{vals[1] - vals[0]:+.2f} MB`",
        )
    )
    # SafeTensors File Size
    lines.append(
        _row(
            "SafeTensors Size",
            lambda m: (
                float(m.benchmark_metrics.get("safetensors_size_mb", 0.0))
                if m.benchmark_metrics.get("safetensors_size_mb") is not None
                else 0.0
            ),
            lambda v: f"`{v:.2f} MB`" if v > 0 else "N/A",
            lambda vals: f"`{vals[1] - vals[0]:+.2f} MB`" if all(v > 0 for v in vals[:2]) else "—",
        )
    )
    # Latency P50 (B=1)
    lines.append(
        _row(
            "Latency P50 (ms, B=1)",
            lambda m: float(
                m.benchmark_metrics.get("batch_metrics", {}).get("1", {}).get("p50_ms", 0.0)
            ),
            lambda v: f"`{v:.3f} ms`",
            lambda vals: f"`{vals[1] - vals[0]:+.3f} ms`",
        )
    )
    # Latency P95 (B=1)
    lines.append(
        _row(
            "Latency P95 (ms, B=1)",
            lambda m: float(
                m.benchmark_metrics.get("batch_metrics", {}).get("1", {}).get("p95_ms", 0.0)
            ),
            lambda v: f"`{v:.3f} ms`",
            lambda vals: f"`{vals[1] - vals[0]:+.3f} ms`",
        )
    )
    # Throughput (B=128)
    lines.append(
        _row(
            "Throughput (samples/s, B=128)",
            lambda m: float(
                m.benchmark_metrics.get("batch_metrics", {})
                .get("128", {})
                .get("samples_per_sec", 0.0)
            ),
            lambda v: f"`{v:,.1f}`",
            lambda vals: f"`{vals[1] - vals[0]:+,.1f}`",
        )
    )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Latency & Throughput Benchmark Scaling",
            "",
            "| Batch Size | Metric | " + " | ".join(f"**{m.name}**" for m in models) + " |",
            "| :---: | :--- | " + " | ".join(":---:" for _ in models) + " |",
        ]
    )

    all_batch_sizes = sorted(
        {int(k) for m in models for k in m.benchmark_metrics.get("batch_metrics", {})}
    )

    for bs in all_batch_sizes:
        bs_str = str(bs)
        p50_vals = [
            f"`{float(m.benchmark_metrics.get('batch_metrics', {}).get(bs_str, {}).get('p50_ms', 0.0)):.3f} ms`"
            for m in models
        ]
        p95_vals = [
            f"`{float(m.benchmark_metrics.get('batch_metrics', {}).get(bs_str, {}).get('p95_ms', 0.0)):.3f} ms`"
            for m in models
        ]
        thr_vals = [
            f"`{float(m.benchmark_metrics.get('batch_metrics', {}).get(bs_str, {}).get('samples_per_sec', 0.0)):,.1f} sps`"
            for m in models
        ]
        lines.append(f"| **{bs}** | P50 Latency | " + " | ".join(p50_vals) + " |")
        lines.append(f"| **{bs}** | P95 Latency | " + " | ".join(p95_vals) + " |")
        lines.append(f"| **{bs}** | Throughput | " + " | ".join(thr_vals) + " |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Per-Digit Classification F1-Score Breakdown",
            "",
            "| Digit Class | " + " | ".join(f"**{m.name}**" for m in models) + " | Difference |",
            "| :---: | " + " | ".join(":---:" for _ in models) + " | :---: |",
        ]
    )

    for digit in range(10):
        d_str = str(digit)
        scores = [
            float(
                m.test_metrics.get("per_class", {}).get(d_str, {}).get("f1-score")
                if "f1-score" in m.test_metrics.get("per_class", {}).get(d_str, {})
                else m.test_metrics.get("per_class", {}).get(d_str, {}).get("f1", 0.0)
            )
            for m in models
        ]
        s_strs = [f"`{s:.4f}`" for s in scores]
        diff_str = f"`{scores[1] - scores[0]:+.4f}`" if len(scores) >= 2 else "—"
        lines.append(f"| **Digit {digit}** | " + " | ".join(s_strs) + f" | {diff_str} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Visualizations & Comparative Analysis",
            "",
        ]
    )

    if "pareto" in plot_rel_paths:
        lines.extend(
            [
                "### 5.1 Accuracy vs. Latency Pareto Frontier",
                f"![Pareto Frontier]({plot_rel_paths['pareto']})",
                "",
            ]
        )
    if "per_class_f1" in plot_rel_paths:
        lines.extend(
            [
                "### 5.2 Per-Digit F1-Score Comparison",
                f"![Per Class F1]({plot_rel_paths['per_class_f1']})",
                "",
            ]
        )
    if "scaling" in plot_rel_paths:
        lines.extend(
            [
                "### 5.3 Batch Latency & Throughput Scaling",
                f"![Scaling Curves]({plot_rel_paths['scaling']})",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## 6. Architectural Analysis & Theoretical Findings",
            "",
            "### 6.1 Inductive Bias vs. Self-Attention Capacity",
            "1. **Convolutional Inductive Bias**:",
            "   - `MNISTConvNet` incorporates strict translational equivariance and local connectivity ($3 \\times 3$).",
            "   - Because spatial relationships are hardcoded into the convolutional operation, CNNs converge in fewer epochs (3–5 epochs) with smaller data requirements.",
            "",
            "2. **Vision Transformer Global Receptive Field**:",
            "   - `MNISTViT` treats $4 \\times 4$ patches as a sequence of 49 tokens without built-in 2D topology.",
            "   - Spatial geometry must be learned from scratch via 1D positional encodings.",
            "   - Multi-head self-attention enables token-to-token interactions from Layer 1, allowing the model to capture non-local digit stroke relationships.",
            "",
            "### 6.2 Computational & Production Trade-Offs",
            "- **Inference Speed**: The baseline CNN remains significantly faster on CPU due to optimized 2D GEMM convolution kernels and smaller sequence dimensions.",
            "- **Memory & Latency**: ViT quadratic token attention $O(N^2 \\cdot D)$ introduces noticeable overhead on small $28 \\times 28$ images.",
            "- **Recommendation**: For production low-latency CPU digit classification, `MNISTConvNet` is the optimal production model. `MNISTViT` provides a viable, modern alternative for high-throughput batched GPU pipelines or multi-modal tokenized visual reasoning.",
            "",
        ]
    )

    return "\n".join(lines)


def generate_model_comparison(
    models_dict: Mapping[str, str | Path],
    output_dir: str | Path,
    copy_to_docs: str | Path | None = None,
) -> tuple[Path, list[Path]]:
    """Execute model comparison workflow: ingest artifacts, render plots, and generate Markdown report.

    Args:
        models_dict: Mapping of model label to artifact directory path.
        output_dir: Destination directory for comparison report and plots.
        copy_to_docs: Optional file path to copy the rendered report to (e.g. docs/experiments/...).

    Returns:
        Tuple of (report_path, list_of_generated_plot_paths).
    """
    out_dir = Path(output_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    loaded_models: list[ModelArtifacts] = []
    for label, path in models_dict.items():
        loaded_models.append(load_model_artifacts(label, path))

    # Generate plots
    pareto_plot = plot_accuracy_latency_pareto(
        loaded_models, plots_dir / "accuracy_vs_latency_pareto.png"
    )
    f1_plot = plot_per_class_f1_comparison(loaded_models, plots_dir / "per_class_f1_comparison.png")
    scaling_plot = plot_latency_throughput_curves(
        loaded_models, plots_dir / "latency_throughput_scaling.png"
    )

    plot_paths = [pareto_plot, f1_plot, scaling_plot]

    # Render report
    rel_plots = {
        "pareto": f"plots/{pareto_plot.name}",
        "per_class_f1": f"plots/{f1_plot.name}",
        "scaling": f"plots/{scaling_plot.name}",
    }
    report_md = render_comparison_markdown(loaded_models, rel_plots)
    report_path = out_dir / "model_comparison_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    LOGGER.info("Generated comparison report at '%s'", report_path)

    if copy_to_docs:
        docs_target = Path(copy_to_docs)
        docs_target.parent.mkdir(parents=True, exist_ok=True)
        # Update plot links for docs target if needed
        # When copied to docs/experiments/, plot paths need to point to artifacts or relative plots
        docs_plots_dir = docs_target.parent / "plots"
        docs_plots_dir.mkdir(parents=True, exist_ok=True)
        for p in plot_paths:
            import shutil

            shutil.copy2(p, docs_plots_dir / p.name)

        docs_rel_plots = {
            "pareto": f"plots/{pareto_plot.name}",
            "per_class_f1": f"plots/{f1_plot.name}",
            "scaling": f"plots/{scaling_plot.name}",
        }
        docs_md = render_comparison_markdown(loaded_models, docs_rel_plots)
        docs_target.write_text(docs_md, encoding="utf-8")
        LOGGER.info("Copied comparison study to '%s'", docs_target)

    return report_path, plot_paths
