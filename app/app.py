"""Interactive Streamlit application for MNIST handwritten digit classification.

Supports three input modes:
  1. Interactive freehand drawing canvas (`streamlit-drawable-canvas`)
  2. Handwritten digit image upload (`PNG`, `JPG`, `JPEG`) with polarity controls
  3. Canonical MNIST/synthetic test sample gallery (`0`–`9`)

Loads the trained model (`MNISTConvNet` or `MNISTViT`) from Hugging Face Model Hub
with automatic fallback to local checkpoints (`artifacts/models/mnist_*`).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from mnist_pipeline.data import generate_synthetic_digit_template, load_processed_split
from mnist_pipeline.inference import (
    PredictionResult,
    load_inference_model,
    predict_pil_image,
    predict_tensor,
)
from mnist_pipeline.model import count_trainable_parameters

MODEL_PRESETS: dict[str, dict[str, str]] = {
    "CNN (Convolutional)": {
        "repo_id": "chrisjcc/mnist-cnn",
        "checkpoint_fallback": str(
            Path("artifacts") / "models" / "mnist_cnn" / "checkpoints" / "checkpoint_best.pt"
        ),
        "description": "2-block ConvNet + BatchNorm + Dropout (~421k params)",
        "badge": "Convolutional Neural Network (CNN)",
    },
    "ViT (Vision Transformer)": {
        "repo_id": "chrisjcc/mnist-vit",
        "checkpoint_fallback": str(
            Path("artifacts") / "models" / "mnist_vit" / "checkpoints" / "checkpoint_best.pt"
        ),
        "description": "Patch-based Vision Transformer + Multi-Head Attention (~318k params)",
        "badge": "Vision Transformer (ViT)",
    },
    "Custom Model / Repo": {
        "repo_id": "",
        "checkpoint_fallback": "",
        "description": "Custom Hugging Face repository ID or local checkpoint path",
        "badge": "Custom Architecture",
    },
}


@st.cache_resource(show_spinner="Loading model...")
def get_cached_model(repo_id: str, checkpoint_path: str) -> tuple[torch.nn.Module, str]:
    """Load and cache the inference model across Streamlit sessions.

    Args:
        repo_id: Hugging Face Model Hub repository identifier.
        checkpoint_path: Local fallback checkpoint path.

    Returns:
        Tuple of `(model_in_eval_mode, source_description)`.
    """
    return load_inference_model(
        repo_id_or_path=repo_id,
        checkpoint_fallback=checkpoint_path,
        device="cpu",
    )


def _render_prediction_panel(result: PredictionResult, model_name: str | None = None) -> None:
    """Render the `28x28` preprocessed image, top prediction metric, and probability chart.

    Args:
        result: Populated `PredictionResult` from inference.
        model_name: Optional class name of the active model for UI display.
    """
    col_preview, col_metrics, col_chart = st.columns([1.1, 1.1, 2.2])

    with col_preview:
        st.subheader("28x28 Model Input")
        preview_uint8 = (np.clip(result.display_image, 0.0, 1.0) * 255.0).astype(np.uint8)
        preview_pil = Image.fromarray(preview_uint8, mode="L").resize(
            (196, 196), Image.Resampling.NEAREST
        )
        st.image(preview_pil, caption="Normalized 28x28 Tensor Preview", width=196)

    with col_metrics:
        st.subheader("Prediction")
        if model_name:
            st.caption(f"**Architecture**: `{model_name}`")
        st.metric(
            label="Predicted Digit",
            value=f"{result.predicted_digit}",
            delta=f"{result.confidence * 100:.2f}% confidence",
        )
        sorted_probs = sorted(
            result.probabilities.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        runner_up_digit, runner_up_prob = sorted_probs[1]
        st.caption(f"**Runner-up**: Digit `{runner_up_digit}` (`{runner_up_prob * 100:.2f}%`)")

    with col_chart:
        st.subheader("Class Probability Distribution (0–9)")
        chart_data = {f"Digit {k}": float(v) for k, v in result.probabilities.items()}
        st.bar_chart(chart_data, height=240)


def run_app() -> None:
    """Configure page layout and render the interactive MNIST classification app."""
    st.set_page_config(
        page_title="MNIST Digit Classifier — PyTorch + HF Hub",
        page_icon="🔢",
        layout="wide",
    )

    st.title("🔢 MNIST Handwritten Digit Classifier")
    st.markdown(
        "Interactive inference demo powered by **PyTorch (`MNISTConvNet` & `MNISTViT`)**, "
        "**Hydra**, **DVC**, and **Hugging Face Model Hub**."
    )

    env_repo = os.environ.get("HF_MODEL_REPO_ID")
    default_index = 0
    if env_repo:
        if "vit" in env_repo.lower() or "transformer" in env_repo.lower():
            default_index = 1
        elif env_repo != "chrisjcc/mnist-cnn":
            default_index = 2

    with st.sidebar:
        st.header("⚙️ Model Configuration")
        selected_model_type = st.selectbox(
            "Model Architecture",
            options=list(MODEL_PRESETS.keys()),
            index=default_index,
            help="Select the neural network architecture to run digit classification with.",
        )
        preset = MODEL_PRESETS[selected_model_type]

        if selected_model_type == "Custom Model / Repo":
            repo_id = st.text_input(
                "Hugging Face Model Repo ID",
                value=env_repo or "chrisjcc/mnist-cnn",
            )
            checkpoint_path = st.text_input(
                "Local Checkpoint Path (fallback)",
                value=os.environ.get(
                    "MNIST_CHECKPOINT_PATH",
                    str(
                        Path("artifacts")
                        / "models"
                        / "mnist_cnn"
                        / "checkpoints"
                        / "checkpoint_best.pt"
                    ),
                ),
            )
        else:
            repo_id = env_repo or preset["repo_id"]
            checkpoint_path = os.environ.get("MNIST_CHECKPOINT_PATH", preset["checkpoint_fallback"])
            st.caption(f"**Architecture**: {preset['description']}")
            st.caption(f"**Target Hub Repo**: `{repo_id}`")

        model, source_desc = get_cached_model(repo_id, checkpoint_path)
        param_count = count_trainable_parameters(model)
        st.success(f"**Loaded Source**: `{source_desc}`")
        st.info(f"**Active Model**: `{model.__class__.__name__}` ({param_count:,} params)")

        st.markdown("---")
        st.subheader("Preprocessing Controls")
        center_crop = st.checkbox(
            "MNIST Normalization (Crop → 20x20 → Center-of-Mass)",
            value=True,
            help="Applies canonical NIST/MNIST pipeline: tight bounding-box crop, aspect-preserving 20x20 Lanczos resize, peak brightness scaling, and Center-of-Mass alignment onto (13.5, 13.5).",
        )

    tab_draw, tab_upload, tab_sample = st.tabs(
        ["✏️ Draw a Digit", "📤 Upload an Image", "🧪 Test Set Gallery"]
    )

    with tab_draw:
        st.markdown("Draw a single digit (`0`–`9`) in **white ink on the black canvas** below:")
        try:
            import inspect
            from typing import Any

            from streamlit_drawable_canvas import st_canvas

            stroke_width = st.slider("Brush Stroke Width", min_value=8, max_value=28, value=14)
            canvas_kwargs: dict[str, Any] = {
                "fill_color": "rgba(255, 255, 255, 1.0)",
                "stroke_width": stroke_width,
                "stroke_color": "#FFFFFF",
                "background_color": "#000000",
                "height": 280,
                "width": 280,
                "drawing_mode": "freedraw",
                "key": "mnist_draw_canvas",
            }
            if "return_image_data" in inspect.signature(st_canvas).parameters:
                canvas_kwargs["return_image_data"] = True

            canvas_result = st_canvas(**canvas_kwargs)
            if canvas_result.image_data is not None:
                img_rgba = np.asarray(canvas_result.image_data, dtype=np.uint8)
                if int(np.max(img_rgba[:, :, :3])) > 10:
                    pil_drawn = Image.fromarray(img_rgba, mode="RGBA")
                    pred_res = predict_pil_image(
                        model=model,
                        image=pil_drawn,
                        invert=False,
                        auto_invert=False,
                        center_crop_foreground=center_crop,
                    )
                    _render_prediction_panel(pred_res, model_name=model.__class__.__name__)
                else:
                    st.info("Draw a digit on the canvas above to run real-time inference.")
            else:
                st.info("Draw a digit on the canvas above to run real-time inference.")
        except Exception as exc:
            st.warning(
                f"Interactive drawing canvas unavailable ({exc}). "
                "Use the **Upload an Image** or **Test Set Gallery** tabs."
            )

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a handwritten digit image (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
        )
        col_inv1, col_inv2 = st.columns(2)
        with col_inv1:
            auto_invert = st.checkbox(
                "Auto-invert dark ink on light paper",
                value=True,
            )
        with col_inv2:
            force_invert = st.checkbox(
                "Force polarity inversion (255 - pixel)",
                value=False,
            )

        if uploaded_file is not None:
            pil_uploaded = Image.open(uploaded_file)
            pred_res = predict_pil_image(
                model=model,
                image=pil_uploaded,
                invert=force_invert,
                auto_invert=auto_invert,
                center_crop_foreground=center_crop,
            )
            _render_prediction_panel(pred_res, model_name=model.__class__.__name__)

    with tab_sample:
        processed_test_path = Path("data") / "processed" / "test.pt"
        if processed_test_path.exists():
            test_images, test_labels = load_processed_split(Path("data") / "processed", "test")
            max_idx = int(test_images.shape[0]) - 1
            sample_idx = st.slider(
                "Select Held-Out MNIST Test Sample Index",
                min_value=0,
                max_value=min(max_idx, 500),
                value=0,
            )
            sample_tensor = test_images[sample_idx : sample_idx + 1]
            true_digit = int(test_labels[sample_idx].item())
            st.caption(f"**Ground-Truth Test Label**: Digit `{true_digit}`")
            pred_res = predict_tensor(model=model, image_tensor=sample_tensor)
            _render_prediction_panel(pred_res, model_name=model.__class__.__name__)
        else:
            digit_choice = st.selectbox("Select Canonical Digit Template", list(range(10)), index=3)
            tpl_tensor = generate_synthetic_digit_template(int(digit_choice))
            norm_tensor = ((tpl_tensor - 0.1307) / 0.3081).unsqueeze(0)
            pred_res = predict_tensor(model=model, image_tensor=norm_tensor)
            _render_prediction_panel(pred_res, model_name=model.__class__.__name__)


if __name__ == "__main__":
    run_app()
