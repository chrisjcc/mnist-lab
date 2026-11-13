# MNIST PyTorch Classification Pipeline — Step-by-Step User Guide

This guide provides end-to-end instructions for setting up the environment, reproducing the DVC pipeline, configuring experiments with Hydra, tracking runs with Weights & Biases (W&B), running held-out test evaluations and latency benchmarks, generating publication-quality reports and cards, interacting with the Hugging Face Hub, and deploying the interactive Streamlit Docker Space.

---

## Table of Contents

1. [Environment Setup and Dependency Installation](#1-environment-setup-and-dependency-installation)
2. [Downloading and Preprocessing the MNIST Dataset](#2-downloading-and-preprocessing-the-mnist-dataset)
3. [Configuring and Running Model Training Using Hydra](#3-configuring-and-running-model-training-using-hydra)
4. [Tracking Experiments and Metrics with Weights & Biases](#4-tracking-experiments-and-metrics-with-weights--biases)
5. [Running Evaluation and Performance Benchmarks](#5-running-evaluation-and-performance-benchmarks)
6. [Generating Evaluation Reports and Visualizations](#6-generating-evaluation-reports-and-visualizations)
7. [Downloading the Trained Model from Hugging Face Hub and Running Inference](#7-downloading-the-trained-model-from-hugging-face-hub-and-running-inference)
8. [Uploading the Model, Model Card, and Artifacts to Hugging Face Hub](#8-uploading-the-model-model-card-and-artifacts-to-hugging-face-hub)
9. [Building and Deploying the Streamlit Demo as a Hugging Face Docker Space](#9-building-and-deploying-the-streamlit-demo-as-a-hugging-face-docker-space)

---

## 1. Environment Setup and Dependency Installation

### Prerequisites

- **Python**: `3.10`, `3.11`, or `3.12`
- **Git**: `>= 2.30`
- **Docker** *(optional, for local Hugging Face Space container testing)*

### Step 1.1: Clone the Repository

```bash
git clone https://github.com/chrisjcc/mnist-lab.git
cd mnist-lab
```

### Step 1.2: Create an Isolated Environment and Install Dependencies

**Option A (Recommended — Conda)**:

```bash
conda create -n mnist-lab python=3.11 -y
conda activate mnist-lab
make setup
```

**Option B (Python `venv`)**:

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
```

> **Note**: The [`Makefile`](../Makefile) automatically detects an active Conda environment (`CONDA_PREFIX`) or a local `.venv` directory.

What `make setup` executes under the hood:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[all]"
```

### Step 1.3: Configure Optional Environment Variables

Copy `.env.example` to `.env` if you plan to use online Weights & Biases logging or push artifacts to the Hugging Face Hub:

```bash
cp .env.example .env
```

Edit `.env` with your credentials (never committed to Git):

```dotenv
WANDB_API_KEY=your_wandb_api_key_here
WANDB_ENTITY=your_wandb_username_or_team
WANDB_PROJECT=mnist-classification
WANDB_MODE=disabled

HF_TOKEN=hf_your_read_write_token_here
HF_MODEL_REPO_ID=chrisjcc/mnist-cnn
HF_DATASET_REPO_ID=chrisjcc/mnist-dataset
HF_SPACE_REPO_ID=chrisjcc/mnist-streamlit-demo
```

### Step 1.4: Verify Installation with the Zero-Budget Smoke Gate

Verify that your PyTorch environment, Hydra schemas, and `MNISTConvNet` forward/backward passes work in under 1 second:

```bash
make smoke-eval
```

---

## 2. Downloading and Preprocessing the MNIST Dataset

The pipeline downloads the canonical 60,000 training and 10,000 test handwritten digit images via `torchvision.datasets.MNIST`, normalizes pixel intensities using training-split statistics ($\mu = 0.1307$, $\sigma = 0.3081$), and performs a **deterministic stratified split** (`seed = 42`, `val_fraction = 0.1`) to create:

- `data/processed/train.pt`: `54,000` training samples (`[54000, 1, 28, 28]`, `float32`)
- `data/processed/val.pt`: `6,000` class-balanced validation samples (`[6000, 1, 28, 28]`, `float32`)
- `data/processed/test.pt`: `10,000` held-out test samples (`[10000, 1, 28, 28]`, `float32`)
- `data/processed/split_metadata.json`: SHA-256 tensor checksums and per-class sample distributions (`0`–`9`)

### Running via Makefile

```bash
make data
```

### Running via Hydra CLI with Custom Overrides

```bash
python scripts/preprocess.py data.val_fraction=0.1 data.seed=42
```

### Running via DVC

```bash
dvc repro preprocess
```

---

## 3. Configuring and Running Model Training Using Hydra

All pipeline parameters are managed through composable Hydra YAML files in `configs/` bridged to `params.yaml` as the Single Source of Truth (SSoT):

```text
configs/
├── config.yaml              # Root composition defaults
├── data/mnist.yaml          # Batch size, normalization, split fraction
├── model/cnn.yaml           # Conv channels [32, 64], dropout, hidden_dim
├── train/default.yaml       # Epochs, learning rate, optimizer, device, seed
├── evaluate/default.yaml    # Eval batch size, benchmark warmup/timed runs
├── wandb/default.yaml       # W&B mode, project, entity, tags
├── hf/default.yaml          # Hugging Face Hub repository IDs
└── experiment/baseline.yaml # MVE baseline preset
```

### Step 3.1: Run Default Baseline Training (`5` Epochs)

```bash
make train
```

### Step 3.2: Override Hyperparameters from the Command Line

Use `ARGS="..."` with `make train` or invoke `scripts/train.py` directly:

```bash
# Train for 10 epochs with AdamW and lr=5e-4
make train ARGS="train.epochs=10 train.optimizer=adamw train.lr=0.0005"

# Or invoke the Hydra script directly
python scripts/train.py +experiment=baseline train.device=auto
```

### Step 3.3: Resume Training from a Saved Checkpoint

```bash
make train ARGS="train.resume_from=artifacts/checkpoints/checkpoint_last.pt train.epochs=8"
```

### Generated Training Artifacts

- `artifacts/checkpoints/checkpoint_best.pt`: Full PyTorch training state at highest validation accuracy
- `artifacts/checkpoints/checkpoint_last.pt`: Most recent epoch checkpoint
- `artifacts/checkpoints/model.safetensors` & `artifacts/checkpoints/config.json`: Hugging Face Hub-ready SafeTensors weights and architecture metadata
- `artifacts/reports/train_history.csv`: Epoch-by-epoch `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `lr`, and `epoch_time_sec`
- `artifacts/reports/environment.json`: Hardware, OS, Python, PyTorch, and seed provenance

---

## 4. Tracking Experiments and Metrics with Weights & Biases

The `ExperimentTracker` (`src/mnist_pipeline/tracker.py`) supports three operational modes controlled via `wandb.mode`:

1. **`disabled` (Default for local/CI reproducibility)**: No network calls or account required.
2. **`offline`**: Writes W&B event files locally to `./wandb/` for later synchronization (`wandb sync`).
3. **`online`**: Streams live epoch metrics, system telemetry, confusion matrices, and model checkpoints to your W&B dashboard.

### Running Training with Live W&B Tracking

```bash
export WANDB_API_KEY="your_api_key"
make train ARGS="wandb.mode=online wandb.project=mnist-classification wandb.run_name=mve-baseline-seed42"
```

### Running in Offline Mode and Syncing Later

```bash
make train ARGS="wandb.mode=offline"
wandb sync wandb/latest-run
```

---

## 5. Running Evaluation and Performance Benchmarks

### Step 5.1: Held-Out Test Set Evaluation (`N = 10,000`)

Evaluate `artifacts/checkpoints/checkpoint_best.pt` on the 10,000 held-out test images:

```bash
make evaluate
```

Outputs written to `artifacts/evaluation/`:
- `test_metrics.json`: Overall accuracy, cross-entropy loss, macro/weighted precision, recall, F1, and per-class (`0`–`9`) metrics.
- `classification_report.txt`: Formatted `scikit-learn` per-class report.
- `confusion_matrix.csv`: `10 × 10` ground-truth vs. predicted confusion matrix.

### Step 5.2: Inference Latency, Throughput & Footprint Benchmarking

Measure inference latency (`mean`, `std`, `p50`, `p95`, `p99` in milliseconds) and throughput (`samples/sec`) across batch sizes `[1, 32, 128]`:

```bash
make benchmark
```

Outputs written to `artifacts/benchmarks/benchmark_metrics.json`:
- Parameter count (`trainable_params`, `total_params`)
- Checkpoint size (`checkpoint_size_mb`, `safetensors_size_mb`)
- Per-batch-size latency percentiles and throughput

### Step 5.3: Verify Minimum Viable Experiment (MVE) Acceptance Gates

Adjudicate all 5 quantitative MVE acceptance gates automatically:

```bash
# Verify full baseline artifacts
make evaluate-mve

# Or run the fast self-contained synthetic CI smoke check
python scripts/run_mve_evaluation.py --smoke
```

---

## 6. Generating Evaluation Reports and Visualizations

Generate all diagnostic figures, the consolidated Markdown evaluation report, and the rendered Hugging Face Model and Dataset Cards:

```bash
make report
```

### Generated Visual & Documentation Artifacts

1. `artifacts/reports/plots/loss_curve.png`: Training vs. validation cross-entropy loss across epochs.
2. `artifacts/reports/plots/accuracy_curve.png`: Training vs. validation accuracy (%) across epochs.
3. `artifacts/reports/plots/confusion_matrix.png`: Annotated `10 × 10` test set confusion matrix heatmap.
4. `artifacts/reports/plots/sample_predictions.png`: `4 × 4` grid of held-out test images with predicted digit, confidence, and ground-truth label.
5. `artifacts/reports/evaluation_report.md`: Executive summary, training progression table, test metrics table, and latency benchmark table.
6. `docs/model_card.md` & `docs/dataset_card.md`: Jinja2-rendered Hugging Face Hub cards populated with empirical run metrics.

To reproduce the entire DAG (`preprocess -> train -> evaluate -> benchmark -> report`) in one command:

```bash
make dvc-repro
```

---

## 7. Downloading the Trained Model from Hugging Face Hub and Running Inference

The inference module (`src/mnist_pipeline/inference.py`) downloads `model.safetensors` and `config.json` from Hugging Face Hub (or falls back to your local `artifacts/checkpoints/` directory when offline).

### Step 7.1: Run CLI Inference on a Held-Out Test Sample

```bash
make hf-infer ARGS="hf.sample_index=7"
```

### Step 7.2: Run CLI Inference on a Custom Image File

```bash
make hf-infer ARGS="hf.image_path=path/to/handwritten_digit.png"
```

### Step 7.3: Python API Usage

```python
from PIL import Image
from mnist_pipeline.inference import load_inference_model, predict_pil_image

model, source = load_inference_model("chrisjcc/mnist-cnn", device="cpu")
image = Image.open("my_digit.png")
result = predict_pil_image(model, image, auto_invert=True)

print(f"Source: {source}")
print(f"Predicted Digit: {result.predicted_digit} ({result.confidence:.2%})")
print("Class Probabilities:", result.probabilities)
```

---

## 8. Uploading the Model, Model Card, and Artifacts to Hugging Face Hub

### Step 8.1: Dry-Run Staging (No Network Calls Required)

Verify that all release files (`model.safetensors`, `config.json`, `checkpoint_best.pt`, `README.md`, `test_metrics.json`, `benchmark_metrics.json`, and `plots/*.png`) stage cleanly:

```bash
make hf-upload ARGS="hf.dry_run=true"
```

### Step 8.2: Render Cards Only

```bash
make hf-card
```

### Step 8.3: Publish Model & Dataset Cards to Hugging Face Hub

Ensure `HF_TOKEN` is exported in your environment with write permissions:

```bash
export HF_TOKEN="hf_your_write_token"
make hf-upload ARGS="hf.repo_id=your-username/mnist-cnn hf.dataset_repo_id=your-username/mnist-dataset hf.dry_run=false"
```

Or use the release helper script:

```bash
make hf-release
```

---

## 9. Building and Deploying the Streamlit Demo as a Hugging Face Docker Space

The interactive Streamlit application (`app/app.py`) lets users:
1. **Draw a digit (`0`–`9`)** freehand on an HTML5 canvas (`streamlit-drawable-canvas`),
2. **Upload an image (`PNG`/`JPG`/`JPEG`)** with automatic or manual foreground/background polarity inversion, or
3. **Explore the Held-Out Test Set Gallery**, viewing the normalized `28 × 28` input tensor, top predicted digit, confidence, and 10-class probability bar chart.

### Step 9.1: Run the Streamlit App Locally

```bash
make app
```

Then open `http://localhost:8501` in your browser.

### Step 9.2: Build and Run the Hugging Face Space Docker Container Locally

```bash
# Build the Docker image
make docker-build

# Run the container on port 8501
make docker-run
```

Visit `http://localhost:8501` to interact with the containerized Space.

### Step 9.3: Automated CI/CD Deployment to Hugging Face Docker Spaces

The repository includes `.github/workflows/deploy-hf-demo.yml`, which automatically deploys the Space whenever changes to `app/**`, `src/mnist_pipeline/**`, `configs/**`, or `Dockerfile` are pushed to `main` (or when triggered manually via `workflow_dispatch`):

1. Create a new **Docker Space** on Hugging Face (`https://huggingface.co/new-space`, select **Docker** -> **Blank**).
2. In your GitHub repository (**Settings -> Secrets and variables -> Actions**), configure:
   - **Secret** `HF_TOKEN`: Your Hugging Face write token.
   - **Variable** `HF_SPACE_REPO`: Your Space repository ID (e.g., `chrisjcc/mnist-streamlit-demo`).
3. Push to `main` or trigger the **Deploy Demo to Hugging Face Spaces** workflow in the GitHub Actions tab. The workflow injects `.github/hf_space_readme.md` (configured with `sdk: docker` and `app_port: 8501`) as the Space's `README.md` and force-pushes the clean application bundle to `https://huggingface.co/spaces/<HF_SPACE_REPO>`.
