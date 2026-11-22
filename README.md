# MNIST Lab — Reproducible PyTorch Classification Pipeline & HF Space Demo

[![CI](https://github.com/chrisjcc/mnist-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/chrisjcc/mnist-lab/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed: mypy](https://img.shields.io/badge/typed-mypy%20strict-blue.svg)](https://mypy-lang.org/)

A reproducible, configuration-driven **MNIST handwritten digit classification pipeline** built with **PyTorch**, **Hydra**, **DVC**, **Weights & Biases (W&B)**, **Hugging Face Hub**, and an interactive **Streamlit Docker Space**.

---

## ✨ Key Architecture Highlights

- **Single Source of Truth (SSoT) Configuration**: Composable **Hydra** YAML schemas (`configs/`) synchronized with **DVC** parameters (`params.yaml`) and validated at runtime via strict Python dataclasses (`src/mnist_pipeline/config.py`).
- **Deterministic Data & Pipeline Versioning (`dvc.yaml`)**: Five-stage DAG (`preprocess -> train -> evaluate -> benchmark -> report`) with stratified train/validation splitting (`54,000 / 6,000 / 10,000`) and SHA-256 tensor checksums.
- **Experiment Tracking (`Weights & Biases`)**: Pluggable `ExperimentTracker` supporting `online`, `offline`, and `disabled` modes with automated metric, figure, and checkpoint artifact logging.
- **Hugging Face Hub Integration**: Automated `safetensors` serialization (`MNISTConvNet.save_pretrained` / `from_pretrained`), Jinja2-rendered Model & Dataset Cards (`docs/model_card.md`, `docs/dataset_card.md`), and CLI upload/inference scripts.
- **Multi-Architecture Support (CNN vs. Vision Transformer)**: Supports both classical convolution ([`MNISTConvNet`](src/mnist_pipeline/model.py) via `model=cnn`, 421k params, 99.04% acc) and pure attention-based vision models ([`MNISTViT`](src/mnist_pipeline/model.py) via `model=transformer`, 540k params, 95.70% acc) with isolated artifact namespacing (`artifacts/models/mnist_cnn/` vs. `artifacts/models/mnist_vit/`) and automated head-to-head comparison reporting (`make compare-models`).
- **Interactive Multi-Model Hugging Face Docker Space (`app/`, `Dockerfile`)**: Streamlit web application with dynamic architecture selector (`MNISTConvNet` vs. `MNISTViT`), in-memory `@st.cache_resource` caching, active architecture badge, freehand digit drawing (`streamlit-drawable-canvas`), image uploads with automatic polarity inversion, and live 10-class probability distributions.

---

## 📊 Minimum Viable Experiment (MVE) Baseline Results

Full experimental protocol, training curves, confusion matrix, and per-class evaluation tables are documented in [**`docs/experiments/mve_baseline_study.md`**](docs/experiments/mve_baseline_study.md).

| Metric / Benchmark | MVE Acceptance Target | Measured Baseline (`seed=42`) | Status |
| :--- | :---: | :---: | :---: |
| **Held-Out Test Accuracy ($N=10,000$)** | $\ge 98.50\%$ | **`98.93%` (`0.9893`)** | ✅ **PASS** |
| **Held-Out Test Macro F1-Score** | $\ge 0.9850$ | **`0.9892`** | ✅ **PASS** |
| **Minimum Per-Class F1 (`0`–`9`)** | $\ge 0.9800$ | **`0.9859` (Digit 7)** | ✅ **PASS** |
| **Best Validation Accuracy ($N=6,000$)** | $\ge 98.50\%$ | **`98.83%` (Epoch 5)** | ✅ **PASS** |
| **Single-Sample Latency (`batch_size=1`, P95)** | $< 25.0\text{ ms}$ | **`0.385 ms` (`3,341.6 samples/s`)** | ✅ **PASS** |
| **Batched Throughput (`batch_size=128`)** | $\ge 1,000\text{ samples/s}$ | **`10,465.1 samples/s`** | ✅ **PASS** |
| **Model Footprint (`MNISTConvNet`)** | $< 1\text{M params} / < 10\text{ MB}$ | **`421,834 params` (`1.61 MB` SafeTensors)** | ✅ **PASS** |

---

## 🔑 Required & Recommended Accounts

To run the full suite of capabilities coded in this repository (tracking experiments, uploading models/datasets, and deploying the interactive demo Space), you will need accounts and credentials for the following platforms:

| Platform | Required For | What It Enables in the Codebase | Setup & Required Credentials |
| :--- | :--- | :--- | :--- |
| **[GitHub](https://github.com/)** | **Core Codebase & CI/CD** | Repository hosting, issue/PR tracking, GitHub Actions CI workflows (`ci.yml`), automated Space deployment (`deploy-hf-demo.yml`), and container image publishing via GHCR (`docker-publish.yml`). | 1. Sign up at [github.com](https://github.com/)<br>2. Set up standard SSH/HTTPS access.<br>3. In your repo, configure GitHub Actions Secrets (`HF_TOKEN`) for deployment workflows. |
| **[Hugging Face](https://huggingface.co/)** | **Hub Releases & Spaces Demo** | Uploading trained SafeTensors weights, metrics, and Model/Dataset Cards (`make hf-upload`, `make publish-hf-release`), remote model downloading (`make hf-infer`), and hosting the interactive Streamlit Docker Space (`app/`). | 1. Sign up at [huggingface.co](https://huggingface.co/)<br>2. Generate a User Access Token with **Write** permissions under **Settings -> Access Tokens**.<br>3. Export `HF_TOKEN=hf_...` in `.env` (and add it as a GitHub Actions secret for CI Space deployment).<br>4. Create your target Space (e.g. `username/mnist-lab-demo`) with SDK set to **Docker**. |
| **[Weights & Biases (W&B)](https://wandb.ai/)** | **Experiment Tracking** | Live visualization of training/validation loss and accuracy curves, system telemetry (GPU/CPU/RAM), hyperparameter logging, and model artifact checkpoint versioning (`src/mnist_pipeline/tracker.py`). | 1. Sign up at [wandb.ai](https://wandb.ai/)<br>2. Retrieve your API key from **User Settings -> API Keys**.<br>3. Set `WANDB_API_KEY=...` and `WANDB_ENTITY=...` in your `.env`.<br>*(Note: Set `wandb.mode=online` when running training to stream live metrics).* |

> **Environment Configuration**: Copy `.env.example` to `.env` and fill in your keys:
> ```bash
> cp .env.example .env
> ```

---

## 🚀 Quickstart

> [!NOTE]
> **DVC is already initialized** in this repository (`.dvc/` configuration and `.dvcignore` are tracked in Git). You do **not** need to run `dvc init` when cloning. Once dependencies are installed via `make setup`, you can immediately run `make dvc-repro` (or `dvc repro`) to reproduce or verify stages.

```bash
# 1. Create and activate a Conda environment (or Python venv) and install dependencies
conda create -n mnist-lab python=3.11 -y
conda activate mnist-lab
make setup

# 2. Run zero-budget synthetic smoke check (< 1s)
make smoke-eval

# 3. Run the end-to-end pipeline (Download -> Train -> Evaluate -> Benchmark -> Report)
make data
make train
make evaluate
make benchmark
make report
make evaluate-mve

# 4. Run inference on a test sample or custom image
make hf-infer ARGS="hf.sample_index=0"

# 5. Launch the interactive Streamlit demo locally
make app
```

---

## 🛠️ Makefile Command Reference

All pipeline operations, training tasks, evaluation benchmarks, and deployment workflows are orchestrated via `make`. Run `make help` to inspect available options in your terminal.

### 1. Environment & Code Quality
| Command | Description |
| :--- | :--- |
| `make setup` | Install `requirements.txt` and editable package (`pip install -e .`) |
| `make install` | Install all project dependencies (training, development, UI) |
| `make install-dev` | Install minimal development dependencies only |
| `make lint` | Run `ruff check`, `ruff format --check`, and strict `mypy src/` |
| `make format` | Auto-format codebase via `ruff format` |
| `make test` | Execute full `pytest` unit and integration test suite |
| `make clean` | Remove build caches, bytecode (`__pycache__`), and temporary test files |

### 2. Data & Pipeline Versioning (DVC)
| Command | Description |
| :--- | :--- |
| `make data` | Download and preprocess MNIST dataset into stratified splits (`54k / 6k / 10k`) |
| `make preprocess` | Alias for `make data` |
| `make pipeline` | Reproduce full 5-stage DVC pipeline DAG (`dvc repro`) |
| `make dvc-repro` | Alias for `make pipeline` |

### 3. Model Training, Evaluation, Benchmarking & Reporting
| Command | Description |
| :--- | :--- |
| `make train` | Train model using Hydra configuration (`ARGS="..."` for overrides, e.g. `model=cnn` or `model=transformer`) |
| `make evaluate` | Evaluate best checkpoint on the 10,000 held-out test set |
| `make benchmark` | Run inference latency (`p50`, `p95`, `p99`) and throughput benchmarks across batch sizes |
| `make report` | Generate training curves, confusion matrix, sample predictions, and HF cards |
| `make compare-models` | Systematically compare CNN vs. ViT metrics and compile comparison report |
| `make smoke-eval` | Execute fast (< 2.0s) zero-budget CI smoke check with synthetic data |
| `make evaluate-mve` | Adjudicate all 5 Minimum Viable Experiment (MVE) acceptance criteria gates |

### 4. Hyperparameter Optimization (W&B Sweeps)
| Command | Description |
| :--- | :--- |
| `make sweep` | Initialize and run W&B Bayesian hyperparameter sweep |
| `make sweep-smoke` | Run fast local 1-epoch sweep trial smoke test |
| `make sweep-report` | Generate local Markdown sweep report and ranked leaderboard |
| `make sweep-promote` | Promote champion sweep model, export SafeTensors, update `params.yaml`, and re-evaluate |
| `make sweep-publish-hf` | Promote best sweep model and release weights and model card to Hugging Face Hub |

### 5. Hugging Face Hub Integration
| Command | Description |
| :--- | :--- |
| `make generate-hf-cards` | Generate Model Card & Dataset Card locally without uploading (`--card-only`) |
| `make hf-upload` | Stage and upload model weights, metrics, plots, and cards to Hugging Face Hub |
| `make hf-inference` | Download model from Hugging Face Hub (or local fallback) and run CLI digit inference (alias: `make hf-infer`) |
| `make publish-hf-release` | Publish verified production model weights, dataset card, and model cards to Hugging Face Hub |

### 6. Interactive Demo App & Docker Space
| Command | Description |
| :--- | :--- |
| `make app-run` | Launch local interactive Streamlit demo UI on `http://localhost:8501` (aliases: `make app`, `make run-app`) |
| `make docker-build` | Build the Hugging Face Docker Space container image |
| `make docker-run` | Run the Hugging Face Docker Space container locally on port `8501` |

---

## 📚 Documentation Index

- [**Step-by-Step User Guide (`docs/user_guide.md`)**](docs/user_guide.md): Comprehensive guide covering setup, data preprocessing, Hydra training (CNN & ViT), W&B tracking, evaluation/benchmarking, reporting, HF Hub upload/inference, and Docker Space deployment.
- [**CNN vs. Vision Transformer Study (`docs/experiments/cnn_vs_transformer_study.md`)**](docs/experiments/cnn_vs_transformer_study.md): Empirical head-to-head comparison study evaluating accuracy, latency scaling, throughput, parameter efficiency, and Pareto frontier.
- [**ViT Feasibility Study (`docs/research/transformer_mnist_feasibility_study.md`)**](docs/research/transformer_mnist_feasibility_study.md): Deep architectural analysis, theoretical formulation, patch projection math, and multi-architecture roadmap.
- [**PyTorch Neural Network & Training Loop Tutorial (`docs/tutorials/pytorch_nn_and_cnn_training_tutorial.md`)**](docs/tutorials/pytorch_nn_and_cnn_training_tutorial.md): Beginner-friendly guide to Feedforward (MLP) and Convolutional Neural Networks (CNNs) with an annotated training loop walkthrough.
- [**MVE Baseline Study (`docs/experiments/mve_baseline_study.md`)**](docs/experiments/mve_baseline_study.md): Reproducible baseline instructions, acceptance gate adjudication, training curves, and test evaluation/benchmark tables.
- [**Model Card (`docs/model_card.md`)**](docs/model_card.md): Architecture details, training hyperparameters, per-class evaluation metrics, and latency benchmarks.
- [**Dataset Card (`docs/dataset_card.md`)**](docs/dataset_card.md): Dataset provenance, normalization formula, and stratified class distribution across splits.
- [**Implementation Plan (`docs/mnist_pytorch_pipeline_implementation_plan.md`)**](docs/mnist_pytorch_pipeline_implementation_plan.md): Complete architectural specification and design blueprint.

