# MNIST Image Classification Pipeline: Implementation Plan

## 1. Objective

Implement a reproducible, configuration-driven, PyTorch-based pipeline for training, evaluating, benchmarking, and reporting the performance of an image classifier on the MNIST handwritten-digit dataset, complete with data/pipeline versioning, experiment tracking, Hugging Face Hub integration, an interactive Hugging Face Docker Space demo, CI/CD automation, Minimum Viable Experiment (MVE) baseline verification, and comprehensive documentation.

The pipeline should:

- Download, validate, and preprocess MNIST data through `torchvision.datasets.MNIST` and version datasets and pipeline stages with **DVC** (`dvc.yaml` and `params.yaml`).
- Manage all configuration for data, model architecture, training, evaluation, benchmarking, experiment tracking, and Hugging Face Hub integration using **Hydra** YAML configurations (`configs/`).
- Expose a standardized **Makefile** interface for environment setup, linting/type-checking, downloading and preprocessing data, training, evaluation, benchmarking, report generation, MVE verification, Hugging Face Hub synchronization, and demo deployment.
- Follow Python software engineering best practices: strict static typing (`mypy --strict`), **Google-style docstrings**, and automated formatting/linting (`ruff`).
- Train a configurable convolutional neural network (CNN) using PyTorch with deterministic data loading and execution controls.
- Track experiments, configurations, training/validation curves, system metrics, and evaluation visualizations with **Weights & Biases (W&B)**.
- Evaluate the best model on a held-out test set only after model selection is complete, and run dedicated latency/throughput performance benchmarks.
- Adjudicate a **Minimum Viable Experiment (MVE)** protocol (`docs/experiments/mve_baseline_study.md`) establishing reproducible baseline acceptance criteria across training curves, classification metrics, and inference latency/throughput benchmarks.
- Generate self-contained evaluation reports, plots, model cards, and dataset cards.
- Integrate with the **Hugging Face Hub** to upload and download trained models, model cards, dataset cards, and evaluation artifacts, and run local or remote inference.
- Provide an interactive **Streamlit** web application packaged with a root `Dockerfile` for deployment as a **Hugging Face Docker Space**, allowing users to upload or draw a handwritten digit and view predicted class probabilities using the model downloaded from the Hugging Face Model Hub.
- Automate continuous integration, Docker image publishing, and Hugging Face Space deployment via **GitHub Actions** (`.github/workflows/ci.yml`, `.github/workflows/docker-publish.yml`, `.github/workflows/deploy-hf-demo.yml`).
- Provide a comprehensive, step-by-step user guide in `docs/user_guide.md`.

---

## 2. Scope and assumptions

### In scope

- Single-label, 10-class classification (`0` through `9`).
- The standard MNIST train/test split: 60,000 training examples and 10,000 test examples, plus a deterministic stratified validation split (`54,000` train / `6,000` validation).
- End-to-end pipeline orchestration and artifact tracking via **DVC** (`dvc.yaml`).
- Hierarchical YAML configuration management and CLI overrides via **Hydra**.
- Standardized developer and pipeline workflows via a **Makefile**.
- Dual dependency specification using `pyproject.toml` (metadata, build configuration, package dependencies, `ruff`/`mypy`/`pytest` config) and `requirements.txt` (reproducible environment setup).
- Experiment tracking, metric logging, configuration tracking, and visual artifact logging via **Weights & Biases (W&B)**.
- Training from scratch, checkpointing, evaluation, latency/throughput benchmarking, MVE acceptance gate adjudication, and automated report generation.
- **Hugging Face Hub integration**:
  - Uploading trained model weights (`model.safetensors`, `checkpoint_best.pt`), resolved configs, model cards, dataset cards, and evaluation artifacts to the Hugging Face Hub.
  - Downloading the trained model from the Hugging Face Model Hub and running batch or single-image inference.
- **Hugging Face Spaces demo**:
  - A Streamlit application (`app/app.py`) packaged with a root `Dockerfile` and `.github/hf_space_readme.md` for deployment as a Hugging Face Docker Space, supporting both image upload and interactive freehand digit drawing.
- **CI/CD & Documentation**:
  - GitHub Actions workflows (`ci.yml`, `docker-publish.yml`, `deploy-hf-demo.yml`).
  - A Minimum Viable Experiment study (`docs/experiments/mve_baseline_study.md`) with generated baseline figures (`docs/experiments/figures/*.png`) and evaluation tables.
  - A comprehensive 9-part step-by-step user guide (`docs/user_guide.md`), `README.md`, Model Card (`docs/model_card.md`), and Dataset Card (`docs/dataset_card.md`).

### Out of scope for the initial implementation

- Distributed or multi-GPU training.
- Automated neural architecture search (NAS).
- Using the held-out test set for hyperparameter tuning or checkpoint selection.

### Proposed baseline

Use a compact CNN (`MNISTConvNet`) with two convolutional blocks, ReLU activations, max pooling, dropout regularization, and a fully connected classification head. The model returns raw logits; `CrossEntropyLoss` applies log-softmax internally during training, and `torch.softmax` is applied during inference to produce class probabilities.

| Setting | Initial value |
| :--- | :--- |
| Input shape | `1 × 28 × 28` |
| Number of classes | `10` |
| Conv channels | `[32, 64]` |
| Hidden dimension | `128` |
| Dropout | `0.25` |
| Batch size | `128` |
| Epochs | `10` (MVE fast baseline: `3`–`10` epochs) |
| Optimizer | `Adam` |
| Learning rate | `1e-3` |
| Weight decay | `0.0` |
| Loss | `CrossEntropyLoss` |
| Validation fraction | `10%` of training data (`54,000` train / `6,000` val) |
| Random seed | `42` |
| Best-checkpoint criterion | Highest validation accuracy |
| Early stopping | Configurable (disabled in baseline) |
| Experiment tracker | Weights & Biases (`online`, `offline`, or `disabled` mode) |

---

## 3. Repository structure

Organize the repository as an installable, strictly typed Python package (`src/mnist_pipeline`) alongside Hydra configs, DVC pipeline definitions, CLI scripts, Makefile targets, GitHub Actions CI/CD workflows, Hugging Face card templates, the Dockerized Streamlit Space, MVE experiment documentation, tests, and user guides.

```text
mnist-lab/
├── README.md
├── LICENSE
├── Dockerfile                         # Container definition for HF Docker Space & GHCR
├── Makefile                           # Standardized operational commands for the full lifecycle
├── pyproject.toml                     # Project metadata, build system, dependencies, ruff/mypy/pytest
├── requirements.txt                   # Locked/installable environment dependencies
├── params.yaml                        # Root parameter mirror for DVC stage tracking
├── dvc.yaml                           # DVC pipeline stages (preprocess, train, evaluate, benchmark, report)
├── .env.example                       # Environment variables template (WANDB_API_KEY, HF_TOKEN)
├── .gitignore
├── .dvcignore
├── .github/
│   ├── hf_space_readme.md             # Hugging Face Space metadata header & description
│   └── workflows/
│       ├── ci.yml                     # Lint (ruff), type-check (mypy), smoke eval, MVE gates, pytest
│       ├── deploy-hf-demo.yml         # Automated deployment of Streamlit demo to HF Space
│       └── docker-publish.yml         # Multi-arch Docker build & GHCR publishing on release
├── configs/                           # Hydra configuration hierarchy (Single Source of Truth)
│   ├── config.yaml                    # Top-level default composition
│   ├── data/
│   │   └── mnist.yaml                 # Dataset paths, normalization, split seed, loader settings
│   ├── model/
│   │   └── cnn.yaml                   # CNN channel sizes, kernel sizes, dropout, num_classes
│   ├── train/
│   │   └── default.yaml               # Epochs, optimizer, lr, weight_decay, checkpointing, device
│   ├── evaluate/
│   │   └── default.yaml               # Evaluation & benchmark batch sizes, warmup/timed iterations
│   ├── wandb/
│   │   └── default.yaml               # W&B project, entity, mode (online/offline/disabled), tags
│   ├── hf/
│   │   └── default.yaml               # HF Hub repo_id, dataset_repo_id, revision, cache_dir
│   └── experiment/
│       └── baseline.yaml              # Reproducible baseline experiment preset
├── src/
│   └── mnist_pipeline/
│       ├── __init__.py
│       ├── config.py                  # Structured config schemas & validation (OmegaConf/Hydra)
│       ├── data.py                    # Download, deterministic split, preprocessing, DataLoaders
│       ├── model.py                   # CNN architecture, safetensors & HF Hub serialization
│       ├── train.py                   # Training & validation loops, checkpointing, W&B logging
│       ├── evaluate.py                # Held-out test evaluation & latency/throughput benchmarking
│       ├── inference.py               # Local & Hugging Face Hub model loading and prediction API
│       ├── metrics.py                 # Accuracy, per-class precision/recall/F1, confusion matrix
│       ├── reporting.py               # Plot generation, Markdown/JSON/CSV report & card builder
│       ├── tracker.py                 # Weights & Biases wrapper (init, config, metrics, artifacts)
│       ├── hf_hub.py                  # Upload/download model, cards, and artifacts to/from HF Hub
│       └── utils.py                   # Seeding, device selection, timing, environment metadata
├── scripts/
│   ├── preprocess.py                  # Stage 1: Download & prepare processed tensors + dataset stats
│   ├── train.py                       # Stage 2: Train model, save checkpoints, log to W&B
│   ├── evaluate.py                    # Stage 3: Evaluate best checkpoint on held-out test set
│   ├── benchmark.py                   # Stage 4: Run inference latency, throughput & memory benchmarks
│   ├── generate_report.py             # Stage 5: Generate plots, summary report, and model/dataset cards
│   ├── run_smoke_eval.py              # Fast (< 2.0s) in-memory synthetic sanity check for CI
│   ├── run_mve_evaluation.py          # Verify MVE acceptance gates across training & benchmark outputs
│   ├── release_to_hf.py               # Generate/publish HF Model & Dataset Cards and release artifacts
│   ├── hf_upload.py                   # Upload trained model, model card, dataset card & metrics to HF Hub
│   └── hf_inference.py                # Download model from HF Hub and run inference on images
├── templates/
│   ├── model_card_template.md         # Template for Hugging Face Model Card (README.md)
│   └── dataset_card_template.md       # Template for Hugging Face Dataset Card (README.md)
├── app/                               # Hugging Face Docker Space (Streamlit demo)
│   ├── requirements.txt               # Runtime dependencies for the Streamlit container
│   ├── app.py                         # Primary Streamlit application entrypoint
│   └── streamlit_app.py               # Alias/wrapper entrypoint for CLI convenience
├── docs/
│   ├── mnist_pytorch_pipeline_implementation_plan.md
│   ├── user_guide.md                  # Comprehensive 9-section step-by-step user guide
│   ├── model_card.md                  # Rendered Hugging Face Model Card documentation
│   ├── dataset_card.md                # Rendered Hugging Face Dataset Card documentation
│   └── experiments/
│       ├── mve_baseline_study.md      # Minimum Viable Experiment (MVE) instructions, curves & tables
│       └── figures/                   # Reproducible baseline curves & visual diagnostics
│           ├── loss_curve.png
│           ├── accuracy_curve.png
│           ├── confusion_matrix.png
│           └── sample_predictions.png
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_metrics.py
│   ├── test_checkpoint.py
│   ├── test_train.py
│   ├── test_inference.py
│   ├── test_reporting.py
│   └── test_hf_hub.py
├── data/                              # Tracked by DVC, ignored by Git
│   ├── raw/                           # Raw torchvision MNIST files
│   └── processed/                     # Deterministic train/val/test tensors & split metadata
└── artifacts/                         # Pipeline outputs tracked by DVC / logged to W&B
    ├── checkpoints/
    │   ├── checkpoint_best.pt
    │   ├── checkpoint_last.pt
    │   ├── config.json
    │   └── model.safetensors          # Weights exported for Hugging Face Hub
    ├── evaluation/
    │   ├── test_metrics.json
    │   ├── classification_report.txt
    │   └── confusion_matrix.csv
    ├── benchmarks/
    │   └── benchmark_metrics.json
    └── reports/
        ├── train_history.csv
        ├── environment.json
        ├── evaluation_report.md
        └── plots/
            ├── loss_curve.png
            ├── accuracy_curve.png
            ├── confusion_matrix.png
            └── sample_predictions.png
```

---

## 4. Architecture and key integrations

### 4.1 Dependency management & code quality (`pyproject.toml` and `requirements.txt`)

- **`pyproject.toml`**:
  - Defines `[build-system]` (`setuptools>=68.0`, `wheel`), `[project]` metadata (`name = "mnist-pipeline"`, `requires-python = ">=3.11"`, core dependencies), and `[project.optional-dependencies]` (`training`, `ui`, `dev`, `all`).
  - Configures `ruff` (`line-length = 100`, Google docstring convention `D`, `E`, `F`, `W`, `I`, `N`, `UP`, `B`, `C4`, `SIM`), `mypy` (`strict = true`), and `pytest`.
- **`requirements.txt`**:
  - Provides direct environment installation (`pip install -r requirements.txt` and `pip install -e .`) for both local virtual environments and Docker builds.

### 4.2 Configuration management with Hydra (`configs/`)

- Use Hydra (`configs/config.yaml`) as the Single Source of Truth (SSoT) for configuration across all pipeline scripts, supporting both `@hydra.main(version_base="1.3", config_path="../configs", config_name="config")` and programmatic composition (`initialize_config_dir` / `compose`).
- Structured dataclasses in `src/mnist_pipeline/config.py` validate all configuration parameters at startup and raise descriptive `ValueError` exceptions on invalid settings.

### 4.3 Data and pipeline versioning with DVC (`dvc.yaml` & `params.yaml`)

Define a 5-stage deterministic DAG in `dvc.yaml`:

1. **`preprocess`**: Runs `python scripts/preprocess.py`, producing `data/processed/train.pt`, `data/processed/val.pt`, `data/processed/test.pt`, and `data/processed/split_metadata.json`.
2. **`train`**: Runs `python scripts/train.py`, producing `artifacts/checkpoints/checkpoint_best.pt`, `artifacts/checkpoints/checkpoint_last.pt`, `artifacts/checkpoints/model.safetensors`, `artifacts/checkpoints/config.json`, `artifacts/reports/train_history.csv`, and `artifacts/reports/environment.json`.
3. **`evaluate`**: Runs `python scripts/evaluate.py`, producing metrics `artifacts/evaluation/test_metrics.json` and outputs `artifacts/evaluation/classification_report.txt` and `artifacts/evaluation/confusion_matrix.csv`.
4. **`benchmark`**: Runs `python scripts/benchmark.py`, producing `artifacts/benchmarks/benchmark_metrics.json`.
5. **`report`**: Runs `python scripts/generate_report.py`, producing plots in `artifacts/reports/plots/`, `artifacts/reports/evaluation_report.md`, `docs/model_card.md`, and `docs/dataset_card.md`.

### 4.4 Makefile workflow automation (`Makefile`)

Provide self-documenting targets in `Makefile` with automatic `.venv` resolution and `PYTHONNOUSERSITE=1` isolation:

| Target | Description | Underlying command |
| :--- | :--- | :--- |
| `make help` | Display categorized operational commands | Echo formatted command summary |
| `make setup` / `make install` | Install dependencies and package in editable mode | `pip install -r requirements.txt && pip install -e ".[all]"` |
| `make lint` | Run `ruff` linter/formatter check and `mypy` type check | `ruff check ... && ruff format --check ... && mypy src/` |
| `make format` | Auto-format codebase via `ruff` | `ruff check --fix ... && ruff format ...` |
| `make test` | Execute `pytest` unit and integration test suite | `pytest tests/ -v` |
| `make smoke-eval` | Execute fast (< 2.0s) in-memory synthetic sanity check | `python scripts/run_smoke_eval.py` |
| `make data` / `make preprocess` | Download and preprocess the MNIST dataset | `python scripts/preprocess.py $(ARGS)` |
| `make train` | Train the model using Hydra configuration | `python scripts/train.py $(ARGS)` |
| `make evaluate` | Evaluate best checkpoint on held-out test set | `python scripts/evaluate.py $(ARGS)` |
| `make benchmark` | Run latency, throughput, and memory benchmarks | `python scripts/benchmark.py $(ARGS)` |
| `make report` | Generate evaluation reports, plots, and HF cards | `python scripts/generate_report.py $(ARGS)` |
| `make evaluate-mve` | Adjudicate MVE baseline acceptance criteria gates | `python scripts/run_mve_evaluation.py $(ARGS)` |
| `make pipeline` / `make dvc-repro` | Run the full pipeline via DVC | `dvc repro $(ARGS)` |
| `make generate-hf-cards` | Generate Model Card & Dataset Card locally | `python scripts/release_to_hf.py --card-only $(ARGS)` |
| `make hf-upload` | Upload model, model card, dataset card & artifacts to HF Hub | `python scripts/hf_upload.py $(ARGS)` |
| `make hf-inference` | Download model from HF Hub and run inference | `python scripts/hf_inference.py $(ARGS)` |
| `make app-run` / `make run-app` | Launch the Streamlit demo locally | `streamlit run app/app.py` |
| `make docker-build` | Build the Hugging Face Space Docker image | `docker build -t mnist-lab-app .` |
| `make docker-run` | Run the Hugging Face Space container on port 8501 | `docker run -p 8501:8501 mnist-lab-app` |
| `make clean` | Clean temporary cache and build artifacts | Remove `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache` |

### 4.5 Experiment tracking with Weights & Biases (`src/mnist_pipeline/tracker.py`)

- Initialize W&B runs in `train.py` and `evaluate.py` using settings from `configs/wandb/default.yaml`.
- Support `wandb.mode` values of `online`, `offline`, and `disabled`.
- Log resolved Hydra configuration, epoch-level training/validation metrics, test evaluation metrics, latency/throughput benchmarks, diagnostic plots (`loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `sample_predictions.png`), and model checkpoint artifacts.

### 4.6 Hugging Face Hub integration (`src/mnist_pipeline/hf_hub.py`)

- **Model serialization and Hub loading**:
  - Save model weights in `safetensors` format (`model.safetensors`) alongside `config.json` and `checkpoint_best.pt`.
  - Support `MNISTConvNet.save_pretrained(...)` and `MNISTConvNet.from_pretrained(...)` from either a local directory or a Hugging Face Hub `repo_id`.
- **Upload and card release workflow (`scripts/hf_upload.py` and `scripts/release_to_hf.py`)**:
  - Bundles `model.safetensors`, `checkpoint_best.pt`, `config.json`, rendered `README.md` (Model Card), `test_metrics.json`, `benchmark_metrics.json`, and diagnostic plots, and uploads them via `huggingface_hub.HfApi`.
  - Generates and optionally publishes the Dataset Card (`docs/dataset_card.md`) to a Hugging Face Dataset Hub repository.
- **Download and inference workflow (`scripts/hf_inference.py` and `src/mnist_pipeline/inference.py`)**:
  - Downloads the trained model from the Hugging Face Model Hub (with local fallback when offline) and runs inference on user-supplied images or MNIST test samples, returning the predicted digit (`0–9`) and 10-class probability distribution.

### 4.7 Hugging Face Spaces interactive demo & CI/CD (`app/`, `Dockerfile`, `.github/`)

- **Streamlit application (`app/app.py` and `app/streamlit_app.py`)**:
  - Downloads and caches (`@st.cache_resource`) the trained model from the Hugging Face Model Hub (with automatic fallback to local `artifacts/checkpoints` or on-the-fly initialization so the UI never crashes in offline environments).
  - Supports **Draw a Digit** (interactive canvas via `streamlit-drawable-canvas`) and **Upload an Image** (PNG/JPG/JPEG with optional polarity inversion), plus sample MNIST test digits.
  - Displays the preprocessed `28 × 28` normalized tensor preview, predicted digit (`0–9`), confidence percentage, and interactive 10-class probability bar chart.
- **Container & CI/CD Workflows**:
  - Root `Dockerfile` builds a lightweight `python:3.11-slim` image exposing port `8501` (matching `.github/hf_space_readme.md` with `sdk: docker` and `app_port: 8501`).
  - `.github/workflows/ci.yml`: Runs `ruff check`, `ruff format --check`, `mypy src/`, `python scripts/run_smoke_eval.py`, `python scripts/run_mve_evaluation.py --smoke`, and `pytest tests/ -v`.
  - `.github/workflows/docker-publish.yml`: Verifies multi-architecture Docker builds on pushes to `main` and publishes to `ghcr.io` on GitHub Releases.
  - `.github/workflows/deploy-hf-demo.yml`: Pushes the minimal inference payload (`Dockerfile`, `pyproject.toml`, `app/`, `configs/`, `src/mnist_pipeline/`, `.github/hf_space_readme.md`) to the Hugging Face Space on release or manual dispatch.

---

## 5. Phased implementation roadmap & GitHub issue tracking

Development is tracked via a **Parent Epic GitHub Issue (#1)** and **5 Phased Sub-Issues (#2–#6)**, each delivered via a dedicated feature branch and Pull Request with a pre-merge verification summary:

| Phase & Sub-Issue | Deliverable Scope | Acceptance Gate |
| :--- | :--- | :--- |
| **Phase 1 (Issue #2)** | **Repository Skeleton, SSoT Hydra Configuration, DVC DAG & CI/CD Workflows**: `pyproject.toml`, `requirements.txt`, `Makefile`, `params.yaml`, `dvc.yaml`, Hydra `configs/`, `.github/workflows/`, `src/mnist_pipeline/config.py`, `src/mnist_pipeline/utils.py`, `tests/test_config.py`. | `make lint` & `pytest tests/test_config.py` pass with 0 errors. |
| **Phase 2 (Issue #3)** | **Deterministic Data Ingestion, Preprocessing & CNN Model Architecture**: `src/mnist_pipeline/data.py`, `src/mnist_pipeline/model.py`, `scripts/preprocess.py`, `scripts/run_smoke_eval.py`, `tests/test_data.py`, `tests/test_model.py`, `tests/test_checkpoint.py`. | `make smoke-eval`, `make data`, and data/model unit tests pass. |
| **Phase 3 (Issue #4)** | **Training Pipeline, Checkpointing & Weights & Biases Experiment Tracking**: `src/mnist_pipeline/tracker.py`, `src/mnist_pipeline/train.py`, `scripts/train.py`, `tests/test_train.py`. | CPU training smoke test, checkpoint resume, and W&B tracker tests pass. |
| **Phase 4 (Issue #5)** | **Held-Out Test Evaluation, Latency/Throughput Benchmarking & Automated Reporting**: `src/mnist_pipeline/metrics.py`, `src/mnist_pipeline/evaluate.py`, `src/mnist_pipeline/reporting.py`, `scripts/evaluate.py`, `scripts/benchmark.py`, `scripts/generate_report.py`, `scripts/run_mve_evaluation.py`, `templates/`, `tests/test_metrics.py`, `tests/test_reporting.py`. | Full evaluation, benchmarking, report generation, and MVE gates pass. |
| **Phase 5 (Issue #6)** | **Hugging Face Hub Integration, Interactive Streamlit Docker Space, MVE Baseline Study & User Guide**: `src/mnist_pipeline/inference.py`, `src/mnist_pipeline/hf_hub.py`, `scripts/hf_upload.py`, `scripts/hf_inference.py`, `scripts/release_to_hf.py`, `app/`, `Dockerfile`, `docs/model_card.md`, `docs/dataset_card.md`, `docs/user_guide.md`, `docs/experiments/mve_baseline_study.md` + `docs/experiments/figures/`, `README.md`, `tests/test_inference.py`, `tests/test_hf_hub.py`. | Full MVE baseline run (`Test Accuracy >= 98.5%`), all 9 user guide sections, HF cards, and 100% test suite pass. |

---

## 6. Minimum Viable Experiment (MVE) baseline & acceptance criteria

Documented in `docs/experiments/mve_baseline_study.md`, the MVE establishes reproducible baseline results on the full MNIST dataset (`54,000` train, `6,000` validation, `10,000` held-out test examples, `seed=42`):

| Gate ID | Metric | Acceptance Threshold | Verification Source |
| :--- | :--- | :--- | :--- |
| **Gate 1 (Data Integrity)** | Exact split sizes & class stratification | `54,000 / 6,000 / 10,000` | `data/processed/split_metadata.json` |
| **Gate 2 (Training Convergence)** | Validation accuracy improvement & finite loss | `val_accuracy >= 98.50%` | `artifacts/reports/train_history.csv` |
| **Gate 3 (Held-Out Test Quality)** | Test accuracy & macro F1 on 10,000 examples | `test_accuracy >= 98.50%`, `macro_f1 >= 0.985` | `artifacts/evaluation/test_metrics.json` |
| **Gate 4 (Inference Latency & Throughput)** | Single-sample (`B=1`) P95 latency & batched (`B=128`) throughput | `latency_p95_ms <= 25.0 ms` (CPU), `throughput >= 1,000 samples/s` | `artifacts/benchmarks/benchmark_metrics.json` |
| **Gate 5 (Artifact & Visual Completeness)** | Generated curves, confusion matrix, evaluation report & HF cards | All 4 PNG figures + Markdown reports present | `docs/experiments/figures/*.png`, `docs/model_card.md` |

---

## 7. Definition of done

The implementation is complete when:

- Parent Epic Issue and Phase Sub-Issues are created on GitHub, and each Phase is implemented, verified, and merged via a documented Pull Request.
- `pyproject.toml` and `requirements.txt` install cleanly, and `make lint` (`ruff` + `mypy --strict`) and `make test` (`pytest`) pass with zero errors.
- `dvc.yaml` defines the 5-stage pipeline (`preprocess`, `train`, `evaluate`, `benchmark`, `report`).
- Hydra configs (`configs/`) manage all parameters without hardcoded drift.
- W&B tracking (`src/mnist_pipeline/tracker.py`) logs configurations, metrics, plots, and checkpoints.
- `docs/experiments/mve_baseline_study.md` documents the MVE protocol, embeds the generated baseline training/validation curves and confusion matrix (`docs/experiments/figures/*.png`), and presents the empirical evaluation and benchmark tables meeting all acceptance criteria.
- Hugging Face Hub integration (`scripts/hf_upload.py`, `scripts/hf_inference.py`, `scripts/release_to_hf.py`, `docs/model_card.md`, `docs/dataset_card.md`) and the Dockerized Streamlit Space (`app/app.py`, `Dockerfile`, `.github/hf_space_readme.md`) are complete and tested.
- GitHub Actions CI/CD workflows (`.github/workflows/ci.yml`, `deploy-hf-demo.yml`, `docker-publish.yml`) are configured.
- `docs/user_guide.md` covers all 9 required topics step by step.
