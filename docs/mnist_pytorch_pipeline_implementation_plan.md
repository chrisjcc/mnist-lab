# MNIST Image Classification Pipeline: Implementation Plan

## 1. Objective

Implement a reproducible, configuration-driven, PyTorch-based pipeline for training, evaluating, benchmarking, and reporting the performance of an image classifier on the MNIST handwritten-digit dataset, complete with data/pipeline versioning, experiment tracking, Hugging Face Hub integration, an interactive Hugging Face Docker Space demo, and comprehensive documentation.

The pipeline should:

- Download, validate, and preprocess MNIST data through `torchvision.datasets.MNIST` and version datasets and pipeline stages with **DVC** (`dvc.yaml`).
- Manage all configuration for data, model architecture, training, evaluation, benchmarking, experiment tracking, and Hugging Face Hub integration using **Hydra** YAML configurations.
- Expose a standardized **Makefile** interface for environment setup, downloading and preprocessing data, training, evaluation, benchmarking, report generation, Hugging Face Hub synchronization, and demo deployment.
- Train a configurable convolutional neural network (CNN) using PyTorch with deterministic data loading and execution controls.
- Track experiments, configurations, training/validation curves, system metrics, and evaluation visualizations with **Weights & Biases (W&B)**.
- Evaluate the best model on a held-out test set only after model selection is complete, and run dedicated latency/throughput performance benchmarks.
- Generate self-contained evaluation reports, plots, model cards, and dataset cards.
- Integrate with the **Hugging Face Hub** to upload and download trained models, model cards, dataset cards, and evaluation artifacts, and run local or remote inference.
- Provide an interactive **Streamlit** web application packaged with a `Dockerfile` for deployment as a **Hugging Face Docker Space**, allowing users to upload or draw a handwritten digit and view predicted class probabilities using the model downloaded from the Hugging Face Model Hub.
- Provide a comprehensive, step-by-step user guide in `docs/user_guide.md`.

---

## 2. Scope and assumptions

### In scope

- Single-label, 10-class classification (`0` through `9`).
- The standard MNIST train/test split: 60,000 training examples and 10,000 test examples, plus a deterministic validation split from the training partition.
- End-to-end pipeline orchestration and artifact tracking via **DVC** (`dvc.yaml`).
- Hierarchical YAML configuration management and CLI overrides via **Hydra**.
- Standardized developer and pipeline workflows via a **Makefile**.
- Dual dependency specification using `pyproject.toml` (metadata, build configuration, package dependencies, CLI scripts) and `requirements.txt` (reproducible environment setup).
- Experiment tracking, metric logging, configuration tracking, and visual artifact logging via **Weights & Biases (W&B)**.
- Training from scratch, checkpointing, evaluation, latency/throughput benchmarking, and automated report generation.
- **Hugging Face Hub integration**:
  - Uploading trained model weights, resolved configs, model cards, dataset cards, and evaluation artifacts to the Hugging Face Hub.
  - Downloading the trained model from the Hugging Face Model Hub and running batch or single-image inference.
- **Hugging Face Spaces demo**:
  - A Streamlit application packaged with a `Dockerfile` for deployment as a Hugging Face Docker Space, supporting both image upload and interactive freehand digit drawing.
- **Documentation**:
  - A comprehensive 9-part step-by-step user guide (`docs/user_guide.md`), `README.md`, Model Card, and Dataset Card.

### Out of scope for the initial implementation

- Distributed or multi-GPU training.
- Automated neural architecture search (NAS).
- Advanced data augmentation or multi-model ensembling in the baseline configuration.
- Using the held-out test set for hyperparameter tuning or checkpoint selection.

### Proposed baseline

Use a compact CNN with two convolutional blocks, ReLU activations, max pooling, dropout regularization, and a fully connected classification head. The model returns raw logits; `CrossEntropyLoss` applies log-softmax internally during training, and `torch.softmax` is applied during inference to produce class probabilities.

| Setting | Initial value |
| :--- | :--- |
| Input shape | `1 × 28 × 28` |
| Number of classes | `10` |
| Batch size | `128` |
| Epochs | `10` |
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

Organize the repository as an installable Python package (`src/mnist_pipeline`) alongside Hydra configs, DVC pipeline definitions, CLI scripts, Makefile targets, Hugging Face card templates, the Dockerized Streamlit Space, tests, and documentation.

```text
mnist-lab/
├── README.md
├── LICENSE
├── Makefile                           # Standardized commands for the full lifecycle
├── pyproject.toml                     # Project metadata, build system, and dependencies
├── requirements.txt                   # Pinned/installable environment dependencies
├── dvc.yaml                           # DVC pipeline stages (preprocess, train, evaluate, benchmark, report)
├── dvc.lock                           # DVC lockfile recording stage hashes (generated)
├── .gitignore
├── .dvcignore
├── configs/                           # Hydra configuration hierarchy
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
│       ├── model.py                   # CNN architecture, PyTorch Hub / HF Hub serialization mixin
│       ├── train.py                   # Training & validation loops, checkpointing, W&B logging
│       ├── evaluate.py                # Held-out test evaluation & latency/throughput benchmarking
│       ├── inference.py               # Local & Hugging Face Hub model loading and prediction API
│       ├── metrics.py                 # Accuracy, per-class precision/recall/F1, confusion matrix
│       ├── reporting.py               # Plot generation, Markdown/JSON/CSV report builder
│       ├── tracker.py                 # Weights & Biases wrapper (init, config, metrics, artifacts)
│       ├── hf_hub.py                  # Upload/download model, cards, and artifacts to/from HF Hub
│       └── utils.py                   # Seeding, device selection, timing, environment metadata
├── scripts/
│   ├── preprocess.py                  # Stage 1: Download & prepare processed tensors + dataset stats
│   ├── train.py                       # Stage 2: Train model, save checkpoints, log to W&B
│   ├── evaluate.py                    # Stage 3: Evaluate best checkpoint on held-out test set
│   ├── benchmark.py                   # Stage 4: Run inference latency, throughput & memory benchmarks
│   ├── generate_report.py             # Stage 5: Generate plots, summary report, and model/dataset cards
│   ├── hf_upload.py                   # Upload trained model, model card, dataset card & metrics to HF Hub
│   └── hf_inference.py                # Download model from HF Hub and run inference on images
├── templates/
│   ├── model_card_template.md         # Jinja2/format template for Hugging Face Model Card (README.md)
│   └── dataset_card_template.md       # Jinja2/format template for Hugging Face Dataset Card (README.md)
├── app/                               # Hugging Face Docker Space (Streamlit demo)
│   ├── Dockerfile                     # Container definition for HF Docker Space (port 7860)
│   ├── README.md                      # HF Space metadata header (sdk: docker, app_port: 7860)
│   ├── requirements.txt               # Minimal runtime dependencies for the Streamlit container
│   └── streamlit_app.py               # Upload/draw digit UI, HF Hub model loader, probability bar chart
├── docs/
│   ├── mnist_pytorch_pipeline_implementation_plan.md
│   ├── user_guide.md                  # Comprehensive 9-section step-by-step user guide
│   ├── model_card.md                  # Rendered model card documentation
│   └── dataset_card.md                # Rendered dataset card documentation
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_metrics.py
│   ├── test_checkpoint.py
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

### 4.1 Dependency management (`pyproject.toml` and `requirements.txt`)

- **`pyproject.toml`**:
  - Defines `[build-system]` (e.g., `setuptools>=68.0`, `wheel`), `[project]` metadata (`name = "mnist-pipeline"`, Python version `>=3.10`, dependencies), `[project.optional-dependencies]` (`dev`, `space`), and `[tool.pytest.ini_options]` / `[tool.ruff]`.
  - Core dependencies: `torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`, `hydra-core`, `omegaconf`, `dvc`, `wandb`, `huggingface_hub`, `safetensors`, `pillow`, `jinja2`, `streamlit`, `streamlit-drawable-canvas`.
- **`requirements.txt`**:
  - Provides direct environment installation (`pip install -r requirements.txt` and `-e .`) so both standard virtual environments and containerized builds work out of the box.

### 4.2 Configuration management with Hydra (`configs/`)

- Use `@hydra.main(version_base="1.3", config_path="../configs", config_name="config")` across CLI entry points or Hydra's Compose API when invoked inside deterministic DVC stages.
- `configs/config.yaml` composes default groups:
  - `data: mnist` (`root_dir`, `raw_dir`, `processed_dir`, `val_fraction: 0.1`, `mean: 0.1307`, `std: 0.3081`, `batch_size: 128`, `num_workers`, `pin_memory`)
  - `model: cnn` (`in_channels: 1`, `conv_channels: [32, 64]`, `kernel_size: 3`, `dropout: 0.25`, `hidden_dim: 128`, `num_classes: 10`)
  - `train: default` (`epochs: 10`, `lr: 1e-3`, `weight_decay: 0.0`, `optimizer: adam`, `seed: 42`, `deterministic: true`, `device: auto`, `early_stopping_patience: null`)
  - `evaluate: default` (`batch_size: 256`, `benchmark_warmup_runs: 10`, `benchmark_timed_runs: 100`, `benchmark_batch_sizes: [1, 32, 128]`)
  - `wandb: default` (`enabled: true`, `project: mnist-classification`, `entity: null`, `mode: online`, `tags: ["baseline", "cnn"]`, `log_model: true`)
  - `hf: default` (`repo_id: "username/mnist-cnn"`, `dataset_repo_id: "username/mnist-dataset"`, `revision: "main"`, `private: false`, `cache_dir: "artifacts/hf_cache"`)
- Structured validation in `src/mnist_pipeline/config.py` validates value ranges at startup (e.g., `batch_size > 0`, `0 < val_fraction < 1`, `lr > 0`).

### 4.3 Data and pipeline versioning with DVC (`dvc.yaml`)

Define a 5-stage reproducible pipeline in `dvc.yaml`:

1. **`preprocess`**:
   - Command: `python scripts/preprocess.py`
   - Dependencies: `scripts/preprocess.py`, `src/mnist_pipeline/data.py`, `configs/data/mnist.yaml`
   - Outputs: `data/raw`, `data/processed` (`train.pt`, `val.pt`, `test.pt`, `split_metadata.json`)
2. **`train`**:
   - Command: `python scripts/train.py`
   - Dependencies: `scripts/train.py`, `src/mnist_pipeline/train.py`, `src/mnist_pipeline/model.py`, `data/processed`, `configs/model/cnn.yaml`, `configs/train/default.yaml`
   - Outputs: `artifacts/checkpoints/checkpoint_best.pt`, `artifacts/checkpoints/checkpoint_last.pt`, `artifacts/checkpoints/model.safetensors`, `artifacts/reports/train_history.csv`, `artifacts/reports/environment.json`
3. **`evaluate`**:
   - Command: `python scripts/evaluate.py`
   - Dependencies: `scripts/evaluate.py`, `src/mnist_pipeline/evaluate.py`, `src/mnist_pipeline/metrics.py`, `artifacts/checkpoints/checkpoint_best.pt`, `data/processed`
   - Metrics: `artifacts/evaluation/test_metrics.json` (untracked/cached metric file for `dvc metrics show/diff`)
   - Outputs: `artifacts/evaluation/classification_report.txt`, `artifacts/evaluation/confusion_matrix.csv`
4. **`benchmark`**:
   - Command: `python scripts/benchmark.py`
   - Dependencies: `scripts/benchmark.py`, `src/mnist_pipeline/evaluate.py`, `artifacts/checkpoints/checkpoint_best.pt`, `configs/evaluate/default.yaml`
   - Metrics: `artifacts/benchmarks/benchmark_metrics.json`
5. **`report`**:
   - Command: `python scripts/generate_report.py`
   - Dependencies: `scripts/generate_report.py`, `src/mnist_pipeline/reporting.py`, `artifacts/reports/train_history.csv`, `artifacts/evaluation/test_metrics.json`, `artifacts/evaluation/confusion_matrix.csv`, `artifacts/benchmarks/benchmark_metrics.json`, `templates/model_card_template.md`, `templates/dataset_card_template.md`
   - Plots: `artifacts/reports/plots/loss_curve.png`, `artifacts/reports/plots/accuracy_curve.png`, `artifacts/reports/plots/confusion_matrix.png`, `artifacts/reports/plots/sample_predictions.png`
   - Outputs: `artifacts/reports/evaluation_report.md`, `docs/model_card.md`, `docs/dataset_card.md`

### 4.4 Makefile workflow automation (`Makefile`)

Provide self-documenting targets in `Makefile`:

| Target | Description | Underlying command |
| :--- | :--- | :--- |
| `make help` | List all available Makefile targets | Awk/grep target descriptions |
| `make setup` | Install dependencies and package in editable mode | `pip install -r requirements.txt && pip install -e .` |
| `make data` | Download and preprocess the MNIST dataset | `python scripts/preprocess.py` |
| `make train` | Train the model using Hydra configuration | `python scripts/train.py $(ARGS)` |
| `make evaluate` | Evaluate best checkpoint on held-out test set | `python scripts/evaluate.py $(ARGS)` |
| `make benchmark` | Run latency, throughput, and memory benchmarks | `python scripts/benchmark.py $(ARGS)` |
| `make report` | Generate evaluation reports, plots, and HF cards | `python scripts/generate_report.py $(ARGS)` |
| `make pipeline` | Run the full pipeline via DVC (or sequentially) | `dvc repro` |
| `make hf-upload` | Upload model, model card, dataset card & artifacts to HF Hub | `python scripts/hf_upload.py $(ARGS)` |
| `make hf-inference` | Download model from HF Hub and run inference | `python scripts/hf_inference.py $(ARGS)` |
| `make app-run` | Launch the Streamlit demo locally | `streamlit run app/streamlit_app.py` |
| `make docker-build` | Build the Hugging Face Space Docker image | `docker build -t mnist-space app/` |
| `make docker-run` | Run the Hugging Face Space container on port 7860 | `docker run -p 7860:7860 mnist-space` |
| `make test` | Execute unit and smoke test suite | `pytest tests/` |
| `make clean` | Clean temporary cache and build artifacts | Remove `__pycache__`, `.pytest_cache`, `*.egg-info` |

### 4.5 Experiment tracking with Weights & Biases (`src/mnist_pipeline/tracker.py`)

- Initialize W&B runs in `train.py` and `evaluate.py` using settings from `configs/wandb/default.yaml`.
- Support `wandb.mode` values of `online`, `offline`, and `disabled` (defaulting to `disabled` or `offline` in unit tests/CI so automated tests never require external network credentials).
- Log:
  - Full resolved Hydra config (`OmegaConf.to_container(cfg, resolve=True)`).
  - Epoch-level metrics (`train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `train/epoch_time_sec`, `train/lr`).
  - Final test evaluation metrics (`test/loss`, `test/accuracy`, `test/macro_f1`, `test/weighted_f1`, per-class precision/recall/F1) and benchmark metrics (`benchmark/latency_ms_p50`, `benchmark/latency_ms_p95`, `benchmark/throughput_samples_per_sec`).
  - Rich visual media: loss/accuracy curves, confusion matrix image, and a table/grid of sample test predictions with true labels, predicted labels, and confidence scores.
  - Model checkpoint artifacts when `wandb.log_model=true`.

### 4.6 Hugging Face Hub integration (`src/mnist_pipeline/hf_hub.py`)

- **Model serialization and Hub loading**:
  - Implement `MNISTConvNet` compatible with `huggingface_hub.PyTorchModelHubMixin` (or `safetensors.torch.save_file` + `config.json`) so models can be saved with `model.save_pretrained(...)`, uploaded with `HfApi.upload_folder(...)` / `push_to_hub(...)`, and downloaded via `MNISTConvNet.from_pretrained(repo_id)` or `hf_hub_download(...)`.
- **Upload workflow (`scripts/hf_upload.py`)**:
  - Bundles `model.safetensors` (and `checkpoint_best.pt`), `config.json` (model architecture & normalization parameters), rendered `README.md` (Model Card from `docs/model_card.md`), `test_metrics.json`, `benchmark_metrics.json`, and diagnostic plots (`plots/*.png`), then pushes them to the configured Hugging Face Model Hub repository.
  - Optionally pushes the rendered Dataset Card (`docs/dataset_card.md`) and dataset metadata to a Hugging Face Dataset repository when requested.
- **Download and inference workflow (`scripts/hf_inference.py` and `src/mnist_pipeline/inference.py`)**:
  - Downloads the model and preprocessing configuration directly from the Hugging Face Model Hub given `hf.repo_id` and `hf.revision`.
  - Runs inference on one or more input images (PNG/JPEG paths or MNIST test samples), returning the predicted class (`0-9`) and full 10-class probability distribution.
- **Model Card (`docs/model_card.md` & `templates/model_card_template.md`)**:
  - Includes YAML frontmatter (`library_name: pytorch`, `tags: [mnist, image-classification, pytorch, cnn]`, `datasets: [mnist]`, `metrics: [accuracy, f1]`, `license: apache-2.0`) and documents:
    1. Model summary and architecture diagram/table (layer shapes, parameter counts).
    2. Intended use, limitations, and out-of-scope use cases.
    3. Training methodology, hyperparameters, deterministic split details, and compute environment.
    4. Quantitative evaluation results (test accuracy, macro/weighted precision, recall, F1, per-class metrics, confusion matrix, and inference latency/throughput benchmarks).
    5. How to load and run inference from the Hugging Face Hub with code examples.
- **Dataset Card (`docs/dataset_card.md` & `templates/dataset_card_template.md`)**:
  - Includes YAML frontmatter and documents:
    1. Dataset summary and provenance (LeCun, Cortes, Burges original MNIST database via `torchvision.datasets.MNIST`).
    2. Dataset structure, tensor shapes (`1 × 28 × 28`), pixel value ranges, and class distribution across train (`54,000`), validation (`6,000`), and test (`10,000`) splits.
    3. Preprocessing and normalization (`mean=0.1307`, `std=0.3081`) and deterministic stratified splitting (`seed=42`).
    4. Usage instructions with DVC/Makefile commands and Python loading snippets.
    5. Licensing, citation, and known limitations.

### 4.7 Hugging Face Spaces interactive demo (`app/`)

- **Streamlit application (`app/streamlit_app.py`)**:
  - Downloads and caches (`@st.cache_resource`) the trained `MNISTConvNet` model and `config.json` from the Hugging Face Model Hub (`HF_MODEL_REPO_ID` environment variable or Hydra/default config, with graceful fallback to a local checkpoint if offline).
  - Provides two input modes:
    1. **Draw a Digit**: Interactive 280×280 canvas (`streamlit-drawable-canvas`) with white brush strokes on a black background (matching MNIST foreground/background polarity), center-of-mass/bounding-box cropping or resizing to `28 × 28` grayscale, and standard MNIST normalization.
    2. **Upload an Image**: File uploader supporting PNG/JPG/JPEG images, automatic grayscale conversion, optional polarity inversion toggle (for black ink on white paper), resizing to `28 × 28`, and normalization.
  - Displays:
    - The preprocessed `28 × 28` tensor preview seen by the model.
    - Top predicted digit (`0–9`) and confidence score.
    - Interactive horizontal bar chart of class probabilities across all 10 digits (`0–9`).
    - Model metadata loaded from the Hugging Face Hub repository.
- **Docker packaging (`app/Dockerfile` and `app/README.md`)**:
  - Base image: `python:3.11-slim`.
  - Runs as non-root user (`uid=1000`) per Hugging Face Docker Spaces security requirements, with `HOME=/home/user`.
  - Exposes port `7860` and launches `streamlit run app/streamlit_app.py --server.port=7860 --server.address=0.0.0.0`.
  - `app/README.md` contains the Hugging Face Spaces YAML header (`title`, `emoji`, `colorFrom`, `colorTo`, `sdk: docker`, `app_port: 7860`).

---

## 5. Implementation phases

### Phase 1: Project setup, dependencies, and Hydra configuration

1. Create the directory structure (`src/mnist_pipeline/`, `configs/`, `scripts/`, `templates/`, `app/`, `docs/`, `tests/`).
2. Author `pyproject.toml` and `requirements.txt` with all required dependencies.
3. Implement Hydra YAML configuration files under `configs/` and structured schema validation in `src/mnist_pipeline/config.py`.
4. Implement deterministic seeding, device resolution, and environment metadata capture in `src/mnist_pipeline/utils.py`.

**Acceptance criteria**
- `make setup` installs all dependencies and the `mnist_pipeline` package cleanly.
- Hydra loads and merges configurations, supports CLI overrides (e.g., `train.epochs=5 train.lr=5e-4`), and rejects invalid configuration values.

### Phase 2: Data ingestion, preprocessing, and DVC stage setup

1. Implement `src/mnist_pipeline/data.py` and `scripts/preprocess.py`:
   - Download MNIST via `torchvision.datasets.MNIST`.
   - Perform a seeded stratified split of the 60,000 training examples into `54,000` training and `6,000` validation examples (`seed=42`).
   - Normalize images with `mean=0.1307`, `std=0.3081` and persist processed datasets (`train.pt`, `val.pt`, `test.pt`) and `split_metadata.json` (including class counts and SHA-256 checksums) to `data/processed/`.
   - Provide `DataLoader` builders with deterministic worker seeding (`worker_init_fn` + `torch.Generator`).
2. Author `dvc.yaml` defining the `preprocess`, `train`, `evaluate`, `benchmark`, and `report` stages.

**Expected batch contract**
- Images: `torch.float32` tensor shaped `[N, 1, 28, 28]`.
- Labels: `torch.int64` tensor shaped `[N]`, with values in `[0, 9]`.

**Acceptance criteria**
- `make data` (or `dvc repro preprocess`) downloads MNIST and writes deterministic splits to `data/processed/`.
- Repeating preprocessing with the same seed yields identical split indices and class distributions.
- Test set is strictly isolated from training and validation loaders.

### Phase 3: Model implementation and Hub serialization

1. Implement `MNISTConvNet(nn.Module, PyTorchModelHubMixin)` in `src/mnist_pipeline/model.py`.
2. Accept `[N, 1, 28, 28]` input tensors and return logits shaped `[N, 10]`.
3. Provide helper methods for parameter counting, weight initialization, local checkpoint saving/loading, and Hugging Face Hub `save_pretrained` / `from_pretrained` compatibility.

**Acceptance criteria**
- Forward-pass test verifies output shape `[N, 10]` and finite logits on CPU/CUDA.
- Backward-pass test verifies non-zero, finite gradients across all trainable parameters.
- Model round-trips through `save_pretrained` and `from_pretrained` with identical predictions.

### Phase 4: Training loop and Weights & Biases integration

1. Implement `src/mnist_pipeline/tracker.py` wrapping Weights & Biases (`wandb.init`, `wandb.log`, `wandb.Artifact`, `wandb.Image`, `wandb.finish`).
2. Implement `src/mnist_pipeline/train.py` and `scripts/train.py`:
   - Execute epoch-level training (`model.train()`) and validation (`model.eval()` + `torch.inference_mode()`).
   - Log training/validation loss, accuracy, learning rate, and epoch duration to both `artifacts/reports/train_history.csv` and W&B.
   - Save `checkpoint_last.pt`, `checkpoint_best.pt` (selected strictly by validation accuracy), and Hugging Face Hub-ready `model.safetensors` + `config.json`.
   - Support resuming interrupted training from `checkpoint_last.pt`.

**Acceptance criteria**
- `make train` completes training, writes checkpoints and `train_history.csv`, and logs metrics to W&B (or runs cleanly in `wandb.mode=offline` / `disabled`).
- Resuming from a checkpoint restores model, optimizer, and epoch state accurately.

### Phase 5: Evaluation and performance benchmarking

1. Implement `src/mnist_pipeline/evaluate.py`, `src/mnist_pipeline/metrics.py`, `scripts/evaluate.py`, and `scripts/benchmark.py`:
   - **Evaluation (`make evaluate`)**:
     - Load `checkpoint_best.pt` and evaluate across all 10,000 test examples exactly once.
     - Compute test loss, overall accuracy, per-class precision/recall/F1, macro/weighted averages, and the `10 × 10` confusion matrix.
     - Write `artifacts/evaluation/test_metrics.json`, `classification_report.txt`, and `confusion_matrix.csv`, and log test results to W&B.
   - **Benchmarking (`make benchmark`)**:
     - Measure single-sample (`batch_size=1`) and batched (`batch_size=32, 128`) inference latency (`mean`, `std`, `p50`, `p95`, `p99` in milliseconds) and throughput (`samples/sec`) using `time.perf_counter()` after warmup iterations.
     - Synchronize CUDA before and after timed regions when running on GPU, and record peak memory usage and model size on disk.
     - Write `artifacts/benchmarks/benchmark_metrics.json`.

**Acceptance criteria**
- Exactly 10,000 test examples are evaluated without gradient updates.
- All evaluation and benchmark metrics are finite, reproducible, and written in machine-readable JSON for DVC and reporting.

### Phase 6: Report, Model Card, and Dataset Card generation

1. Implement `src/mnist_pipeline/reporting.py` and `scripts/generate_report.py`:
   - Generate diagnostic plots in `artifacts/reports/plots/`:
     - `loss_curve.png` (train vs. validation loss)
     - `accuracy_curve.png` (train vs. validation accuracy)
     - `confusion_matrix.png` (annotated `10 × 10` heatmap)
     - `sample_predictions.png` (grid of test digits with predicted probabilities)
   - Generate `artifacts/reports/evaluation_report.md` summarizing configuration, dataset splits, validation vs. test metrics, per-class metrics, benchmark results, hardware/software environment, and embedded plot links.
   - Render `docs/model_card.md` and `docs/dataset_card.md` from `templates/model_card_template.md` and `templates/dataset_card_template.md` populated with actual run metrics and configuration values.

**Acceptance criteria**
- `make report` produces all plots, `evaluation_report.md`, `docs/model_card.md`, and `docs/dataset_card.md` without errors.

### Phase 7: Hugging Face Hub upload/download and Streamlit Docker Space

1. Implement `src/mnist_pipeline/hf_hub.py`, `src/mnist_pipeline/inference.py`, `scripts/hf_upload.py`, and `scripts/hf_inference.py`:
   - Support uploading the trained model (`model.safetensors`, `config.json`), rendered Model Card (`README.md`), Dataset Card, and evaluation/benchmark reports to the Hugging Face Hub.
   - Support downloading the trained model from the Hugging Face Model Hub and running CLI or programmatic inference on local image files or test samples.
2. Implement `app/streamlit_app.py`, `app/Dockerfile`, `app/requirements.txt`, and `app/README.md`:
   - Build the interactive Streamlit app supporting freehand digit drawing and image upload, downloading the trained model from the Hugging Face Model Hub, and visualizing predicted digits and class probabilities.

**Acceptance criteria**
- `make hf-inference` loads weights from a Hugging Face repo (or local cache) and outputs predicted class and probabilities.
- `make hf-upload` supports `--dry-run` validation (packaging the staging folder and cards) as well as live upload with `HF_TOKEN`.
- `app/streamlit_app.py` and `app/Dockerfile` pass syntax/health checks and expose the demo on port `7860`.

### Phase 8: Comprehensive user guide and automated testing

1. Author `docs/user_guide.md` covering all 9 required user workflows step by step:
   1. Environment setup and dependency installation (`pyproject.toml`, `requirements.txt`, `make setup`).
   2. Downloading and preprocessing the MNIST dataset (`make data`, `dvc repro preprocess`).
   3. Configuring and running model training using Hydra (`make train`, CLI overrides, experiment presets).
   4. Tracking experiments and metrics with Weights & Biases (`wandb login`, configuration options, offline/online modes, viewing dashboards).
   5. Running evaluation and performance benchmarks (`make evaluate`, `make benchmark`, interpreting metrics).
   6. Generating evaluation reports, including relevant metrics and visualizations (`make report`, viewing plots and cards).
   7. Downloading the trained model from the Hugging Face Model Hub and running inference (`make hf-inference`, Python API).
   8. Uploading the model, model card, dataset card, and associated artifacts to the Hugging Face Hub (`make hf-upload`, `HF_TOKEN` setup).
   9. Building and deploying the Streamlit demo as a Hugging Face Docker Space (`make app-run`, `make docker-build`, `make docker-run`, pushing to `huggingface.co/spaces/...`).
2. Update `README.md` with project overview, architecture summary, quickstart Makefile commands, and links to `docs/user_guide.md`, `docs/model_card.md`, and `docs/dataset_card.md`.
3. Implement unit and integration tests under `tests/`.

---

## 6. Command-line and Makefile interface

All stages can be invoked via `Makefile` targets, `dvc repro`, or direct Python scripts with Hydra overrides:

```bash
# 1. Setup environment
make setup

# 2. Download and preprocess MNIST data
make data

# 3. Train baseline model (with optional Hydra overrides)
make train
make train ARGS="train.epochs=5 train.lr=0.0005 wandb.mode=offline"

# 4. Evaluate best checkpoint on held-out test set
make evaluate

# 5. Run latency and throughput benchmarks
make benchmark

# 6. Generate evaluation report, plots, Model Card, and Dataset Card
make report

# 7. Run entire DVC pipeline end-to-end
make pipeline   # runs: dvc repro

# 8. Upload model, Model Card, Dataset Card, and artifacts to Hugging Face Hub
make hf-upload ARGS="hf.repo_id=username/mnist-cnn"

# 9. Download trained model from Hugging Face Hub and run inference
make hf-inference ARGS="hf.repo_id=username/mnist-cnn +image_path=assets/sample_digit.png"

# 10. Run or containerize the Hugging Face Spaces Streamlit demo
make app-run
make docker-build
make docker-run
```

---

## 7. Testing and quality checks

| Test file | Scope & expected result |
| :--- | :--- |
| `tests/test_config.py` | Hydra configs compose cleanly; invalid parameters raise `ValueError`. |
| `tests/test_data.py` | Preprocessing produces `[N, 1, 28, 28]` tensors, valid labels `[0, 9]`, and identical stratified splits for the same seed. |
| `tests/test_model.py` | Forward pass outputs `[N, 10]` finite logits; backward pass populates finite gradients; parameter count matches specification. |
| `tests/test_checkpoint.py` | Checkpoint and `safetensors` round-trip preserves exact logits in `model.eval()` mode. |
| `tests/test_metrics.py` | Toy predictions produce exact accuracy, precision, recall, F1, and confusion matrix values. |
| `tests/test_inference.py` | Single-image and batch inference return valid class indices (`0–9`) and probabilities summing to `1.0`. |
| `tests/test_reporting.py` | Report generator creates `evaluation_report.md`, plots (`*.png`), `docs/model_card.md`, and `docs/dataset_card.md`. |
| `tests/test_hf_hub.py` | Local staging, `save_pretrained` / `from_pretrained` loading, and dry-run Hub upload succeed without network dependency. |

---

## 8. Reproducibility and experiment integrity

- Seed Python `random`, `numpy`, `torch`, `torch.cuda`, DataLoader workers, and the stratified validation split generator.
- Track raw and processed data plus stage outputs in `dvc.yaml` so any modification to code, configuration, or data triggers only the affected downstream stages.
- Save the resolved Hydra configuration, package versions, Git commit hash (if available), and hardware metadata in `artifacts/reports/environment.json` and W&B run configs.
- Keep the 10,000-example test partition completely isolated from training, early stopping, and hyperparameter selection.
- Use validation accuracy exclusively to select `checkpoint_best.pt`.

---

## 9. Definition of done

The implementation is complete when:

- Both `pyproject.toml` and `requirements.txt` are present and `make setup` installs the environment cleanly.
- `dvc.yaml` defines the end-to-end pipeline stages (`preprocess`, `train`, `evaluate`, `benchmark`, `report`) with consistent dependencies, outputs, metrics, and plots.
- Hydra YAML configs under `configs/` manage all data, model, training, evaluation, W&B, and Hugging Face settings.
- `Makefile` provides working targets for data download/preprocessing, training, evaluation, benchmarking, report generation, HF Hub upload/inference, and Streamlit Space execution.
- W&B integration logs configurations, training/validation metrics, evaluation results, and visual plots (supporting `online`, `offline`, and `disabled` modes).
- Evaluation and benchmarking produce `test_metrics.json`, `benchmark_metrics.json`, `evaluation_report.md`, and diagnostic plots.
- Hugging Face Hub integration includes `docs/model_card.md`, `docs/dataset_card.md`, upload script (`scripts/hf_upload.py`), and download/inference support (`scripts/hf_inference.py`).
- The Hugging Face Docker Space (`app/streamlit_app.py`, `app/Dockerfile`, `app/README.md`) allows users to upload or draw a digit and view predicted class probabilities from the Hub model.
- `docs/user_guide.md` provides a complete, step-by-step guide covering all 9 required topics.
- Automated tests (`make test`) pass.
