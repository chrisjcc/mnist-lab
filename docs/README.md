# 📚 MNIST Lab — Documentation Hub

Welcome to the documentation hub for **MNIST Lab**, a production-grade, reproducible PyTorch convolutional neural network pipeline integrated with Hydra, DVC, Weights & Biases (W&B), and the Hugging Face Hub.

---

## 🗂️ Documentation Sitemap

| Category | Document | Description |
| :--- | :--- | :--- |
| **User & Operations** | [`user_guide.md`](user_guide.md) | Comprehensive CLI reference, Makefile targets, environment setup, and deployment |
| **Architecture** | [`mnist_pytorch_pipeline_implementation_plan.md`](mnist_pytorch_pipeline_implementation_plan.md) | 5-phase engineering roadmap, directory contracts, and interface definitions |
| **Tutorials** | [`tutorials/pytorch_nn_and_cnn_training_tutorial.md`](tutorials/pytorch_nn_and_cnn_training_tutorial.md) | Educational deep dive into CNN layer mechanics, backpropagation, and training loops |
| **Model Card** | [`model_card.md`](model_card.md) | Hugging Face-compliant model card for `MNISTConvNet` (architecture, performance, licenses) |
| **Dataset Card** | [`dataset_card.md`](dataset_card.md) | Dataset curation, deterministic stratified splits (54k/6k/10k), and preprocessing specs |
| **Baseline Study** | [`experiments/mve_baseline_study.md`](experiments/mve_baseline_study.md) | Minimum Viable Experiment (MVE-001) results & 5/5 quantitative acceptance gates |
| **Sweep Study** | [`experiments/sweep_study.md`](experiments/sweep_study.md) | Bayesian hyperparameter search study (`c6dont3r`), Hyperband early stopping & champion analysis |
| **Research & Feasibility** | [`research/transformer_mnist_feasibility_study.md`](research/transformer_mnist_feasibility_study.md) | Vision Transformer (ViT) vs. CNN comparative feasibility study & integration blueprint |

---

## 🧪 Experiments, Diagnostic Figures & Metrics Catalog

All persistent experiment findings and diagnostic figures are curated under [`docs/experiments/`](experiments/):

### 1. Studies
- **[Minimum Viable Experiment (MVE) Baseline](experiments/mve_baseline_study.md)**: Deterministic 5-epoch baseline evaluation achieving **98.93% test accuracy** and passing all 5 acceptance criteria gates.
- **[Bayesian Hyperparameter Sweep Study](experiments/sweep_study.md)**: 5-trial Bayesian search optimizing learning rate, optimizer (`adam`, `adamw`, `sgd`), dropout, batch size, and weight decay, with champion model **`stellar-sweep-5`** achieving **98.77% validation accuracy**.

### 2. Diagnostic Figures
- **Baseline Figures** ([`docs/experiments/figures/`](experiments/figures/)): Loss curve, accuracy curve, 10x10 confusion matrix, and 4x4 test sample prediction grid.
- **Sweep Champion Figures** ([`docs/experiments/figures/sweep/`](experiments/figures/sweep/)): Diagnostic plots generated from the hyperparameter sweep evaluation run.

### 3. Evaluation & Benchmark Metrics
Stored under [`docs/experiments/metrics/`](experiments/metrics/):
- **`sweep_c6dont3r_trials.json`**: Complete trial ledger recording parameters, validation metrics, epoch runtimes, and W&B URLs.
- **`test_metrics.json`**: Detailed test loss, overall accuracy, macro/weighted precision, recall, and per-class F1-scores.
- **`classification_report.txt`**: Standard scikit-learn classification summary text.
- **`confusion_matrix.csv`**: Full 10x10 digit classification counts.
- **`benchmark_metrics.json`**: CPU inference latency percentiles (P50, P95, P99) and throughput (samples/sec) across batch sizes 1, 32, and 128.

---

## ⚙️ Quick Reference: Report & Artifact Generation

The reports, plots, and cards in this directory are generated directly via automated pipeline targets:

```bash
# 1. Generate local diagnostic figures, evaluation_report.md, and update model & dataset cards
make report

# 2. Compile or refresh the local Markdown hyperparameter sweep report from trial ledgers
make sweep-report SWEEP_ID=c6dont3r

# 3. Promote champion model locally (updates checkpoint, params.yaml, test eval & model card)
make sweep-promote SWEEP_ID=c6dont3r

# 4. Promote champion model and publish directly to Hugging Face Hub
make sweep-publish-hf SWEEP_ID=c6dont3r
```

> 📖 **Full Instructions**: For in-depth CLI options, output schemas, and pipeline DAG reproduction, see the [User Guide — Section 6: Generating Evaluation Reports & Visualizations](user_guide.md#6-generating-evaluation-reports-and-visualizations).

---

## 📦 Storage Separation Policy (MLOps Best Practice)

To keep this Git repository fast to clone and lightweight:
- **Git Tracks**: Code (`src/`, `scripts/`, `tests/`, `app/`), configurations (`configs/`, `sweeps/`), documentation, Markdown reports, diagnostic `.png` figures, and small summary JSONs in `docs/`.
- **Model Checkpoints**: Trained weights (`*.pt`, `*.safetensors`) and local trial runs in `/artifacts/` are ignored by Git ([`.gitignore`](../.gitignore)) and distributed via the **[Hugging Face Model Hub](https://huggingface.co/chrisjcc/mnist-classification)**, **[GitHub Releases](user_guide.md#10-attaching-trained-model-checkpoints-to-github-releases-offline-distribution)**, and **Weights & Biases Model Registry**.
