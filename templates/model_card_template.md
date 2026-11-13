---
language: en
license: mit
library_name: pytorch
tags:
  - pytorch
  - mnist
  - image-classification
  - computer-vision
  - dvc
  - hydra
  - reproducible-ml
datasets:
  - ylecun/mnist
  - {{ hf_dataset_repo_id }}
metrics:
  - accuracy
  - f1
  - precision
  - recall
model-index:
  - name: {{ hf_model_repo_id }}
    results:
      - task:
          type: image-classification
          name: Handwritten Digit Classification
        dataset:
          name: MNIST
          type: ylecun/mnist
          split: test
        metrics:
          - type: accuracy
            value: {{ "%.4f"|format(test_metrics.test_acc) }}
            name: Test Accuracy
          - type: f1
            value: {{ "%.4f"|format(test_metrics.macro_f1) }}
            name: Macro F1
          - type: precision
            value: {{ "%.4f"|format(test_metrics.macro_precision) }}
            name: Macro Precision
          - type: recall
            value: {{ "%.4f"|format(test_metrics.macro_recall) }}
            name: Macro Recall
---

# Model Card: `{{ hf_model_repo_id }}`

## 1. Model Details

- **Model Architecture**: `MNISTConvNet` (2-Block BatchNorm Convolutional Neural Network with Dropout Classifier Head)
- **Framework**: PyTorch (`{{ env_meta.torch_version | default("2.x") }}`) + `safetensors`
- **Trainable Parameters**: `{{ "{:,}".format(benchmark_metrics.trainable_params) }}` (`{{ "%.2f"|format(benchmark_metrics.checkpoint_size_mb) }} MB` full checkpoint / `{{ "%.2f"|format(benchmark_metrics.safetensors_size_mb or benchmark_metrics.checkpoint_size_mb) }} MB` SafeTensors weights)
- **Input Contract**: `(B, {{ model_cfg.in_channels }}, 28, 28)` float32 tensor normalized with $\mu = {{ data_cfg.mean }}$, $\sigma = {{ data_cfg.std }}$
- **Output Contract**: `(B, {{ model_cfg.num_classes }})` unnormalized class logits (`0`–`9`)
- **Repository & Pipeline**: Reproducible DVC + Hydra + Weights & Biases pipeline (`seed = {{ seed }}`)
- **License**: MIT

### Layer-by-Layer Architecture

| Stage | Layer Composition | Output Shape |
| :--- | :--- | :--- |
| **Input** | Normalized Grayscale Digit Image | `(B, 1, 28, 28)` |
| **Conv Block 1** | `Conv2d(1 -> {{ model_cfg.conv_channels[0] }}, 3x3, pad=1)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2x2)` | `(B, {{ model_cfg.conv_channels[0] }}, 14, 14)` |
| **Conv Block 2** | `Conv2d({{ model_cfg.conv_channels[0] }} -> {{ model_cfg.conv_channels[1] }}, 3x3, pad=1)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2x2)` | `(B, {{ model_cfg.conv_channels[1] }}, 7, 7)` |
| **Classifier Head** | `Flatten` -> `Dropout(p={{ model_cfg.dropout }})` -> `Linear({{ model_cfg.conv_channels[1] * 49 }} -> {{ model_cfg.hidden_dim }})` -> `ReLU` -> `Dropout(p={{ model_cfg.dropout }})` -> `Linear({{ model_cfg.hidden_dim }} -> {{ model_cfg.num_classes }})` | `(B, {{ model_cfg.num_classes }})` |

---

## 2. Intended Use & Limitations

### Intended Use
- Handwritten single-digit classification (`0`–`9`) on $28 \times 28$ grayscale images.
- Reference implementation for reproducible MLOps pipelines combining **Hydra**, **DVC**, **Weights & Biases**, **Hugging Face Hub**, and **Hugging Face Docker Spaces**.
- Interactive educational demonstrations via the bundled Streamlit application.

### Out-of-Scope Use & Limitations
- Not designed for multi-digit OCR, alphanumeric text recognition, or natural scene text detection (SVHN/ICDAR).
- Sensitive to foreground/background polarity: MNIST digits are **white strokes on a black background**. Real-world photos of dark ink on white paper require polarity inversion (`invert=True` in `preprocess_pil_image`).

---

## 3. Training Methodology & Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Dataset** | `ylecun/mnist` (54,000 train / 6,000 stratified val / 10,000 held-out test) |
| **Global Seed** | `{{ seed }}` (`deterministic = {{ deterministic }}`) |
| **Optimizer** | `{{ train_cfg.optimizer }}` (`lr = {{ train_cfg.lr }}`, `weight_decay = {{ train_cfg.weight_decay }}`) |
| **Epochs** | `{{ train_cfg.epochs }}` (early stopping patience = `{{ train_cfg.early_stopping_patience }}`) |
| **Batch Size** | `{{ data_cfg.batch_size }}` (train) / `{{ evaluate_cfg.batch_size }}` (eval) |
| **Hardware Device** | `{{ benchmark_metrics.device }}` |

---

## 4. Evaluation Results (Held-Out Test Set, $N = {{ test_metrics.num_samples }}$)

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **{{ "%.4f"|format(test_metrics.test_acc) }} ({{ "%.2f"|format(test_metrics.test_acc * 100) }}%)** |
| **Test Cross-Entropy Loss** | `{{ "%.4f"|format(test_metrics.test_loss) }}` |
| **Macro Precision** | `{{ "%.4f"|format(test_metrics.macro_precision) }}` |
| **Macro Recall** | `{{ "%.4f"|format(test_metrics.macro_recall) }}` |
| **Macro F1-Score** | `{{ "%.4f"|format(test_metrics.macro_f1) }}` |
| **Weighted F1-Score** | `{{ "%.4f"|format(test_metrics.weighted_f1) }}` |

### Per-Class Performance (`0`–`9`)

| Digit Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
{% for digit, stats in test_metrics.per_class.items() -%}
| **Digit {{ digit }}** | {{ "%.4f"|format(stats.precision) }} | {{ "%.4f"|format(stats.recall) }} | {{ "%.4f"|format(stats.f1) }} | {{ stats.support }} |
{% endfor %}

---

## 5. Inference Latency & Throughput Benchmarks

Measured over `{{ benchmark_metrics.timed_runs }}` timed forward passes following `{{ benchmark_metrics.warmup_runs }}` warm-up runs on device `{{ benchmark_metrics.device }}`:

| Batch Size | Mean Latency (ms) | Std (ms) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (samples/sec) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{% for bs, bm in benchmark_metrics.batch_metrics.items() -%}
| **{{ bs }}** | {{ "%.3f"|format(bm.mean_ms) }} | {{ "%.3f"|format(bm.std_ms) }} | {{ "%.3f"|format(bm.p50_ms) }} | {{ "%.3f"|format(bm.p95_ms) }} | {{ "%.3f"|format(bm.p99_ms) }} | {{ "%.1f"|format(bm.samples_per_sec) }} |
{% endfor %}

---

## 6. How to Load & Run Inference

```python
from PIL import Image
from mnist_pipeline.inference import load_inference_model, predict_digit, preprocess_pil_image

# Download from Hugging Face Hub (or load from local fallback checkpoint)
model = load_inference_model("{{ hf_model_repo_id }}", device="cpu")

# Load any handwritten digit image
image = Image.open("digit_sample.png")
tensor = preprocess_pil_image(image, invert=False)
result = predict_digit(model, tensor, device="cpu")

print(f"Predicted Digit: {result.predicted_digit} (Confidence: {result.confidence:.2%})")
```
