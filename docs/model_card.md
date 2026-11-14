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
  - chrisjcc/mnist-dataset
metrics:
  - accuracy
  - f1
  - precision
  - recall
model-index:
  - name: chrisjcc/mnist-cnn
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
            value: 0.2293
            name: Test Accuracy
          - type: f1
            value: 0.1631
            name: Macro F1
          - type: precision
            value: 0.3225
            name: Macro Precision
          - type: recall
            value: 0.2292
            name: Macro Recall
---

# Model Card: `chrisjcc/mnist-cnn`

## 1. Model Details

- **Model Architecture**: `MNISTConvNet` (2-Block BatchNorm Convolutional Neural Network with Dropout Classifier Head)
- **Framework**: PyTorch (`2.15.0`) + `safetensors`
- **Trainable Parameters**: `421,834` (`4.85 MB` full checkpoint / `1.61 MB` SafeTensors weights)
- **Input Contract**: `(B, 1, 28, 28)` float32 tensor normalized with $\mu = 0.1307$, $\sigma = 0.3081$
- **Output Contract**: `(B, 10)` unnormalized class logits (`0`–`9`)
- **Repository & Pipeline**: Reproducible DVC + Hydra + Weights & Biases pipeline (`seed = 42`)
- **License**: MIT

### Layer-by-Layer Architecture

| Stage | Layer Composition | Output Shape |
| :--- | :--- | :--- |
| **Input** | Normalized Grayscale Digit Image | `(B, 1, 28, 28)` |
| **Conv Block 1** | `Conv2d(1 -> 32, 3x3, pad=1)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2x2)` | `(B, 32, 14, 14)` |
| **Conv Block 2** | `Conv2d(32 -> 64, 3x3, pad=1)` -> `BatchNorm2d` -> `ReLU` -> `MaxPool2d(2x2)` | `(B, 64, 7, 7)` |
| **Classifier Head** | `Flatten` -> `Dropout(p=0.25)` -> `Linear(3136 -> 128)` -> `ReLU` -> `Dropout(p=0.25)` -> `Linear(128 -> 10)` | `(B, 10)` |

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
| **Global Seed** | `42` (`deterministic = True`) |
| **Optimizer** | `adam` (`lr = 0.001`, `weight_decay = 0.0`) |
| **Epochs** | `5` (early stopping patience = `None`) |
| **Batch Size** | `128` (train) / `256` (eval) |
| **Hardware Device** | `cpu` |

---

## 4. Evaluation Results (Held-Out Test Set, $N = 10000$)

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **0.2293 (22.93%)** |
| **Test Cross-Entropy Loss** | `2.2496` |
| **Macro Precision** | `0.3225` |
| **Macro Recall** | `0.2292` |
| **Macro F1-Score** | `0.1631` |
| **Weighted F1-Score** | `0.1647` |

### Per-Class Performance (`0`–`9`)

| Digit Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **Digit 0** | 0.0000 | 0.0000 | 0.0000 | 980 |
| **Digit 1** | 1.0000 | 0.0696 | 0.1301 | 1135 |
| **Digit 2** | 0.7778 | 0.0339 | 0.0650 | 1032 |
| **Digit 3** | 0.5785 | 0.4485 | 0.5053 | 1010 |
| **Digit 4** | 0.2157 | 0.0336 | 0.0582 | 982 |
| **Digit 5** | 0.1365 | 0.3890 | 0.2021 | 892 |
| **Digit 6** | 0.0000 | 0.0000 | 0.0000 | 958 |
| **Digit 7** | 0.3220 | 0.8677 | 0.4697 | 1028 |
| **Digit 8** | 0.0690 | 0.0021 | 0.0040 | 974 |
| **Digit 9** | 0.1256 | 0.4480 | 0.1962 | 1009 |


---

## 5. Inference Latency & Throughput Benchmarks

Measured over `50` timed forward passes following `10` warm-up runs on device `cpu`:

| Batch Size | Mean Latency (ms) | Std (ms) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (samples/sec) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.309 | 0.121 | 0.280 | 0.607 | 0.682 | 3236.2 |
| **32** | 2.695 | 0.231 | 2.693 | 3.041 | 3.231 | 11875.0 |
| **128** | 7.250 | 0.284 | 7.246 | 7.756 | 7.818 | 17656.0 |


---

## 6. How to Load & Run Inference

```python
from PIL import Image
from mnist_pipeline.inference import load_inference_model, predict_digit, preprocess_pil_image

# Download from Hugging Face Hub (or load from local fallback checkpoint)
model = load_inference_model("chrisjcc/mnist-cnn", device="cpu")

# Load any handwritten digit image
image = Image.open("digit_sample.png")
tensor = preprocess_pil_image(image, invert=False)
result = predict_digit(model, tensor, device="cpu")

print(f"Predicted Digit: {result.predicted_digit} (Confidence: {result.confidence:.2%})")
```