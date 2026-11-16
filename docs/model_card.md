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
            value: 0.9874
            name: Test Accuracy
          - type: f1
            value: 0.9873
            name: Macro F1
          - type: precision
            value: 0.9873
            name: Macro Precision
          - type: recall
            value: 0.9873
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

> [!NOTE]
> Sibling Vision Transformer (ViT) model card: [`chrisjcc/mnist-vit`](https://huggingface.co/chrisjcc/mnist-vit). Both models are available for comparative inference in the [Streamlit demo space](https://huggingface.co/spaces/chrisjcc/mnist-lab).
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
| **Test Accuracy** | **0.9874 (98.74%)** |
| **Test Cross-Entropy Loss** | `0.0348` |
| **Macro Precision** | `0.9873` |
| **Macro Recall** | `0.9873` |
| **Macro F1-Score** | `0.9873` |
| **Weighted F1-Score** | `0.9874` |

### Per-Class Performance (`0`–`9`)

| Digit Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **Digit 0** | 0.9919 | 0.9949 | 0.9934 | 980 |
| **Digit 1** | 0.9886 | 0.9965 | 0.9925 | 1135 |
| **Digit 2** | 0.9864 | 0.9864 | 0.9864 | 1032 |
| **Digit 3** | 0.9911 | 0.9921 | 0.9916 | 1010 |
| **Digit 4** | 0.9839 | 0.9939 | 0.9889 | 982 |
| **Digit 5** | 0.9800 | 0.9910 | 0.9855 | 892 |
| **Digit 6** | 0.9905 | 0.9802 | 0.9853 | 958 |
| **Digit 7** | 0.9873 | 0.9815 | 0.9844 | 1028 |
| **Digit 8** | 0.9867 | 0.9867 | 0.9867 | 974 |
| **Digit 9** | 0.9869 | 0.9703 | 0.9785 | 1009 |


---

## 5. Inference Latency & Throughput Benchmarks

Measured over `50` timed forward passes following `10` warm-up runs on device `cpu`:

| Batch Size | Mean Latency (ms) | Std (ms) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Throughput (samples/sec) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.171 | 0.011 | 0.167 | 0.188 | 0.211 | 5854.9 |
| **32** | 2.261 | 0.144 | 2.234 | 2.373 | 2.925 | 14151.6 |
| **128** | 7.283 | 1.165 | 6.794 | 9.078 | 11.678 | 17576.3 |


---

## 6. How to Load & Run Inference

```python
from PIL import Image
from mnist_pipeline.inference import load_inference_model, predict_pil_image

# Download from Hugging Face Hub (or load from local fallback checkpoint)
model, source = load_inference_model("chrisjcc/mnist-cnn", device="cpu")

# Load any handwritten digit image
image = Image.open("digit_sample.png")
result = predict_pil_image(model, image, device="cpu")

print(f"Predicted Digit: {result.predicted_digit} (Confidence: {result.confidence:.2%})")
```