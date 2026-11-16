# Systematic Model Comparison: Convolutional Neural Network vs. Vision Transformer

> **Project**: `mnist-lab` | **Study**: Architecture Comparison (`MNISTConvNet` vs. `MNISTViT`)

---

## 1. Executive Summary & Comparison Verdict

This report presents a head-to-head empirical comparison between **CNN (MNISTConvNet)** and **Transformer (MNISTViT)** trained and evaluated under identical, isolated experimental protocols on the MNIST benchmark dataset.

- **Accuracy Delta**: **Transformer (MNISTViT)** achieves `95.65%` vs. **CNN (MNISTConvNet)** `98.93%` (difference: `-3.28%`).
- **Inference Efficiency**: **CNN (MNISTConvNet)** is `1.1x` faster at single-sample P95 latency (`0.94 ms` vs. `0.99 ms`).
- **Parameter Footprint**: **CNN (MNISTConvNet)** has `421,834` trainable parameters vs. `540,170` for **Transformer (MNISTViT)**.

---

## 2. Comprehensive Metrics Comparison Table

| Performance Metric | **CNN (MNISTConvNet)** | **Transformer (MNISTViT)** | Delta / Comparison |
| :--- | :---: | :---: | :---: |
| **Test Accuracy (%)** | `98.93%` | `95.65%` | `-3.28%` |
| **Test Loss (Cross-Entropy)** | `0.0315` | `0.1371` | `+0.1056` |
| **Macro F1-Score** | `0.9892` | `0.9560` | `-0.0333` |
| **Trainable Parameters** | `421,834` | `540,170` | `+118,336` |
| **Checkpoint Size (.pt)** | `4.85 MB` | `6.25 MB` | `+1.40 MB` |
| **SafeTensors Size** | `1.61 MB` | `2.07 MB` | `+0.45 MB` |
| **Latency P50 (ms, B=1)** | `0.371 ms` | `0.852 ms` | `+0.481 ms` |
| **Latency P95 (ms, B=1)** | `0.944 ms` | `0.992 ms` | `+0.048 ms` |
| **Throughput (samples/s, B=128)** | `12,084.8` | `3,401.2` | `-8,683.7` |

---

## 3. Latency & Throughput Benchmark Scaling

| Batch Size | Metric | **CNN (MNISTConvNet)** | **Transformer (MNISTViT)** |
| :---: | :--- | :---: | :---: |
| **1** | P50 Latency | `0.371 ms` | `0.852 ms` |
| **1** | P95 Latency | `0.944 ms` | `0.992 ms` |
| **1** | Throughput | `2,171.7 sps` | `1,144.6 sps` |
| **32** | P50 Latency | `3.685 ms` | `9.371 ms` |
| **32** | P95 Latency | `8.296 ms` | `9.945 ms` |
| **32** | Throughput | `7,307.8 sps` | `3,389.3 sps` |
| **128** | P50 Latency | `10.135 ms` | `37.202 ms` |
| **128** | P95 Latency | `12.640 ms` | `40.810 ms` |
| **128** | Throughput | `12,084.8 sps` | `3,401.2 sps` |

---

## 4. Per-Digit Classification F1-Score Breakdown

| Digit Class | **CNN (MNISTConvNet)** | **Transformer (MNISTViT)** | Difference |
| :---: | :---: | :---: | :---: |
| **Digit 0** | `0.9903` | `0.9772` | `-0.0132` |
| **Digit 1** | `0.9938` | `0.9864` | `-0.0074` |
| **Digit 2** | `0.9860` | `0.9482` | `-0.0379` |
| **Digit 3** | `0.9920` | `0.9504` | `-0.0416` |
| **Digit 4** | `0.9944` | `0.9494` | `-0.0450` |
| **Digit 5** | `0.9888` | `0.9454` | `-0.0435` |
| **Digit 6** | `0.9875` | `0.9628` | `-0.0247` |
| **Digit 7** | `0.9859` | `0.9528` | `-0.0330` |
| **Digit 8** | `0.9866` | `0.9451` | `-0.0416` |
| **Digit 9** | `0.9870` | `0.9423` | `-0.0447` |

---

## 5. Visualizations & Comparative Analysis

### 5.1 Accuracy vs. Latency Pareto Frontier
![Pareto Frontier](plots/accuracy_vs_latency_pareto.png)

### 5.2 Per-Digit F1-Score Comparison
![Per Class F1](plots/per_class_f1_comparison.png)

### 5.3 Batch Latency & Throughput Scaling
![Scaling Curves](plots/latency_throughput_scaling.png)

---

## 6. Architectural Analysis & Theoretical Findings

### 6.1 Inductive Bias vs. Self-Attention Capacity
1. **Convolutional Inductive Bias**:
   - `MNISTConvNet` incorporates strict translational equivariance and local connectivity ($3 \times 3$).
   - Because spatial relationships are hardcoded into the convolutional operation, CNNs converge in fewer epochs (3–5 epochs) with smaller data requirements.

2. **Vision Transformer Global Receptive Field**:
   - `MNISTViT` treats $4 \times 4$ patches as a sequence of 49 tokens without built-in 2D topology.
   - Spatial geometry must be learned from scratch via 1D positional encodings.
   - Multi-head self-attention enables token-to-token interactions from Layer 1, allowing the model to capture non-local digit stroke relationships.

### 6.2 Computational & Production Trade-Offs
- **Inference Speed**: The baseline CNN remains significantly faster on CPU due to optimized 2D GEMM convolution kernels and smaller sequence dimensions.
- **Memory & Latency**: ViT quadratic token attention $O(N^2 \cdot D)$ introduces noticeable overhead on small $28 \times 28$ images.
- **Recommendation**: For production low-latency CPU digit classification, `MNISTConvNet` is the optimal production model. `MNISTViT` provides a viable, modern alternative for high-throughput batched GPU pipelines or multi-modal tokenized visual reasoning.
