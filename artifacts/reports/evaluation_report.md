# MNIST Classification Pipeline — Evaluation & Benchmark Report

## 1. Executive Summary

- **Test Accuracy**: **22.93%** (`0.2293`)
- **Macro F1-Score**: `0.1631`
- **Test Cross-Entropy Loss**: `2.2496`
- **Evaluated Test Samples**: `10,000`
- **Trainable Parameters**: `421,834` (`4.85 MB`)
- **Execution Device**: `cpu`
- **Global Seed**: `42` (`deterministic=True`)

---

## 2. Training Progression Summary

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate | Epoch Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4.0115 | 8.20% | 2.2341 | 23.44% | 0.001000 | 0.11 |

### Training Curves

- Loss Curve: `plots/loss_curve.png`
- Accuracy Curve: `plots/accuracy_curve.png`

---

## 3. Held-Out Test Set Classification Metrics

| Metric | Value |
| :--- | :---: |
| **Overall Accuracy** | `0.2293` (22.93%) |
| **Cross-Entropy Loss** | `2.2496` |
| **Macro Precision** | `0.3225` |
| **Macro Recall** | `0.2292` |
| **Macro F1-Score** | `0.1631` |
| **Weighted Precision** | `0.3381` |
| **Weighted Recall** | `0.2293` |
| **Weighted F1-Score** | `0.1647` |

### Per-Class Precision, Recall & F1-Score

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

## 4. Inference Latency & Throughput Benchmarks

| Batch Size | Mean Latency (ms) | Std (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (samples/sec) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.309 | 0.121 | 0.280 | 0.607 | 0.682 | 3236.2 |
| **32** | 2.695 | 0.231 | 2.693 | 3.041 | 3.231 | 11875.0 |
| **128** | 7.250 | 0.284 | 7.246 | 7.756 | 7.818 | 17656.0 |

---

## 5. Runtime Environment Metadata

- **Python**: `3.12.12`
- **PyTorch**: `2.15.0`
- **Platform**: `macOS-15.2-arm64-arm-64bit`
- **Timestamp (UTC)**: `2026-11-14T02:41:05.027145+00:00`
