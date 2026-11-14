# 🧹 Hyperparameter Sweep Report: `unit_test_sweep_abc`

> **Generated at**: 2026-11-14 03:14:52 UTC  
> **Project**: `mnist-classification` | **Entity**: `rl4aa` | **Total Trials**: 2  
> **W&B Dashboard**: [https://wandb.ai/rl4aa/mnist-classification/sweeps/unit_test_sweep_abc](https://wandb.ai/rl4aa/mnist-classification/sweeps/unit_test_sweep_abc)

---

## 🏆 Champion Model Overview

The best performing model identified during this sweep is trial **`run_alpha`**.

| Metric / Property | Champion Value |
| :--- | :--- |
| **Run ID** | `run_alpha` |
| **Run Name** | `run_alpha` |
| **Best Validation Accuracy** | **`23.4375%`** |
| **Best Validation Loss** | **`2.2341`** |
| **Best Epoch** | `1` of `1` |
| **Learning Rate** | `0.001` |
| **Optimizer** | `adam` |
| **Dropout** | `0.2` |
| **Batch Size** | `N/A` |
| **Weight Decay** | `N/A` |
| **Checkpoint Path** | `artifacts/sweeps/unit_test_sweep_abc/run_alpha/checkpoint_best.pt` |

---

## 📊 Sweep Leaderboard

| Rank | Run Name | Run ID | Val Acc | Val Loss | Best Epoch | LR | Optimizer | Dropout | Batch Size | Weight Decay |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 🥇 | `run_alpha` | `run_alpha` | **23.44%** | 2.2341 | 1 | 1.00e-03 | adam | 0.200 | N/A | N/A |
| 2 🥈 | `run_beta` | `run_beta` | **15.23%** | 2.2636 | 1 | 5.00e-03 | sgd | 0.350 | N/A | N/A |

---

## 📈 Sweep Performance Statistics

| Metric | Minimum | Mean | Maximum |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 15.23% | 19.34% | 23.44% |
| **Validation Loss** | 2.2341 | 2.2488 | 2.2636 |

---

## 🚀 Promotion & Deployment

To promote this sweep's champion model into canonical checkpoints and update `params.yaml`:

```bash
make sweep-promote SWEEP_ID=unit_test_sweep_abc
```

To promote the champion model and immediately release it to Hugging Face Hub:

```bash
make sweep-publish-hf SWEEP_ID=unit_test_sweep_abc
```

