# 🧹 Hyperparameter Sweep Report: `smoke_sweep`

> **Generated at**: 2026-11-14 03:14:49 UTC  
> **Project**: `mnist-classification` | **Entity**: `rl4aa` | **Total Trials**: 6  
> **W&B Dashboard**: [https://wandb.ai/rl4aa/mnist-classification/sweeps/smoke_sweep](https://wandb.ai/rl4aa/mnist-classification/sweeps/smoke_sweep)

---

## 🏆 Champion Model Overview

The best performing model identified during this sweep is trial **`trial_622381`**.

| Metric / Property | Champion Value |
| :--- | :--- |
| **Run ID** | `trial_622381` |
| **Run Name** | `trial_622381` |
| **Best Validation Accuracy** | **`29.8828%`** |
| **Best Validation Loss** | **`2.9172`** |
| **Best Epoch** | `1` of `1` |
| **Learning Rate** | `0.003` |
| **Optimizer** | `N/A` |
| **Dropout** | `0.3` |
| **Batch Size** | `N/A` |
| **Weight Decay** | `N/A` |
| **Checkpoint Path** | `artifacts/sweeps/smoke_sweep/trial_622381/checkpoint_best.pt` |

---

## 📊 Sweep Leaderboard

| Rank | Run Name | Run ID | Val Acc | Val Loss | Best Epoch | LR | Optimizer | Dropout | Batch Size | Weight Decay |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 🥇 | `trial_622381` | `trial_622381` | **29.88%** | 2.9172 | 1 | 3.00e-03 | N/A | 0.300 | N/A | N/A |
| 2 🥈 | `trial_712940` | `trial_712940` | **29.88%** | 2.9172 | 1 | 3.00e-03 | N/A | 0.300 | N/A | N/A |
| 3 🥉 | `trial_417520` | `trial_417520` | **29.88%** | 2.9172 | 1 | 3.00e-03 | N/A | 0.300 | N/A | N/A |
| 4 | `trial_62799` | `trial_62799` | **29.88%** | 2.9172 | 1 | 3.00e-03 | N/A | 0.300 | N/A | N/A |
| 5 | `trial_89784` | `trial_89784` | **29.88%** | 2.9172 | 1 | 3.00e-03 | N/A | 0.300 | N/A | N/A |
| 6 | `trial_28958` | `trial_28958` | **29.10%** | 2.6401 | 1 | 3.00e-03 | N/A | 0.300 | N/A | N/A |

---

## 📈 Sweep Performance Statistics

| Metric | Minimum | Mean | Maximum |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 29.10% | 29.75% | 29.88% |
| **Validation Loss** | 2.6401 | 2.8710 | 2.9172 |

---

## 🚀 Promotion & Deployment

To promote this sweep's champion model into canonical checkpoints and update `params.yaml`:

```bash
make sweep-promote SWEEP_ID=smoke_sweep
```

To promote the champion model and immediately release it to Hugging Face Hub:

```bash
make sweep-publish-hf SWEEP_ID=smoke_sweep
```

