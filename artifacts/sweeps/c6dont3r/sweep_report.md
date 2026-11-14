# 🧹 Hyperparameter Sweep Report: `c6dont3r`

> **Generated at**: 2026-11-14 02:47:51 UTC  
> **Project**: `mnist-classification` | **Entity**: `rl4aa` | **Total Trials**: 5  
> **W&B Dashboard**: [https://wandb.ai/rl4aa/mnist-classification/sweeps/c6dont3r](https://wandb.ai/rl4aa/mnist-classification/sweeps/c6dont3r)

---

## 🏆 Champion Model Overview

The best performing model identified during this sweep is trial **`stellar-sweep-5`**.

| Metric / Property | Champion Value |
| :--- | :--- |
| **Run ID** | `lkylhv40` |
| **Run Name** | `stellar-sweep-5` |
| **Best Validation Accuracy** | **`98.7667%`** |
| **Best Validation Loss** | **`0.0399`** |
| **Best Epoch** | `5` of `5` |
| **Learning Rate** | `0.00047354245728168347` |
| **Optimizer** | `adam` |
| **Dropout** | `0.19755322696695937` |
| **Batch Size** | `256` |
| **Weight Decay** | `0.0007675838608259235` |
| **Checkpoint Path** | `artifacts/sweeps/c6dont3r/lkylhv40/checkpoint_best.pt` |

---

## 📊 Sweep Leaderboard

| Rank | Run Name | Run ID | Val Acc | Val Loss | Best Epoch | LR | Optimizer | Dropout | Batch Size | Weight Decay |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 🥇 | `stellar-sweep-5` | `lkylhv40` | **98.77%** | 0.0399 | 5 | 4.74e-04 | adam | 0.198 | 256 | 7.68e-04 |
| 2 🥈 | `copper-sweep-4` | `r2009xy3` | **98.75%** | 0.0461 | 5 | 8.79e-04 | adamw | 0.442 | 128 | 4.65e-06 |
| 3 🥉 | `avid-sweep-1` | `h50ja9pk` | **98.70%** | 0.0443 | 4 | 7.13e-03 | sgd | 0.285 | 128 | 6.02e-04 |
| 4 | `spring-sweep-3` | `o1x4cay3` | **98.62%** | 0.0485 | 5 | 3.32e-03 | adamw | 0.310 | 256 | 3.22e-05 |
| 5 | `peachy-sweep-2` | `us12wo0x` | **98.22%** | 0.0615 | 5 | 2.48e-03 | sgd | 0.421 | 128 | 1.81e-04 |

---

## 📈 Sweep Performance Statistics

| Metric | Minimum | Mean | Maximum |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | 98.22% | 98.61% | 98.77% |
| **Validation Loss** | 0.0399 | 0.0481 | 0.0615 |

---

## 🚀 Promotion & Deployment

To promote this sweep's champion model into canonical checkpoints and update `params.yaml`:

```bash
make sweep-promote SWEEP_ID=c6dont3r
```

To promote the champion model and immediately release it to Hugging Face Hub:

```bash
make sweep-publish-hf SWEEP_ID=c6dont3r
```

