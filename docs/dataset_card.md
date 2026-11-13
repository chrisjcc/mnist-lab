---
language: en
license: cc-by-sa-3.0
task_categories:
  - image-classification
pretty_name: MNIST Handwritten Digits (Stratified DVC Split)
size_categories:
  - 10K<n<100K
tags:
  - mnist
  - image-classification
  - computer-vision
  - dvc
  - pytorch
---

# Dataset Card: `chrisjcc/mnist-dataset`

## 1. Dataset Summary & Provenance

The **MNIST (Modified National Institute of Standards and Technology)** database of handwritten digits (LeCun, Cortes, & Burges, 1998) is a canonical computer vision benchmark consisting of $28 \times 28$ 8-bit grayscale images of handwritten digits (`0` through `9`).

- **Original Authors**: Yann LeCun (Courant Institute, NYU), Corinna Cortes (Google Labs, NY), Christopher J.C. Burges (Microsoft Research, Redmond)
- **Upstream Source**: `torchvision.datasets.MNIST` / `ylecun/mnist`
- **Data Source Used in Run**: `torchvision.datasets.MNIST`
- **Deterministic Split Seed**: `42` (`val_fraction = 0.1`)

---

## 2. Dataset Splits & Preprocessing Pipeline

To prevent test-set leakage during hyperparameter selection and early stopping, the upstream 60,000-sample training set is deterministically partitioned using `stratified_train_val_split(val_fraction=0.1, seed=42)` into:

1. **Train Split** (`data/processed/train.pt`): `54,000` samples (`sha256: e166767e5a42177b...`)
2. **Validation Split** (`data/processed/val.pt`): `6,000` samples (`sha256: 5631ffe885229a98...`)
3. **Held-Out Test Split** (`data/processed/test.pt`): `10,000` samples (`sha256: 757b3cee157a4aa2...`)

### Normalization & Tensor Contract

1. **Pixel Scaling**: Raw `uint8` pixels in `[0, 255]` are scaled to `float32` in `[0.0, 1.0]` by dividing by `255.0`.
2. **Z-Score Standardization**: Standardized using the training-split statistics:
   - Mean: $\mu = 0.1307$
   - Standard Deviation: $\sigma = 0.3081$
   - Formula: $x_{\text{norm}} = \frac{x / 255.0 - 0.1307}{ 0.3081 }$
3. **Tensor Shape**: `(N, 1, 28, 28)` (`torch.float32`), paired with labels `(N,)` (`torch.int64` in `[0, 9]`).

---

## 3. Stratified Class Distribution Across Splits

| Digit Class | Train Samples | Validation Samples | Test Samples | Total |
| :---: | :---: | :---: | :---: | :---: |
| **Digit 0** | 5331 | 592 | 980 | 6903 |
| **Digit 1** | 6067 | 675 | 1135 | 7877 |
| **Digit 2** | 5362 | 596 | 1032 | 6990 |
| **Digit 3** | 5518 | 613 | 1010 | 7141 |
| **Digit 4** | 5258 | 584 | 982 | 6824 |
| **Digit 5** | 4879 | 542 | 892 | 6313 |
| **Digit 6** | 5326 | 592 | 958 | 6876 |
| **Digit 7** | 5639 | 626 | 1028 | 7293 |
| **Digit 8** | 5266 | 585 | 974 | 6825 |
| **Digit 9** | 5354 | 595 | 1009 | 6958 |
| **Total** | **54,000** | **6,000** | **10,000** | **70,000** |

---

## 4. Reproducing the Preprocessed Splits via DVC / Makefile

```bash
# Run via Makefile
make data

# Or run the Hydra CLI entrypoint directly
python scripts/preprocess.py data.seed=42 data.val_fraction=0.1

# Or reproduce the DVC stage
dvc repro preprocess
```