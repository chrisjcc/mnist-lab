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

# Dataset Card: `{{ hf_dataset_repo_id }}`

## 1. Dataset Summary & Provenance

The **MNIST (Modified National Institute of Standards and Technology)** database of handwritten digits (LeCun, Cortes, & Burges, 1998) is a canonical computer vision benchmark consisting of $28 \times 28$ 8-bit grayscale images of handwritten digits (`0` through `9`).

- **Original Authors**: Yann LeCun (Courant Institute, NYU), Corinna Cortes (Google Labs, NY), Christopher J.C. Burges (Microsoft Research, Redmond)
- **Upstream Source**: `torchvision.datasets.MNIST` / `ylecun/mnist`
- **Data Source Used in Run**: `{{ split_meta.source }}`
- **Deterministic Split Seed**: `{{ split_meta.seed }}` (`val_fraction = {{ split_meta.val_fraction }}`)

---

## 2. Dataset Splits & Preprocessing Pipeline

To prevent test-set leakage during hyperparameter selection and early stopping, the upstream 60,000-sample training set is deterministically partitioned using `stratified_train_val_split(val_fraction={{ split_meta.val_fraction }}, seed={{ split_meta.seed }})` into:

1. **Train Split** (`data/processed/train.pt`): `{{ "{:,}".format(split_meta.splits.train.num_examples) }}` samples (`sha256: {{ split_meta.splits.train.images_sha256[:16] }}...`)
2. **Validation Split** (`data/processed/val.pt`): `{{ "{:,}".format(split_meta.splits.val.num_examples) }}` samples (`sha256: {{ split_meta.splits.val.images_sha256[:16] }}...`)
3. **Held-Out Test Split** (`data/processed/test.pt`): `{{ "{:,}".format(split_meta.splits.test.num_examples) }}` samples (`sha256: {{ split_meta.splits.test.images_sha256[:16] }}...`)

### Normalization & Tensor Contract

1. **Pixel Scaling**: Raw `uint8` pixels in `[0, 255]` are scaled to `float32` in `[0.0, 1.0]` by dividing by `255.0`.
2. **Z-Score Standardization**: Standardized using the training-split statistics:
   - Mean: $\mu = {{ split_meta.normalization.mean }}$
   - Standard Deviation: $\sigma = {{ split_meta.normalization.std }}$
   - Formula: $x_{\text{norm}} = \frac{x / 255.0 - {{ split_meta.normalization.mean }}}{ {{ split_meta.normalization.std }} }$
3. **Tensor Shape**: `(N, 1, 28, 28)` (`torch.float32`), paired with labels `(N,)` (`torch.int64` in `[0, 9]`).

---

## 3. Stratified Class Distribution Across Splits

| Digit Class | Train Samples | Validation Samples | Test Samples | Total |
| :---: | :---: | :---: | :---: | :---: |
{% for cls_id in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] -%}
| **Digit {{ cls_id }}** | {{ split_meta.splits.train.class_distribution[cls_id] }} | {{ split_meta.splits.val.class_distribution[cls_id] }} | {{ split_meta.splits.test.class_distribution[cls_id] }} | {{ split_meta.splits.train.class_distribution[cls_id] + split_meta.splits.val.class_distribution[cls_id] + split_meta.splits.test.class_distribution[cls_id] }} |
{% endfor -%}
| **Total** | **{{ "{:,}".format(split_meta.splits.train.num_examples) }}** | **{{ "{:,}".format(split_meta.splits.val.num_examples) }}** | **{{ "{:,}".format(split_meta.splits.test.num_examples) }}** | **{{ "{:,}".format(split_meta.splits.train.num_examples + split_meta.splits.val.num_examples + split_meta.splits.test.num_examples) }}** |

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
