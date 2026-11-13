---
title: MNIST Lab Digit Classifier Demo
emoji: 🔢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
---

# MNIST-Lab Handwritten Digit Classifier 🔢

> Reproducible, configuration-driven PyTorch CNN classifier for handwritten digit recognition with interactive canvas drawing, image upload, and real-time probability distribution inspection.

**MNIST-Lab** provides a complete, reproducible deep learning pipeline combining **DVC** data/pipeline versioning, **Hydra** configuration management, **Weights & Biases** experiment tracking, and **Hugging Face Hub** model serving.

## 🚀 Application Details
This Hugging Face Space runs an interactive Streamlit UI deployed via a Docker container. It downloads the trained `MNISTConvNet` checkpoint (`model.safetensors` & `config.json`) from the Hugging Face Model Hub (`chrisjcc/mnist-cnn`) and runs real-time inference.

### Features
- **Interactive Freehand Canvas:** Draw any digit (`0`–`9`) directly in the browser with customizable stroke width and automatic `28 × 28` MNIST-normalized preprocessing.
- **Image Upload & Polarity Inversion:** Upload external PNG/JPG digit images with one-click foreground/background polarity inversion.
- **Synthetic & Test Sample Explorer:** Test canonical handwritten digit templates and inspect the exact normalized `28 × 28` tensor fed into the convolutional network.
- **10-Class Probability Distribution:** View top-1 predicted digit, confidence score, and full softmax probability distribution across classes `0` through `9`.

For more information, training methodology, DVC pipeline definitions, and Minimum Viable Experiment (MVE) benchmarks, please refer to the [main GitHub repository](https://github.com/chrisjcc/mnist-lab).
