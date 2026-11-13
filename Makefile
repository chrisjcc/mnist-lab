-include .env
export

# Dynamic Environment Isolation
export PYTHONNOUSERSITE := 1

ifneq ($(CONDA_PREFIX),)
export LD_LIBRARY_PATH := $(CONDA_PREFIX)/lib:$(LD_LIBRARY_PATH)
endif

# ------------------------------------------------------------
# Environment & Tooling Resolution
# ------------------------------------------------------------

VENV ?= .venv
PYTHON := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP := $(if $(wildcard $(VENV)/bin/pip),$(VENV)/bin/pip,pip)
RUFF := $(if $(wildcard $(VENV)/bin/ruff),$(VENV)/bin/ruff,ruff)
MYPY := $(if $(wildcard $(VENV)/bin/mypy),$(VENV)/bin/mypy,mypy)
PYTEST := $(if $(wildcard $(VENV)/bin/pytest),$(VENV)/bin/pytest,pytest)
DVC := $(if $(wildcard $(VENV)/bin/dvc),$(VENV)/bin/dvc,dvc)
STREAMLIT := $(if $(wildcard $(VENV)/bin/streamlit),$(VENV)/bin/streamlit,streamlit)

# Deployment & Container Configuration
APP_IMAGE ?= mnist-lab-app
MODEL_REPO ?= chrisjcc/mnist-cnn
DATASET_REPO ?= chrisjcc/mnist-dataset

.PHONY: help \
        setup install install-dev lint format test setup-hooks clean \
        data preprocess dvc-repro pipeline \
        train evaluate benchmark report \
        smoke-eval evaluate-mve \
        generate-hf-cards publish-hf-cards publish-hf-model publish-hf-dataset publish-hf-release \
        hf-upload hf-inference \
        app-run run-app docker-build docker-run

# ------------------------------------------------------------
# Operational Help Command
# ------------------------------------------------------------

help:
	@echo "================================================================================"
	@echo "                        MNIST-Lab Operational Commands                          "
	@echo "================================================================================"
	@echo "Environment & Code Quality:"
	@echo "  make setup                    Install requirements.txt and editable package"
	@echo "  make install                  Install all project dependencies (training, dev, UI)"
	@echo "  make install-dev              Install minimal development dependencies only"
	@echo "  make lint                     Run ruff linter, format check, and mypy type check"
	@echo "  make format                   Auto-format codebase via ruff"
	@echo "  make test                     Execute pytest unit and integration test suite"
	@echo "  make clean                    Remove build caches and temporary files"
	@echo ""
	@echo "Data & Pipeline Versioning (DVC):"
	@echo "  make data                     Download and preprocess MNIST dataset"
	@echo "  make preprocess               Alias for make data"
	@echo "  make pipeline                 Reproduce full DVC pipeline (dvc repro)"
	@echo "  make dvc-repro                Alias for make pipeline"
	@echo ""
	@echo "Model Training, Evaluation, Benchmarking & Reporting:"
	@echo "  make train                    Train MNIST CNN model using Hydra configuration"
	@echo "  make evaluate                 Evaluate best checkpoint on held-out test set"
	@echo "  make benchmark                Run inference latency, throughput & memory benchmarks"
	@echo "  make report                   Generate evaluation report, plots, and HF cards"
	@echo "  make smoke-eval               Execute fast (< 2.0s) zero-budget CI smoke evaluation"
	@echo "  make evaluate-mve             Adjudicate MVE baseline acceptance criteria gates"
	@echo ""
	@echo "Hugging Face Hub Integration:"
	@echo "  make generate-hf-cards        Generate Model Card & Dataset Card locally (--card-only)"
	@echo "  make hf-upload                Upload trained model, cards & artifacts to HF Hub"
	@echo "  make hf-inference             Download model from HF Hub and run inference"
	@echo "  make publish-hf-release       Publish verified production model & cards to HF Hub"
	@echo ""
	@echo "Interactive Demo App & Docker Space:"
	@echo "  make app-run                  Launch local interactive Streamlit demo UI"
	@echo "  make run-app                  Alias for make app-run"
	@echo "  make docker-build             Build Hugging Face Space Docker container image"
	@echo "  make docker-run               Run interactive Streamlit demo in Docker (port 8501)"
	@echo "================================================================================"

# ------------------------------------------------------------
# 1. Environment & Code Quality
# ------------------------------------------------------------

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[all]"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(RUFF) check src/ scripts/ tests/ app/
	$(RUFF) format --check src/ scripts/ tests/ app/
	$(MYPY) src/

format:
	$(RUFF) check --fix src/ scripts/ tests/ app/
	$(RUFF) format src/ scripts/ tests/ app/

test:
	$(PYTEST) tests/

setup-hooks:
	@echo "Installing native git pre-commit hook..."
	@mkdir -p .git/hooks
	@printf '#!/usr/bin/env bash\nset -e\nif [ -d .venv/bin ]; then export PATH="$$PWD/.venv/bin:$$PATH"; fi\necho "[pre-commit] Running ruff lint & format checks..."\nruff check src/ scripts/ tests/ app/\nruff format --check src/ scripts/ tests/ app/\necho "[pre-commit] Running fast smoke evaluation gate..."\npython scripts/run_smoke_eval.py\n' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Git pre-commit hook installed successfully!"

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

# ------------------------------------------------------------
# 2. Data Ingestion & DVC Pipeline
# ------------------------------------------------------------

data:
	$(PYTHON) scripts/preprocess.py $(ARGS)

preprocess:
	$(PYTHON) scripts/preprocess.py $(ARGS)

pipeline:
	$(DVC) repro $(ARGS)

dvc-repro:
	$(DVC) repro $(ARGS)

# ------------------------------------------------------------
# 3. Training, Evaluation, Benchmarking & Reporting
# ------------------------------------------------------------

train:
	$(PYTHON) scripts/train.py $(ARGS)

evaluate:
	$(PYTHON) scripts/evaluate.py $(ARGS)

benchmark:
	$(PYTHON) scripts/benchmark.py $(ARGS)

report:
	$(PYTHON) scripts/generate_report.py $(ARGS)

smoke-eval:
	$(PYTHON) scripts/run_smoke_eval.py $(ARGS)

evaluate-mve:
	$(PYTHON) scripts/run_mve_evaluation.py $(ARGS)

# ------------------------------------------------------------
# 4. Hugging Face Hub Integration
# ------------------------------------------------------------

generate-hf-cards:
	$(PYTHON) scripts/release_to_hf.py --card-only $(ARGS)

publish-hf-cards:
	$(PYTHON) scripts/release_to_hf.py --publish-cards $(ARGS)

hf-upload:
	$(PYTHON) scripts/hf_upload.py $(ARGS)

hf-inference:
	$(PYTHON) scripts/hf_inference.py $(ARGS)

publish-hf-release:
	@echo "=========================================================="
	@echo "Deploying verified MNIST model & cards to Hugging Face Hub"
	@echo "  Dataset: https://huggingface.co/datasets/$(DATASET_REPO)"
	@echo "  Model:   https://huggingface.co/$(MODEL_REPO)"
	@echo "=========================================================="
	@test -n "$$HF_TOKEN" || ( \
		echo "ERROR: HF_TOKEN environment variable is not set."; \
		echo "Export HF_TOKEN=<your-token> or add it to .env"; \
		exit 1; \
	)
	@RELEASE_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.1.0") && \
	$(PYTHON) scripts/release_to_hf.py --publish-cards \
		--model-repo $(MODEL_REPO) \
		--dataset-repo $(DATASET_REPO) \
		--release-tag "$$RELEASE_TAG" $(ARGS)

# ------------------------------------------------------------
# 5. Interactive Demo App & Docker Space
# ------------------------------------------------------------

app-run:
	$(STREAMLIT) run app/app.py $(ARGS)

run-app:
	$(STREAMLIT) run app/app.py $(ARGS)

docker-build:
	docker build -t $(APP_IMAGE) .

docker-run:
	docker run -p 8501:8501 $(if $(wildcard .env),--env-file .env,) $(APP_IMAGE)
