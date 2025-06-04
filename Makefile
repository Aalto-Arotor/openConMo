# Makefile for openconmo project

VENV ?= venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# ---------------------------
# Environment & Installation
# ---------------------------

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	$(PIP) install -r requirements.txt


ui:
	openconmo-ui

# ---------------------------
# Maintenance
# ---------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist build .pytest_cache
	find . -type d -iname "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

.PHONY: install test notebook ui clean