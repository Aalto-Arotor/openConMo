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
# Documentation
# ---------------------------

docs-html:
	$(VENV)/bin/sphinx-build -b html docs docs/_build/html


docs-open:
	xdg-open docs/_build/html/index.html || open docs/_build/html/index.html

# ---------------------------
# Maintenance
# ---------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist build .pytest_cache
	find . -type d -iname "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf docs/_build

.PHONY: install test notebook ui clean docs-html docs-clean docs-open
