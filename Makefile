VENV := venv
PYTHON := python3
PIP := pip
OC := oc
FASTAPI_PORT := 8000
CHAINLIT_PORT := 8001

default: help

help:
	@echo "Please use 'make <target>' where <target>' is one of"
	@echo "  setup           to create a virtual environment"
	@echo "  train        	 to run train.py, training the model"
	@echo "  run-bigram      to run bigram.py"
	@echo "  run-custom-gpt  to run custom_gpt.py"

.PHONY: check-python-version
check-python-version:
	@$(PYTHON) -c 'import sys; sys.exit("\033[31mPython 3.11 or higher is required to run this project. Please ensure that `python3` is pointing at a valid python version\033[0m" if sys.version_info < (3, 11) else 0)' || exit 1

.PHONY: setup
setup: check-python-version
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created."
	@echo "Installing dependencies..."
	@. $(VENV)/bin/activate && $(PIP) install --no-cache-dir --upgrade -r requirements.txt

.PHONY: train
train: check-python-version
	@. $(VENV)/bin/activate && $(PYTHON) train.py

.PHONY: run-bigram
run-bigram: check-python-version
	@. $(VENV)/bin/activate && $(PYTHON) bigram.py

.PHONY: run-custom-gpt
run-custom-gpt: check-python-version
	@. $(VENV)/bin/activate && $(PYTHON) custom_gpt.py