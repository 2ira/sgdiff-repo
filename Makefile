# Thin wrappers around the CLI / test suite.
PYTHON ?= python
CONFIG ?= configs/sgdiff_cora.yaml

.PHONY: help install data test smoke train baselines clean

help:  ## show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install:  ## install the package in editable mode
	$(PYTHON) -m pip install -e ".[dev]"

data:  ## download / preprocess every auto-downloadable dataset
	$(PYTHON) -m sgdiff prepare

test:  ## run the unit tests
	$(PYTHON) -m pytest tests/ -q

smoke:  ## 3-epoch end-to-end run on a synthetic graph (CPU friendly)
	$(PYTHON) -m sgdiff train --config configs/smoke_synthetic.yaml

train:  ## train SGDiff with CONFIG=configs/xxx.yaml
	$(PYTHON) -m sgdiff train --config $(CONFIG)

baselines:  ## train the baseline link predictors on Cora
	$(PYTHON) scripts/run_baselines.py --dataset cora --epochs 60

clean:  ## remove caches and processed artifacts (keeps raw downloads)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf runs/ .pytest_cache/
