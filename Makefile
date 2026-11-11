# Thin wrappers around the CLI / test suite.
PYTHON ?= python
CONFIG ?= configs/sgdiff_cora.yaml

.PHONY: help install data test smoke train baselines benchmark ablation experiments transfer lowdata attack robustness clean

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

benchmark:  ## data-identical comparison of SGDiff vs baselines (Cora, CPU profile)
	$(PYTHON) scripts/run_benchmark.py --dataset cora --profile cpu --seeds 0

ablation:  ## GCCM shortcut check: is the likelihood really conditioned on y?
	$(PYTHON) scripts/run_ablation.py --dataset cora --profile cpu --seed 0

experiments:  ## in-domain grid (5 datasets x 3 seeds, CPU profile)
	$(PYTHON) scripts/run_experiments.py --datasets cora citeseer usair ns pubmed \
		--seeds 0 1 2 --profile cpu

transfer:  ## RQ1: zero-shot cross-dataset transfer
	$(PYTHON) scripts/run_experiments.py --transfer cora --profile cpu

lowdata:  ## RQ2: 1% training data
	$(PYTHON) scripts/run_experiments.py --lowdata cora --seeds 0 1 2 --profile cpu

attack:  ## RQ3: SGDiff under RF / DICE / EA
	$(PYTHON) scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5 --attack-methods rf dice ea --profile cpu

robustness:  ## RQ3 for every model (SGDiff + all baselines) under RF / DICE / EA
	$(PYTHON) scripts/run_robustness.py --dataset cora --profile cpu --seeds 0 1 2

clean:  ## remove caches and processed artifacts (keeps raw downloads)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf runs/ .pytest_cache/
