#!/usr/bin/env bash
# Reproduce the paper's three research questions end to end.
#
#   RQ1  zero-shot cross-dataset transfer      (Sec. 4.2)
#   RQ2  1% training data                      (Sec. 4.3)
#   RQ3  robustness under structural attacks   (Sec. 4.4)
#
# Usage: bash scripts/run_all_experiments.sh [PROFILE] [SEEDS]
#        PROFILE = cpu (default, laptop friendly) | paper (Table 4, needs a GPU)
#        SEEDS   = space separated seeds, e.g. "0 1 2"
set -euo pipefail

PROFILE=${1:-cpu}
SEEDS=${2:-0 1 2}
PY=${PYTHON:-python}
OUT=${OUT:-results}
mkdir -p "$OUT"

echo "== data =="
$PY -m sgdiff prepare --datasets cora citeseer pubmed usair ns || true

echo "== in-domain (5 datasets x ${SEEDS}) =="
$PY scripts/run_experiments.py --datasets cora citeseer usair ns pubmed \
    --seeds ${SEEDS} --profile "${PROFILE}" | tee "$OUT/indomain.log"

echo "== RQ1: zero-shot transfer (structure only) =="
for SRC in cora citeseer pubmed; do
  $PY scripts/run_experiments.py --transfer "${SRC}" \
      --targets cora citeseer pubmed usair ns --profile "${PROFILE}" | tee "$OUT/transfer_${SRC}.log"
done

echo "== RQ2: 1% training data =="
$PY scripts/run_experiments.py --lowdata cora --seeds ${SEEDS} --profile "${PROFILE}" \
    | tee "$OUT/lowdata_cora.log"

echo "== RQ3: SGDiff under random flip / DICE / embedding attack (0.1-0.5) =="
$PY scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5 \
    --attack-methods rf dice ea --profile "${PROFILE}" | tee "$OUT/attack_cora.log"

echo "== RQ3 (all models): RF / DICE / EA against SGDiff and every baseline =="
$PY scripts/run_robustness.py --dataset cora --profile "${PROFILE}" --seeds ${SEEDS} \
    | tee "$OUT/robustness_cora.log"

echo "== baselines =="
$PY scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal \
    --output "$OUT/baselines_cora.json" || true

echo "== done: see $OUT/ =="
