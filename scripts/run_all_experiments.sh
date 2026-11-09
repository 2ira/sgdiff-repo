#!/usr/bin/env bash
# Reproduce the three research questions of the paper (small-budget defaults).
#
#   RQ1  cross-dataset transfer
#   RQ2  1% training data
#   RQ3  robustness against structural attacks
#
# Usage: bash scripts/run_all_experiments.sh [DATASET] [EPOCHS]
set -euo pipefail

DATASET=${1:-cora}
EPOCHS=${2:-20}
PY=${PYTHON:-python}
OUT=${OUT:-results}
mkdir -p "$OUT"

echo "== 0. prepare data =="
$PY -m sgdiff prepare --datasets cora citeseer pubmed usair ns || true

echo "== 1. train on ${DATASET} (${EPOCHS} epochs) =="
$PY -m sgdiff train --config configs/sgdiff_${DATASET}.yaml \
  --run-dir runs --set train.run_name=${DATASET}_main train.epochs=${EPOCHS} \
  | tee "$OUT/train_${DATASET}.log"

CKPT=$(ls -t runs/${DATASET}_main/best.pt | head -1)

echo "== 2. in-domain test =="
$PY -m sgdiff evaluate --checkpoint "$CKPT" --split test --device cpu \
  --output "$OUT/test_${DATASET}.json"

echo "== 3. RQ1: zero-shot transfer =="
$PY -m sgdiff transfer --checkpoint "$CKPT" --targets citeseer pubmed usair ns \
  --device cpu --output "$OUT/transfer_${DATASET}.json"

echo "== 4. RQ2: 1% training data =="
$PY -m sgdiff train --config configs/lowdata_cora_1pct.yaml --run-dir runs \
  --set train.run_name=lowdata_${DATASET} train.epochs=${EPOCHS} \
  | tee "$OUT/lowdata_train_${DATASET}.log" || true
LOW=$(ls -t runs/lowdata_${DATASET}/best.pt 2>/dev/null | head -1 || true)
if [ -n "${LOW}" ]; then
  $PY -m sgdiff evaluate --checkpoint "$LOW" --split test --device cpu \
    --output "$OUT/lowdata_test_${DATASET}.json"
fi

echo "== 5. RQ3: robustness under structural attacks =="
for RATIO in 0.05 0.10 0.20; do
  $PY -m sgdiff evaluate --checkpoint "$CKPT" --split test --device cpu \
    --attack random --ratio "$RATIO" --output "$OUT/attack_random_${RATIO}_${DATASET}.json"
  $PY -m sgdiff evaluate --checkpoint "$CKPT" --split test --device cpu \
    --attack dice --ratio "$RATIO" --output "$OUT/attack_dice_${RATIO}_${DATASET}.json"
done

echo "== done. results written to $OUT/ =="
