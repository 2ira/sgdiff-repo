#!/bin/bash
# SGDiff-paper-v2 audit queue: likelihood ablation + mechanism probe chain.
# 8 jobs, one worker per GPU (0-7), flock-protected job list.
cd /apdcephfs_sgxy/share_304335953/hunyuan/xinyuirali/dev/sgdiff-repo || exit 1

JOBS=/tmp/sgdiff_audit_jobs.txt
LOCK=/tmp/sgdiff_audit_jobs.lock
LOG_DIR=results/sgdiff-paper-v2/logs
OUT=results/sgdiff-paper-v2
mkdir -p "$LOG_DIR" "$OUT"

run_worker() {
  local gpu=$1
  while true; do
    local line
    line=$(flock "$LOCK" bash -c "head -1 '$JOBS' 2>/dev/null; sed -i '1d' '$JOBS' 2>/dev/null")
    [ -z "$line" ] && break
    local tag=${line%%|||*}
    local cmd=${line#*|||}
    echo "[$(date '+%m-%d %H:%M:%S')] GPU${gpu} START ${tag}"
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "$LOG_DIR/${tag}.log" 2>&1
    local rc=$?
    echo "[$(date '+%m-%d %H:%M:%S')] GPU${gpu} DONE  ${tag} (exit ${rc})"
  done
  echo "[$(date '+%m-%d %H:%M:%S')] GPU${gpu} audit queue empty, worker exit"
}

: > "$JOBS"
# --- likelihood ablation (full audit incl. randomized-condition retrain) ---
for ds in cora citeseer usair; do
  echo "ablation_${ds}|||python -u scripts/run_ablation.py --dataset $ds --profile paper --seed 0 --output $OUT --tag _v2" >> "$JOBS"
done
# --- mechanism probes (which property does each model read) ---
for ds in cora citeseer usair; do
  echo "probe_${ds}|||python -u scripts/run_probe.py --dataset $ds --profile paper --seeds 0 1 2 --output $OUT --tag _v2" >> "$JOBS"
done
# --- deep probes (layer/operator/edge-kind audit; old protocol: seeds 0-1) ---
for ds in cora citeseer; do
  echo "deep_${ds}|||python -u scripts/run_deep_probe.py --dataset $ds --profile paper --seeds 0 1 --output $OUT --tag _v2" >> "$JOBS"
done
touch "$LOCK"

for gpu in 0 1 2 3 4 5 6 7; do
  run_worker "$gpu" &
  sleep 2
done
wait
echo "ALL AUDIT JOBS FINISHED"
