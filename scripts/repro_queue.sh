#!/bin/bash
# SGDiff paper-profile reproduction queue: usair/ns/citeseer/pubmed x 10 seeds.
# 4 workers pull jobs one at a time from a shared list (flock-protected).
cd /apdcephfs_sgxy/share_304335953/hunyuan/xinyuirali/dev/sgdiff-repo || exit 1

JOBS_FILE=/tmp/sgdiff_joblist.txt
LOCK=/tmp/sgdiff_joblist.lock
LOG_DIR=results/sgdiff-paper-v2/logs
mkdir -p "$LOG_DIR" runs/sgdiff-paper-v2

run_worker() {
  local gpu=$1
  while true; do
    local job
    job=$(flock "$LOCK" bash -c "head -1 '$JOBS_FILE' 2>/dev/null; sed -i '1d' '$JOBS_FILE' 2>/dev/null")
    [ -z "$job" ] && break
    local ds seed
    ds=$(echo "$job" | cut -d, -f1)
    seed=$(echo "$job" | cut -d, -f2)
    echo "[$(date '+%m-%d %H:%M:%S')] GPU${gpu} START ${ds} seed${seed}"
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/run_experiments.py \
        --datasets "$ds" --seeds "$seed" --profile paper \
        > "$LOG_DIR/${ds}_seed${seed}_paper.log" 2>&1
    echo "[$(date '+%m-%d %H:%M:%S')] GPU${gpu} DONE  ${ds} seed${seed} (exit $?)"
  done
  echo "[$(date '+%m-%d %H:%M:%S')] GPU${gpu} queue empty, worker exit"
}

# job list: small datasets first so they finish while Cora still runs
: > "$JOBS_FILE"
for s in 0 1 2 3 4 5 6 7 8 9; do echo "usair,$s"    >> "$JOBS_FILE"; done
for s in 0 1 2 3 4 5 6 7 8 9; do echo "ns,$s"      >> "$JOBS_FILE"; done
for s in 0 1 2 3 4 5 6 7 8 9; do echo "citeseer,$s" >> "$JOBS_FILE"; done
for s in 0 1 2 3 4 5 6 7 8 9; do echo "pubmed,$s"  >> "$JOBS_FILE"; done
touch "$LOCK"

for gpu in 4 5 6 7; do
  run_worker "$gpu" &
  sleep 2
done
wait
echo "ALL JOBS FINISHED"
