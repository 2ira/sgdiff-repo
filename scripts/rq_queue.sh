#!/bin/bash
# SGDiff-paper-v2 RQ rerun queue: lowdata (RQ2), robustness (RQ3 all models),
# attack (RQ3 SGDiff), baselines, transfer (RQ1).
# 4 workers on GPUs 0-3 pull jobs from a flock-protected list.
cd /apdcephfs_sgxy/share_304335953/hunyuan/xinyuirali/dev/sgdiff-repo || exit 1

JOBS=/tmp/sgdiff_rq_jobs.txt
LOCK=/tmp/sgdiff_rq_jobs.lock
LOG_DIR=results/sgdiff-paper-v2/logs
mkdir -p "$LOG_DIR" results/sgdiff-paper-v2

# Backup in-domain cora checkpoints: lowdata/attack reuse the same run_name
# (cora_seed{K}_paper) and would overwrite them with different models.
for s in 0 1 2 3 4 5 6 7 8 9; do
  src="runs/sgdiff-paper-v2/cora_seed${s}_paper"
  dst="runs/sgdiff-paper-v2/indomain_cora_seed${s}_paper_bak"
  if [ -d "$src" ] && [ ! -d "$dst" ]; then
    cp -r "$src" "$dst" && echo "backed up $src -> $dst"
  fi
done

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
  echo "[$(date '+%m-%d %H:%M:%S')] GPU${gpu} RQ queue empty, worker exit"
}

PY="python -u"
TGTS="cora citeseer pubmed usair ns"
OUT=results/sgdiff-paper-v2

# --- job list: fast jobs first, heavy pubmed/cora transfer last ---
: > "$JOBS"
# RQ2: 1% training data (fast: ~1% subgraphs, longer schedule)
for s in 0 1 2 3 4 5 6 7 8 9; do
  echo "lowdata_cora_s${s}|||$PY scripts/run_experiments.py --lowdata cora --seeds $s --profile paper" >> "$JOBS"
done
# baselines on the same split (cheap)
echo "baselines_cora|||$PY scripts/run_baselines.py --dataset cora --baselines gcn gat sage vgae seal --output $OUT/baselines_cora.json" >> "$JOBS"
# RQ3 all models (SGDiff v2 + baselines) under RF/DICE/EA, 3 seeds
for s in 0 1 2; do
  echo "robustness_cora_s${s}|||$PY scripts/run_robustness.py --dataset cora --profile paper --seeds $s --output $OUT --tag _v2_seed${s}" >> "$JOBS"
done
# RQ3 SGDiff-only attack grid (retrains cora per seed, ~4.7h each)
for s in 0 1 2; do
  echo "attack_cora_s${s}|||$PY scripts/run_experiments.py --attack cora --ratios 0.1 0.25 0.5 --attack-methods rf dice ea --seeds $s --profile paper" >> "$JOBS"
done
# RQ1 transfer: train source structure-only once per seed, eval on all targets
for src in usair ns citeseer cora pubmed; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    echo "transfer_${src}_s${s}|||$PY scripts/run_experiments.py --transfer $src --targets $TGTS --seeds $s --profile paper" >> "$JOBS"
  done
done
touch "$LOCK"

for gpu in 0 1 2 3; do
  run_worker "$gpu" &
  sleep 2
done
wait
echo "ALL RQ JOBS FINISHED"
