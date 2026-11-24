#!/bin/bash
# Submit a slice of the paper-code scaling grid, one Slurm job per run, with shorter time limits than
# `submit_paper.sh scaling` (30 jobs at 30 minutes each). Run names and folders match submit_paper.sh, so finished
# runs (DONE file) and queued runs are skipped. GPU_MONITOR=1 (the default here) adds an nvidia-smi CSV per node.
#
#   export NERSC_ACCOUNT=<your project>
#   DRY_RUN=1 BACKENDS=nccl WORKERS="8 16" ./perlmutter/submit_scaling.sh     # list the jobs and the node-hour cap
#   BACKENDS=nccl WORKERS="8 16" ./perlmutter/submit_scaling.sh               # matched NCCL baseline for the Gloo runs
#   BACKENDS="nccl gloo" WORKERS="1 2 4" ./perlmutter/submit_scaling.sh       # the rest of the scaling curve
#   BACKENDS=gloo WORKERS="8 16" SEEDS="2 3" ./perlmutter/submit_scaling.sh   # more seeds
#   GROUP=gloo16_full BACKENDS=gloo WORKERS=16 SEEDS="1 2 3" EPOCHS=300 TIME=00:30:00 ./perlmutter/submit_scaling.sh
#
# Variables and defaults:
#   GROUP=scaling  TASK=cifar  PRESETS="SGD;Rank 2" (';'-separated)  BACKENDS="nccl gloo"  WORKERS="1 2 4 8 16"
#   SEEDS=1  EPOCHS=10  TIME=00:10:00  QOS=regular  GPU_MONITOR=1  DRY_RUN=0  RESERVATION (optional)
# The run name has no epoch count, so runs with different EPOCHS need a different GROUP.
set -eo pipefail
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
export REPO_DIR
source "$REPO_DIR/perlmutter/env.sh"
DRY_RUN=${DRY_RUN:-0}
[[ $DRY_RUN == 1 ]] || : "${NERSC_ACCOUNT:?export NERSC_ACCOUNT=<your NERSC project, e.g. m1234>}"
GROUP=${GROUP:-scaling}
TASK=${TASK:-cifar}
PRESETS=${PRESETS:-"SGD;Rank 2"}
BACKENDS=${BACKENDS:-"nccl gloo"}
WORKERS=${WORKERS:-"1 2 4 8 16"}
SEEDS=${SEEDS:-1}
EPOCHS=${EPOCHS:-10}
TIME=${TIME:-00:10:00}
QOS=${QOS:-regular}
export GPU_MONITOR=${GPU_MONITOR:-1}
LOG_DIR=$RESULTS/slurm-logs
mkdir -p "$LOG_DIR"
QUEUED=$(squeue --me -h -o %j 2>/dev/null || true)
limit_h=$(awk -F: '{print $1 + $2 / 60 + $3 / 3600}' <<<"$TIME")
IFS=';' read -r -a preset_list <<<"$PRESETS"

n_jobs=0
node_hours=0
for backend in $BACKENDS; do
  for preset in "${preset_list[@]}"; do
    for workers in $WORKERS; do
      for seed in $SEEDS; do
        slug=$(tr '[:upper:]' '[:lower:]' <<<"$preset" | sed -E 's/[^a-z0-9]+/-/g; s/^-|-$//g')
        name="${TASK}_${slug}_w${workers}_${backend}_s${seed}"
        out=$RESULTS/$GROUP/$name
        job_name="${GROUP}_$name"
        if [[ -f $out/DONE ]]; then echo "done      $GROUP/$name"; continue; fi
        if grep -qx "$job_name" <<<"$QUEUED"; then echo "queued    $GROUP/$name"; continue; fi
        nodes=$(((workers + 3) / 4))
        tpn=$((workers < 4 ? workers : 4))
        n_jobs=$((n_jobs + 1))
        node_hours=$(awk -v a="$node_hours" -v n="$nodes" -v h="$limit_h" 'BEGIN {print a + n * h}')
        if [[ $DRY_RUN == 1 ]]; then echo "would run $GROUP/$name ($nodes node(s), --epochs $EPOCHS, -t $TIME)"; continue; fi
        job=$(env TASK="$TASK" PRESET="$preset" SEED="$seed" WORKERS="$workers" TASKS_PER_NODE="$tpn" \
            BACKEND="$backend" OUT="$out" EXTRA_ARGS="--epochs $EPOCHS" \
            sbatch --parsable -A "$NERSC_ACCOUNT" -q "$QOS" -t "$TIME" -N "$nodes" --ntasks-per-node="$tpn" \
            ${RESERVATION:+--reservation=$RESERVATION} \
            -J "$job_name" -o "$LOG_DIR/%x-%j.out" "$REPO_DIR/perlmutter/run.sbatch")
        QUEUED+=$'\n'"$job_name"
        echo "submitted $GROUP/$name (job $job)"
      done
    done
  done
done
echo "$n_jobs job(s); time limits add up to at most $node_hours node-hours (actual use is usually far less)"
