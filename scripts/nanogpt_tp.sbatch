#!/bin/bash
# NanoGPT with tensor parallelism (train_tensor.py): the "data vs tensor parallelism" measurement.
# Each config is tp:mode:rank. tp=1 is plain data parallelism through the same trainer (the baseline); tp=2 and 4 split
# the model across 2 or 4 GPUs of a node, with data parallelism across the tensor-parallel groups. PowerSGD compresses
# only the data-parallel gradient all-reduce, never the tensor-parallel activation all-reduces.
#   export SBATCH_ACCOUNT=<your project>
#   sbatch scripts/nanogpt_tp.sbatch                  # tp1 ddp, tp1 r4, tp2 ddp, tp2 r4 (6 heads, as in results/)
#   sbatch -N 2 scripts/nanogpt_tp.sbatch             # the same on 2 nodes: data parallelism crosses the nodes
#   N_HEAD=12 sbatch scripts/nanogpt_tp.sbatch        # 12 heads: tp1 ddp, tp1 r4, tp4 ddp (+ tp4 r4 on 2+ nodes)
#   CONFIGS="2:ddp:0 2:powersgd:4" MAX_STEPS=1000 sbatch scripts/nanogpt_tp.sbatch
# tp must divide the head count: the 6-head model in results/ splits only 2 ways, so tp 4 runs use N_HEAD=12 (same
# parameter count, head width 32) and need their own tp 1 baseline with N_HEAD=12, which that default list includes.
# The per-replica batch defaults to 64 x tp, so every config has the same global batch (64 sequences per GPU).
# Run scripts/smoke_tp.sh first. Cost: the first smoke test measured 78 ms (tp1) and 100 ms (tp2) per step on 1 node,
# so 4 configs x 2000 steps is about 15 minutes plus evaluation. The time limit caps it at 0.75 node-hours per node
# (1.5 on 2 nodes).
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:45:00
#SBATCH -J gpt_tp
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
SEED=${SEED:-1}
N_HEAD=${N_HEAD:-6}
if [[ -z ${CONFIGS:-} ]]; then
  if [[ $N_HEAD == 6 ]]; then
    CONFIGS="1:ddp:0 1:powersgd:4 2:ddp:0 2:powersgd:4"
  else
    CONFIGS="1:ddp:0 1:powersgd:4 4:ddp:0"
    if [[ $SLURM_NNODES -gt 1 ]]; then CONFIGS="$CONFIGS 4:powersgd:4"; fi
  fi
fi
for c in $CONFIGS; do  # fail now, not after an hour of other configs
  tp=${c%%:*}
  if (( N_HEAD % tp )); then echo "tp $tp does not divide N_HEAD=$N_HEAD (config $c)" >&2; exit 1; fi
done
HTAG=$([[ $N_HEAD == 6 ]] && echo "" || echo "_h$N_HEAD")
COMMON="--seed $SEED --n-head $N_HEAD --max-steps ${MAX_STEPS:-2000} --eval-interval 250 --start-iter 10"

start_gpu_monitor
for c in $CONFIGS; do
  IFS=: read -r tp mode r <<<"$c"
  tag=$([[ $mode == ddp ]] && echo ddp || echo "r$r")
  launch_with train_tensor.py $COMMON --tp-size "$tp" --mode "$mode" --rank "${r:-2}" \
    --run-name "gpt_tp${tp}${HTAG}_${tag}_${SLURM_NNODES}node_s${SEED}_${SLURM_JOB_ID}"
done
finish
