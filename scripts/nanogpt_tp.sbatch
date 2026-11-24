#!/bin/bash
# NanoGPT with tensor parallelism (train_tensor.py): the "data vs tensor parallelism" measurement.
# Each config is tp:mode:rank. tp=1 is plain data parallelism through the same trainer (the baseline); tp=2 and 4 split
# the model across 2 or 4 GPUs of a node, with data parallelism across the tensor-parallel groups. PowerSGD compresses
# only the data-parallel gradient all-reduce, never the tensor-parallel activation all-reduces.
#   export SBATCH_ACCOUNT=<your project>
#   sbatch scripts/nanogpt_tp.sbatch                  # 1 node: tp1 ddp, tp1 r4, tp2 ddp, tp2 r4, tp4 ddp
#   sbatch -N 2 scripts/nanogpt_tp.sbatch             # 2 nodes: the same plus tp4 with PowerSGD across the nodes
#   CONFIGS="2:ddp:0 2:powersgd:4" MAX_STEPS=1000 sbatch scripts/nanogpt_tp.sbatch
# The per-replica batch defaults to 64 x tp, so every config has the same global batch (64 sequences per GPU).
# Run scripts/smoke_tp.sh first. Cost: not measured; a 2000-step data-parallel run took about 3 minutes, so 5 configs
# should take roughly 15-30 minutes. The time limit caps it at 0.75 node-hours per node (1.5 on 2 nodes).
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
if [[ -z ${CONFIGS:-} ]]; then
  CONFIGS="1:ddp:0 1:powersgd:4 2:ddp:0 2:powersgd:4 4:ddp:0"
  if [[ $SLURM_NNODES -gt 1 ]]; then CONFIGS="$CONFIGS 4:powersgd:4"; fi
fi
COMMON="--seed $SEED --max-steps ${MAX_STEPS:-2000} --eval-interval 250 --start-iter 10"

start_gpu_monitor
for c in $CONFIGS; do
  IFS=: read -r tp mode r <<<"$c"
  tag=$([[ $mode == ddp ]] && echo ddp || echo "r$r")
  launch_with train_tensor.py $COMMON --tp-size "$tp" --mode "$mode" --rank "${r:-2}" \
    --run-name "gpt_tp${tp}_${tag}_${SLURM_NNODES}node_s${SEED}_${SLURM_JOB_ID}"
done
finish
