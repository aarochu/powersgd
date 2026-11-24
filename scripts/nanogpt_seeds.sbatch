#!/bin/bash
# NanoGPT, 1 node (4 GPUs): the 1-node sweep (DDP vs PowerSGD rank 1, 2, 4; 2000 steps) with more seeds, so the
# DDP-vs-PowerSGD gaps can be compared with seed-to-seed spread. Seed 1 is already in results/.
#   export SBATCH_ACCOUNT=<your project>
#   sbatch scripts/nanogpt_seeds.sbatch                     # seeds 2 and 3
#   SEEDS="2 3 4" sbatch scripts/nanogpt_seeds.sbatch
# Cost: about 3 min per run (DDP took 166 s in job 60622124), 8 runs: about 0.4-0.5 node-hours.
# The time limit below caps it at 0.75 node-hours.
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:45:00
#SBATCH -J gpt_seeds
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
SEEDS=${SEEDS:-"2 3"}
COMMON="--model nanogpt --max-steps 2000 --eval-interval 250 --start-iter 10"

start_gpu_monitor
for s in $SEEDS; do
  launch $COMMON --seed "$s" --mode ddp --run-name "gpt_ddp_s${s}_1node_${SLURM_JOB_ID}"
  for r in 1 2 4; do
    launch $COMMON --seed "$s" --mode powersgd --rank "$r" --run-name "gpt_r${r}_s${s}_1node_${SLURM_JOB_ID}"
  done
done
finish
