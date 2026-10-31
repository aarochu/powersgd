#!/bin/bash
# NanoGPT / Shakespeare-char, 1 node (4x A100): does PowerSGD catch up to DDP's best loss with more steps?
# The 2000-step sweep (gpt_*_1node) showed DDP hitting its best val loss (1.48) at step 750 and then overfitting, while
# PowerSGD ranks 1/2/4 were still improving at step 2000. Here DDP gets 1000 steps (enough to pass its best) and
# PowerSGD ranks 2 and 4 get 6000. Eval every 500 steps. Same seed, start_iter and defaults as the sweep.
#
# STATUS: submitted on Perlmutter as job 60623330 (gpt_1n_long), PENDING at the time of writing. No results yet.
# NOTE: this file was written directly on Perlmutter and re-created in the repo from its description, so diff it
#       against the Perlmutter copy before relying on it. COST: 1 node-hour per wall-clock hour (<= ~0.7 node-h at the 40 min limit).
#   sbatch scripts/nanogpt_1node_long.sbatch
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:40:00
#SBATCH -J gpt_1n_long
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
COMMON="--model nanogpt --seed 1 --eval-interval 500 --start-iter 10"

start_gpu_monitor
launch $COMMON --mode ddp                --max-steps 1000 --run-name gpt_ddp_1000_1node
launch $COMMON --mode powersgd --rank 2  --max-steps 6000 --run-name gpt_r2_6000_1node
launch $COMMON --mode powersgd --rank 4  --max-steps 6000 --run-name gpt_r4_6000_1node
finish
