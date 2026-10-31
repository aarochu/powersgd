#!/bin/bash
# NanoGPT / Shakespeare-char on 2 nodes x 4 A100 = 8 GPUs: plain DDP vs PowerSGD rank 2 and rank 4.
# Question: once the all-reduce crosses the node-to-node link, does compression start to help?
# Same seed/steps/eval settings as the 1-node sweep (gpt_*_1node) so step-time and loss curves can be compared;
# per-GPU batch is the trainer default (64), so the global batch is 512 here vs 256 on 1 node (lr is NOT rescaled for GPT).
#
# STATUS: submitted on Perlmutter as job 60623329 (gpt_2n), PENDING at the time of writing. No results yet.
# NOTE: this file was written directly on Perlmutter and re-created in the repo from its description, so diff it
#       against the Perlmutter copy before relying on it. COST: 2 node-hours per wall-clock hour (<= ~1.3 node-h at the 40 min limit).
#   sbatch scripts/nanogpt_2node.sbatch
#SBATCH -A m4431_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:40:00
#SBATCH -J gpt_2n
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
COMMON="--model nanogpt --seed 1 --max-steps 2000 --eval-interval 250 --start-iter 10"

start_gpu_monitor
launch $COMMON --mode ddp                --run-name gpt_ddp_2node
launch $COMMON --mode powersgd --rank 2  --run-name gpt_r2_2node
launch $COMMON --mode powersgd --rank 4  --run-name gpt_r4_2node
finish
