#!/bin/bash
# NanoGPT with a MODELED slow network (train_ddp.py --sim-bandwidth-gbps): DDP vs PowerSGD rank 2 and 4, each with
# no modeled delay and at 10 Gbit/s (the paper's Ethernet), 2000 steps, in one job so the hardware is the same.
# After every all-reduce the trainer sleeps for latency + 2(N-1)/N * bytes / bandwidth. That ignores overlap with
# compute and contention, so report it as "modeled 10 Gbit/s link", not as a measured network.
#   export SBATCH_ACCOUNT=<your project>
#   sbatch scripts/nanogpt_slowlink.sbatch
#   SIM_BWS="0 1 10 25" SIM_LAT=50 sbatch scripts/nanogpt_slowlink.sbatch     # Gbit/s list, latency in microseconds
#   sbatch -N 2 scripts/nanogpt_slowlink.sbatch                               # 8 GPUs (40 GB A100s)
# Cost: on 4 GPUs at 10 Gbit/s DDP gains about 52 ms per step (2 x 3/4 x 43.2 MB / 1.25 GB/s), about +100 s per run;
# 6 runs: about 0.35 node-hours. Capped at 0.75 node-hours by the time limit.
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:45:00
#SBATCH -J gpt_slow
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
SEED=${SEED:-1}
SIM_BWS=${SIM_BWS:-"0 10"}
SIM_LAT=${SIM_LAT:-0}
COMMON="--model nanogpt --seed $SEED --max-steps 2000 --eval-interval 250 --start-iter 10"

start_gpu_monitor
for bw in $SIM_BWS; do
  SIM="--sim-bandwidth-gbps $bw --sim-latency-us $SIM_LAT"
  launch $COMMON $SIM --mode ddp --run-name "gpt_ddp_bw${bw}_lat${SIM_LAT}_${SLURM_NNODES}node_${SLURM_JOB_ID}"
  for r in 2 4; do
    launch $COMMON $SIM --mode powersgd --rank "$r" --run-name "gpt_r${r}_bw${bw}_lat${SIM_LAT}_${SLURM_NNODES}node_${SLURM_JOB_ID}"
  done
done
finish
