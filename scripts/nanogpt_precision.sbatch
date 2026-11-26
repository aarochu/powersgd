#!/bin/bash
# NanoGPT, 1 node (4 GPUs): gradient communication precision, 2000 steps each.
#   DDP in fp32, fp16, bf16, fp8_e4m3, fp8_e5m2, and PowerSGD rank 4 in fp32 and fp16.
# fp16/bf16 really halve the all-reduce payload. fp8 is EMULATED (A100 has no fp8): the gradient is quantized to fp8
# and back before an fp32 all-reduce, so the accuracy effect is real but the logged bytes and step time are fp32's.
# fp8 would send 1/4 of the fp32 bytes; report that as computed, not measured.
#   export SBATCH_ACCOUNT=<your project>
#   sbatch scripts/nanogpt_precision.sbatch
#   CONFIGS="ddp:0:fp32 ddp:0:fp8_e4m3" SEED=2 sbatch scripts/nanogpt_precision.sbatch     # mode:rank:precision
# Cost: about 3 min per run, 7 runs: about 0.35-0.45 node-hours; capped at 0.67 by the time limit.
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:40:00
#SBATCH -J gpt_prec
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
SEED=${SEED:-1}
CONFIGS=${CONFIGS:-"ddp:0:fp32 ddp:0:fp16 ddp:0:bf16 ddp:0:fp8_e4m3 ddp:0:fp8_e5m2 powersgd:4:fp32 powersgd:4:fp16"}
COMMON="--model nanogpt --seed $SEED --max-steps 2000 --eval-interval 250 --start-iter 10"

start_gpu_monitor
for c in $CONFIGS; do
  IFS=: read -r mode r prec <<<"$c"
  tag=$([[ $mode == ddp ]] && echo ddp || echo "r$r")
  # a failed config (e.g. PowerSGD in fp16) is logged as an error and the loop goes on
  launch $COMMON --mode "$mode" --rank "${r:-2}" --precision "$prec" --run-name "gpt_${tag}_${prec}_s${SEED}_${SLURM_JOB_ID}"
done
finish
