#!/bin/bash
# Larger NanoGPT: is PowerSGD more useful when the gradient is bigger? DDP vs PowerSGD rank 2 and 4.
#   SIZE=medium  8 layers,  8 heads, width 512: about 25M parameters, about 100 MB fp32 gradient
#   SIZE=large  12 layers, 12 heads, width 768: about 85M parameters, about 340 MB fp32 gradient
#   (12 x layers x width^2; the run logs the exact count.) The 10.8M model in results/ is 6 layers, width 384.
# Cost has NOT been measured. Do the timing check first, then set -t from its median step time:
#   export SBATCH_ACCOUNT=<your project>
#   SIZE=large MAX_STEPS=50 EVAL=50 sbatch -q debug -t 00:15:00 scripts/nanogpt_large.sbatch    # timing check
#   SIZE=large sbatch scripts/nanogpt_large.sbatch                                              # 1 node
#   SIZE=large sbatch -N 2 scripts/nanogpt_large.sbatch                                         # 2 nodes (40 GB A100s)
#   SIZE=large PROFILE_STEP=100 sbatch -N 4 scripts/nanogpt_large.sbatch                        # 16 GPUs, one profiled step
# -N 4 costs 4x the node-hours of a 1-node job. PROFILE_STEP=k records a torch.profiler trace of step k on rank 0
# (profile_table.txt and trace_rank0.json in the run folder): compute vs all-reduce time for that step. Pick k after
# PowerSGD starts (step 10) and the step-time warm-up (20); that one step runs slower, the median step time does not
# change.
# Shakespeare is about 1 MB of text, so a model this size overfits quickly: compare step time and bytes first,
# and validation loss only with that caveat. BATCH lowers the per-GPU batch (default 64) if memory runs out.
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 01:00:00
#SBATCH -J gpt_large
#SBATCH -o logs/%x_%j.out

source "${PROJ:-$SCRATCH/powersgd}/scripts/common.sh"
SIZE=${SIZE:-large}
case $SIZE in
  medium) ARCH="--n-layer 8 --n-head 8 --n-embd 512" ;;
  large)  ARCH="--n-layer 12 --n-head 12 --n-embd 768" ;;
  *) echo "SIZE must be medium or large" >&2; exit 1 ;;
esac
SEED=${SEED:-1}
COMMON="--model nanogpt $ARCH --seed $SEED --max-steps ${MAX_STEPS:-1000} --eval-interval ${EVAL:-250} --start-iter 10"
[[ -n ${BATCH:-} ]] && COMMON="$COMMON --batch-size $BATCH"
[[ -n ${PROFILE_STEP:-} ]] && COMMON="$COMMON --profile-step $PROFILE_STEP"

start_gpu_monitor
launch $COMMON --mode ddp --run-name "gpt_${SIZE}_ddp_${SLURM_NNODES}node_s${SEED}_${SLURM_JOB_ID}"
for r in 2 4; do
  launch $COMMON --mode powersgd --rank "$r" --run-name "gpt_${SIZE}_r${r}_${SLURM_NNODES}node_s${SEED}_${SLURM_JOB_ID}"
done
finish
