# Larger NanoGPT models, 1 node (part of issue #15)

`train_ddp.py --model nanogpt` at two larger sizes, 1000 steps, batch 64 per GPU, seed 1, on 1 node (4 x A100 40 GB):
DDP, PowerSGD rank 2 and rank 4. Submitted with `scripts/nanogpt_large.sbatch` (`SIZE=large`, `SIZE=medium -t 00:30:00`)
from branch `experiments/tensor-parallel` (commit fe9e8ae) on 2026-11-25.

| Size | Layers / heads / width | Parameters | fp32 gradient |
|---|---|---|---|
| small (reference, `results/nanogpt_seeds/`) | 6 / 6 / 384 | 10,795,776 | 43.2 MB |
| medium | 8 / 8 / 512 | 25,417,728 | 101.7 MB |
| large | 12 / 12 / 768 | 85,352,448 | 341.4 MB |

**Not run yet** (the rest of issue #15): the large model on 2 nodes, and the 16-GPU run with one profiled step. The
issue stays open for those.

## Jobs

| Job | Runs | Nodes (GPUs) | Elapsed | State |
|---|---|---|---|---|
| 62006924 | large: DDP, r2, r4 | 1 (4) | 00:28:51 | COMPLETED |
| 62006927 | medium: DDP, r2, r4 | 1 (4) | 00:10:09 | COMPLETED |

0.65 node-hours by elapsed time. Source: `jobs.txt` (`sacct`).

## Step time and data per step

| Size | Method | Bytes per step | vs DDP | Median step | vs DDP |
|---|---|---|---|---|---|
| small | rank 2 / rank 4 | 430,096 / 737,312 | 100.4x / 58.6x less | 84.5-85.0 / 89.3-89.4 ms | +6.1-6.7% / +12.0-12.2% |
| medium | DDP | 101,670,912 | - | 170.4 ms | - |
| medium | rank 2 | 756,752 | 134.4x less | 176.4 ms | +3.5% |
| medium | rank 4 | 1,296,416 | 78.4x less | 182.8 ms | +7.3% |
| large | DDP | 341,409,792 | - | 518.0 ms | - |
| large | rank 2 | 1,686,544 | 202.4x less | 529.3 ms | +2.2% |
| large | rank 4 | 2,887,712 | 118.2x less | 537.3 ms | +3.7% |

(The small-model row is seeds 2-3 on 1 node from `results/nanogpt_seeds/`, same GPU type.)

**The cost of compressing shrinks as the model grows**: rank 4 goes from +12% per step on the 10.8M model to +7.3% on
25M and +3.7% on 85M, and the compression ratio grows. On one NVLink node DDP is still faster per step at every size.

## Validation loss (with a caveat)

| Size | DDP best @ step (final) | rank 2 best @ step | rank 4 best @ step |
|---|---|---|---|
| medium | 1.502 @ 500 (1.670 at 1000) | 2.160 @ 1000 | 1.759 @ 1000 |
| large | 1.517 @ 500 (2.191 at 1000) | 2.236 @ 1000 | 1.782 @ 1000 |

Shakespeare is about 1 MB of text, so these models overfit quickly: DDP peaks at step 500 and then gets worse.
PowerSGD is still improving at step 1000. Read these as "PowerSGD needs more steps", not as final quality.

## Figures

`figures/large_bars.png` (step time and bytes per step, log scale) and `figures/large_curves.png` (validation loss by step).

## Files

`runs/<run>/` (`config.json`, `metrics.jsonl`, `step_times.csv`), `runs.jsonl`, `slurm-logs/`, `gpu/` (`nvidia-smi`:
mean GPU utilization 97% for large, 91% for medium; at most 14.7 GB memory), `jobs.txt`, `gpu_usage_slurm.txt`. Copied
with `perlmutter/bundle_results.sh` (scratch path, user name and account redacted). Every `runs.jsonl` number was
recomputed from the run's own files and matched.
