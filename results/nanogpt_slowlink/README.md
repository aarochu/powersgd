# NanoGPT on a modeled 10 Gbit/s link (issue #14)

`train_ddp.py --model nanogpt` (10.8M parameters, 2000 steps, batch 64 per GPU, seed 1) on 1 node (4 x A100 40 GB):
DDP, PowerSGD rank 2 and rank 4, each with no modeled link and with `--sim-bandwidth-gbps 10`. Submitted with
`scripts/nanogpt_slowlink.sbatch` from branch `experiments/tensor-parallel` (commit fe9e8ae) on 2026-11-25.

**This is a model of a slow network, not a measurement of one.** After each all-reduce the trainer waits for the GPU,
then sleeps for `2(N-1)/N x bytes / bandwidth` (the per-GPU traffic of a ring all-reduce at 10 Gbit/s). No overlap
with compute and no contention are modeled, and the wait itself removes DDP's usual compute/communication overlap.
Compare methods on the same modeled link.

## Job

| Job | Runs | Nodes (GPUs) | Elapsed | State |
|---|---|---|---|---|
| 62006920 | 3 methods x 2 links | 1 (4) | 00:21:39 | COMPLETED |

0.36 node-hours by elapsed time. Source: `jobs.txt` (`sacct`).

## Step time

| Method | Bytes per step | No modeled link | Modeled 10 Gbit/s | Added by the link |
|---|---|---|---|---|
| DDP | 43,183,104 | 79.7 ms | 133.9 ms | +54.2 ms (the model predicts at least 51.8) |
| rank 2 | 430,096 | 84.9 ms (+6.5% vs DDP) | 88.5 ms (**-33.9% vs DDP**) | +3.6 ms |
| rank 4 | 737,312 | 89.5 ms (+12.2% vs DDP) | 91.0 ms (**-32.0% vs DDP**) | +1.5 ms |

On the modeled link PowerSGD steps are about a third faster than DDP's.

## Validation loss against training time

Training time (`train_time_s` in each eval line; evaluation excluded). Evaluated every 250 steps.

| Link | Method | Best val @ step | Training time to best | First eval at or below 1.55 | First eval at or below 1.51 |
|---|---|---|---|---|---|
| none | DDP | 1.480 @ 750 | 61 s | 41 s (step 500) | 41 s (step 500) |
| none | rank 2 | 1.759 @ 2000 | 172 s | never | never |
| none | rank 4 | 1.507 @ 2000 | 182 s | 159 s (step 1750) | 182 s (step 2000) |
| 10 Gbit/s | DDP | 1.476 @ 750 | 101 s | 68 s (step 500) | 68 s (step 500) |
| 10 Gbit/s | rank 2 | 1.758 @ 2000 | 180 s | never | never |
| 10 Gbit/s | rank 4 | 1.503 @ 2000 | 185 s | 162 s (step 1750) | 185 s (step 2000) |

**PowerSGD does not reach DDP's best loss sooner on the modeled 10 Gbit/s link.** Its steps are faster, but it needs
far more of them: DDP is at 1.502 after 68 s of training and at its best (1.476) after 101 s, while rank 4 first gets
below 1.55 after 162 s and ends at 1.503 after 185 s; rank 2 never gets below 1.75 in 2000 steps. At this link speed
and model size the per-step saving (about 43 ms) does not make up for PowerSGD's slower progress per step. A slower
modeled link, or a larger model, would move the break-even point; neither was run.

## Figures

- `figures/slowlink_vs_time.png`: validation loss by wall time (evaluation included) for all six runs.
- `figures/slowlink_bars.png`: step time and bytes per step.

## Files

`runs/<run>/` (`config.json`, `metrics.jsonl`, `step_times.csv`), `runs.jsonl` (the `sim_link` field holds the modeled
bandwidth, latency and total modeled delay), `slurm-logs/`, `gpu/`, `jobs.txt`, `gpu_usage_slurm.txt`. Copied with
`perlmutter/bundle_results.sh` (scratch path, user name and account redacted). Every `runs.jsonl` number was recomputed
from the run's own files and matched.
