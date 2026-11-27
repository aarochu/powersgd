# NanoGPT across seeds (issue #12)

`train_ddp.py --model nanogpt` (10.8M parameters, Shakespeare characters, 2000 steps, batch 64 per GPU) with seeds 2
and 3, on 1 node (4 GPUs) and 2 nodes (8 GPUs): DDP and PowerSGD rank 1, 2 and 4. Seed 1 is the earlier runs already on
`master` in `results/organized/`. Submitted with `scripts/nanogpt_seeds.sbatch` from branch `experiments/tensor-parallel`
(commit fe9e8ae) on 2026-11-25.

## Jobs

| Job | Runs | Nodes (GPUs) | Elapsed | State |
|---|---|---|---|---|
| 62006912 | seeds 2, 3 x DDP, r1, r2, r4 | 1 (4) | 00:25:54 | COMPLETED |
| 62006917 | seeds 2, 3 x DDP, r1, r2, r4 | 2 (8) | 00:26:17 | COMPLETED |

1.31 node-hours by elapsed time (0.43 + 2 x 0.44). Source: `jobs.txt` (`sacct`). Charged hours may differ.

## Best validation loss over seeds 1-3

Mean [min, max] of each run's best validation loss (evaluated every 250 steps), and the step where it was reached.

| Setup | Method | Seeds | Best val, mean [min, max] | Per seed (best @ step) |
|---|---|---|---|---|
| 1 node | DDP | 3 | 1.481 [1.475, 1.488] | s1 1.481 @ 750, s2 1.488 @ 750, s3 1.475 @ 750 |
| 1 node | rank 1 | 3 | 2.307 [2.277, 2.335] | s1 2.277 @ 2000, s2 2.310 @ 2000, s3 2.335 @ 2000 |
| 1 node | rank 2 | 3 | 1.765 [1.762, 1.767] | s1 1.767 @ 2000, s2 1.762 @ 2000, s3 1.766 @ 2000 |
| 1 node | rank 4 | 3 | 1.509 [1.505, 1.511] | s1 1.505 @ 2000, s2 1.511 @ 2000, s3 1.511 @ 2000 |
| 2 nodes | DDP | 3 | 1.488 [1.477, 1.495] | s1 1.490 @ 500, s2 1.495 @ 500, s3 1.477 @ 500 |
| 2 nodes | rank 1 | 2 | 2.262 [2.249, 2.275] | s2 2.249 @ 2000, s3 2.275 @ 2000 (no seed-1 run) |
| 2 nodes | rank 2 | 3 | 1.754 [1.748, 1.758] | s1 1.756 @ 2000, s2 1.758 @ 2000, s3 1.748 @ 2000 |
| 2 nodes | rank 4 | 3 | 1.499 [1.494, 1.504] | s1 1.494 @ 2000, s2 1.504 @ 2000, s3 1.500 @ 2000 |

- The seed spread is small next to the gaps between methods, so the ordering from the single-seed runs holds:
  DDP < rank 4 < rank 2 < rank 1.
- DDP reaches its best early (step 750 on 4 GPUs, 500 on 8, where the global batch is twice as large) and then
  overfits. PowerSGD is still improving at step 2000, so "best" compares a finished DDP run with an unfinished
  PowerSGD one. The earlier 6000-step run (`results/organized/job60623330_1node_batch/`) has rank 4 reaching 1.475.
- Seed 1 on 2 nodes uses job 60623329. Job 60627319 is a repeat of the same seed-1 configuration (DDP 1.489, r2 1.755,
  r4 1.500); it is not counted as a separate seed.

## Step time and data per step

Median step time (steps after warm-up). Bytes are the logical all-reduce payload per GPU per step, not wire traffic.

| Method | Bytes per step | vs DDP | 1 node, seeds 2/3 (40 GB A100) | 1 node, seed 1 (80 GB A100) | 2 nodes, seeds 1/2/3 (40 GB) |
|---|---|---|---|---|---|
| DDP | 43,183,104 | - | 79.7 / 79.7 ms | 78.0 ms | 79.8 / 79.8 / 79.8 ms |
| rank 1 | 276,488 | 156.2x less | 83.0 / 82.8 ms (+4.1% / +3.9%) | 81.1 ms | - / 83.1 / 83.1 ms |
| rank 2 | 430,096 | 100.4x less | 84.5 / 85.0 ms (+6.1% / +6.7%) | 83.3 ms | 85.1 / 85.3 / 85.3 ms |
| rank 4 | 737,312 | 58.6x less | 89.3 / 89.4 ms (+12.0% / +12.2%) | 87.8 ms | 89.6 / 89.7 / 89.6 ms |

Seed 1 at 1 node ran on 80 GB A100s and the new 1-node runs on 40 GB A100s (`device` in each `metrics.jsonl` start
line), so compare step times within a column. On every setup PowerSGD steps are 4-13% slower than DDP: on Perlmutter's
network the all-reduce is cheap, and the compression work costs more than the bytes it saves.

## Figures

- `figures/seeds_spread.png`: best validation loss per configuration, seeds 2 and 3 (the plotting tool groups runs by
  name, and seed 1 has older names; its values are in the table above).
- `figures/seeds_curves_1node.png`: validation loss by step for the 1-node runs.

## Files

- `runs/<run>/`: `config.json`, `metrics.jsonl` (start, eval and done lines), `step_times.csv` (every step).
- `runs.jsonl`: the trainer's summary record for each run. `slurm-logs/`: the jobs' output. `gpu/`: `nvidia-smi`
  samples every second (mean GPU utilization 89-90%, at most 5.6 GB memory). `gpu_usage_slurm.txt`: Slurm accounting.
- Copied with `perlmutter/bundle_results.sh`, which redacts the scratch path, user name and account. Every number in
  each `runs.jsonl` record was recomputed from that run's `metrics.jsonl` and `step_times.csv` and matched.
