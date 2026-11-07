# Gloo scaling runs (slower communication backend)

PowerSGD paper code (`paper-code/run_experiment.py`) on Perlmutter with `--backend gloo`, on 8 and 16 GPUs: SGD against Rank 2.
CIFAR-10, ResNet-18, 10 epochs, seed 1. A 1-epoch, 4-GPU run is included as a smoke test.
The jobs ran on 2026-11-06 and were submitted through `perlmutter/run.sbatch` (regular queue, 30-minute limit; the smoke test used the debug queue).
Gloo here runs over Perlmutter's high-speed network interface (`GLOO_SOCKET_IFNAME=hsn0` in `run.sbatch`).
**This is a slower software backend on a fast network. It does not emulate the paper's 10 Gbit/s Ethernet.**

## Jobs

| Job | Run | Nodes (GPUs) | Elapsed | State |
|---|---|---|---|---|
| 60949651 | smoke: SGD, 1 epoch | 1 (4) | 00:00:37 | COMPLETED |
| 60951117 | SGD, 8 GPUs | 2 (8) | 00:01:13 | COMPLETED |
| 60951118 | SGD, 16 GPUs | 4 (16) | 00:01:02 | COMPLETED |
| 60951119 | Rank 2, 8 GPUs | 2 (8) | 00:00:59 | COMPLETED |
| 60951120 | Rank 2, 16 GPUs | 4 (16) | 00:00:50 | COMPLETED |

All exited with code 0. Together they used about 0.21 node-hours (0.83 GPU-hours) by elapsed time, smoke test included. Charged hours may differ.
Source: `gloo_jobs.txt` (`sacct`).

## Time per batch

From each run's `timer_summary.json` (the timer file the run wrote; other ranks' printed summaries in the `.out` logs differ by up to about 3%).

| GPUs | Method | Batch | Communication (`batch.reduce`) | Communication / batch | Forward + backward |
|---|---|---|---|---|---|
| 8 | SGD | 64.45 ms | 44.22 ms | 68.6% | 14.30 ms |
| 8 | Rank 2 | 32.31 ms (**-50%**) | 11.35 ms | 35.1% | 13.94 ms |
| 16 | SGD | 72.10 ms | 51.85 ms | 71.9% | 14.37 ms |
| 16 | Rank 2 | 38.36 ms (**-47%**) | 17.68 ms | 46.1% | 14.04 ms |

- Compute is the same for both methods (about 14 ms). The difference is all in communication.
- For Rank 2, `batch.reduce` includes the compression work as well as the smaller all-reduce.
- The `Frac.` column printed in the `.out` logs is the share of total run time including evaluation, not the share of a batch. Use the table above.
- For comparison, Table 3 (`results/paper_reproduction/`: NCCL, 16 GPUs, 300 epochs, 3 seeds) has Rank 2 **13% slower** than SGD (29.6 vs 26.2 ms per batch). That is a different experiment, not a matched baseline.

## Data sent

`bits` in `metrics.jsonl` is the cumulative communicated payload for the run. SGD: 85.8 billion bits at 16 GPUs (about 1.07 GB per epoch), Rank 2: 0.63 billion bits (about 7.9 MB per epoch), a ratio of 135.8x.
At 8 GPUs the totals are double (171.6e9 and 1.26e9) because the run takes twice as many steps.

## Accuracy at epoch 10 (test split)

| GPUs | Method | `last_accuracy` | `last_cross_entropy` | `runavg_accuracy` | `runavg_cross_entropy` |
|---|---|---|---|---|---|
| 8 | SGD | 66.2% | 1.079 | 31.6% | 1.895 |
| 8 | Rank 2 | 58.2% | 1.224 | 37.5% | 1.774 |
| 16 | SGD | 41.7% | 1.974 | 27.9% | 1.921 |
| 16 | Rank 2 | 43.1% | 1.545 | 25.3% | 2.092 |

- `last_*` is the model after the final step. `runavg_*` is the running-average model; `config.json` shows `average_reset_epoch_interval` 30, so at epoch 10 it averages weights from the start of training and is a poor indicator. This reading of `runavg` is mine, from the config.
- At 8 GPUs, Rank 2 is 8.0 points behind SGD. At 16 GPUs it is 1.4 points ahead. These are 10-epoch runs from a 300-epoch schedule, with a single seed, so neither gap is established.
- A faster batch is not faster training. Time to a target accuracy was not measured.

## GPU usage

`gloo_gpu_usage.txt` holds the raw Slurm accounting (`TRESUsageInAve`) for these jobs, in the same way as `results/paper_reproduction/gpu_usage.txt`.
These jobs have no `nvidia-smi` samples (`run.sbatch` does not start a monitor).

| Job | Run | `gpuutil` | `gpumem` |
|---|---|---|---|
| 60949651 | smoke, 4 GPUs | 15 | 2470M |
| 60951117 | SGD, 8 GPUs | 28 | 2470M |
| 60951118 | SGD, 16 GPUs | 0 | 1230080K (unit differs; looks like a missing sample) |
| 60951119 | Rank 2, 8 GPUs | 7 | 2468M |
| 60951120 | Rank 2, 16 GPUs | 12 | 2470M |

**Do not quote the utilization column as a measurement.** The jobs ran about a minute, Slurm samples coarsely, and one job reports 0. The memory figure (about 2.4 GB per GPU) is the more reliable column.

## Notes

- The logs contain many repeats of a PyTorch `UserWarning` from `paper-code/gradient_reducers.py` ("An output with one or more elements was resized ..."). It is a deprecation warning, not an error; the runs completed and wrote `DONE`.
- Single seed, 10 epochs. No NCCL run was made at these sizes in this experiment, so there is no matched baseline.
- Identifiers were redacted: the cluster scratch path in the logs reads `$SCRATCH`. Nothing else was changed.

## Files

| Path | Contents |
|---|---|
| `scaling/<run>/` | `config.json`, `metrics.jsonl`, `timer_summary.json`, `DONE` for each of the four runs |
| `smoke/cifar_sgd_w4_gloo_s1/` | the same files for the 1-epoch smoke test |
| `slurm-logs/` | the `.out` log of each job |
| `gloo_jobs.txt` | `sacct` job summary |
| `gloo_gpu_usage.txt` | `sacct` GPU accounting |
