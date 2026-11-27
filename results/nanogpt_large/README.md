# Larger NanoGPT models, 1 to 4 nodes, with a profiled step (issue #15)

`train_ddp.py --model nanogpt` at two larger sizes, batch 64 per GPU, seed 1: DDP, PowerSGD rank 2 and rank 4.
Submitted with `scripts/nanogpt_large.sbatch` from branch `experiments/tensor-parallel` (commits fe9e8ae and 4f408b1,
same training code) on 2026-11-25.

| Size | Layers / heads / width | Parameters | fp32 gradient |
|---|---|---|---|
| small (reference, `results/nanogpt_seeds/`) | 6 / 6 / 384 | 10,795,776 | 43.2 MB |
| medium | 8 / 8 / 512 | 25,417,728 | 101.7 MB |
| large | 12 / 12 / 768 | 85,352,448 | 341.4 MB |

## Jobs

| Job | Runs | Nodes (GPUs) | GPU | Elapsed | Node-hours | State |
|---|---|---|---|---|---|---|
| 62006927 | medium: DDP, r2, r4; 1000 steps | 1 (4) | A100 40 GB | 00:10:09 | 0.17 | COMPLETED |
| 62006924 | large: DDP, r2, r4; 1000 steps | 1 (4) | A100 40 GB | 00:28:51 | 0.48 | COMPLETED |
| 62010554 | large: DDP, r2, r4; 1000 steps | 2 (8) | A100 40 GB | 00:29:19 | 0.98 | COMPLETED |
| 62010560 | large: DDP, r2, r4; 1000 steps; step 100 profiled | 4 (16) | A100 40 GB | 00:29:01 | 1.93 | COMPLETED |
| 62010559 | large: DDP, r2, r4; 300 steps; step 100 profiled | 4 (16) | A100 80 GB | 00:09:20 | 0.62 | COMPLETED |

4.18 node-hours by elapsed time. Source: `jobs.txt` (`sacct`). Job 62010559 is a shorter 16-GPU run submitted
alongside 62010560; both are kept, and they agree (below).

## Step time and data per step

Median step time (steps after warm-up). Bytes are the logical all-reduce payload per GPU per step, not wire traffic.

| Size | GPUs | DDP | rank 2 | rank 4 |
|---|---|---|---|---|
| small | 4 | 79.7 ms | +6.1-6.7% | +12.0-12.2% |
| medium | 4 | 170.4 ms | 176.4 ms (+3.5%) | 182.8 ms (+7.3%) |
| large | 4 | 518.0 ms | 529.3 ms (+2.2%) | 537.3 ms (+3.7%) |
| large | 8 | 522.1 ms | 534.9 ms (+2.4%) | 542.2 ms (+3.9%) |
| large | 16 | 518.5 ms | 531.7 ms (+2.5%) | 540.0 ms (+4.1%) |
| large | 16 (job 62010559, 80 GB) | 511.7 ms | 524.5 ms (+2.5%) | 532.9 ms (+4.2%) |

| Size | DDP bytes per step | rank 2 | rank 4 |
|---|---|---|---|
| small | 43,183,104 | 430,096 (100.4x less) | 737,312 (58.6x less) |
| medium | 101,670,912 | 756,752 (134.4x less) | 1,296,416 (78.4x less) |
| large | 341,409,792 | 1,686,544 (202.4x less) | 2,887,712 (118.2x less) |

(The small-model row is seeds 2-3 on 1 node from `results/nanogpt_seeds/`, same GPU type.)

- **The cost of compressing shrinks as the model grows**: rank 4 goes from +12% per step on the 10.8M model to +7.3%
  on 25M and +3.7% on 85M, and the compression ratio grows.
- **DDP's step time does not grow from 4 to 16 GPUs** (518.0, 522.1, 518.5 ms), even though every step all-reduces
  341 MB per GPU and from 8 GPUs on that crosses nodes. On Perlmutter's network the gradient all-reduce is not a
  bottleneck at this size, so PowerSGD stays 2-4% slower per step at every node count.

## The profiled step (job 62010560, 16 GPUs: one step, step 100, on one rank, rank 0)

`profile_table.txt` in each 4-node run folder is PyTorch's per-operation summary; `trace_rank0.json.gz` is the full
trace (gzipped; open it in Perfetto or `chrome://tracing` after unzipping). From the traces, summing GPU kernel time:

| | DDP | rank 2 | rank 4 |
|---|---|---|---|
| GPU kernel time in the step | 528.9 ms | 524.3 ms | 530.8 ms |
| of which matrix multiplies (`aten::mm` + `aten::addmm`, from the table) | 451.7 ms (85%) | 451.7 ms | 451.7 ms |
| of which NCCL all-reduce kernels | 13.3 ms (2.5%), 13 kernels | 2.4 ms (0.5%), 39 kernels | 2.7 ms (0.5%), 39 kernels |
| QR orthogonalization (`aten::linalg_qr`, from the table) | - | not in the top 30 rows | 7.5 ms |
| First to last GPU kernel | 518.7 ms | 533.4 ms | 542.6 ms |

- Computation dominates: matrix multiplies are 85% of GPU kernel time. Gradient communication is 13.3 ms of GPU time
  for DDP.
- Measured: DDP's kernel time exceeds its first-to-last-kernel span by 10.2 ms (528.9 vs 518.7 ms), so kernels ran
  at the same time on different streams, communication alongside the backward pass. For rank 4 it is the other way
  round: the span exceeds the kernel time by 11.8 ms (542.6 vs 530.8 ms), time in which the GPU ran nothing.
- PowerSGD cuts the NCCL time to 2.4-2.7 ms (about 10.6 ms saved) but the step gets 14.7 ms (rank 2) and 23.9 ms
  (rank 4) longer from first to last kernel. Rank 4 spends 7.5 ms on QR alone. Our reading of the idle time is that
  PowerSGD's steps (compress, all-reduce P, orthogonalize, all-reduce Q) run in sequence and overlap less with
  backward than DDP's single all-reduce; the profile shows the gaps but does not prove their cause.
- The 300-step job (62010559, 80 GB nodes) shows the same: NCCL 13.0 ms for DDP against 2.0-2.5 ms for PowerSGD.
  Every 4-node run's numbers are in its `profile_table.txt`.

## Validation loss (with a caveat)

| Size | GPUs | DDP best @ step | rank 2 best @ step | rank 4 best @ step |
|---|---|---|---|---|
| medium | 4 | 1.502 @ 500 (final 1.670) | 2.160 @ 1000 | 1.759 @ 1000 |
| large | 4 | 1.517 @ 500 (final 2.191) | 2.236 @ 1000 | 1.782 @ 1000 |
| large | 8 | 1.580 @ 500 | 2.257 @ 1000 | 1.784 @ 1000 |
| large | 16 | 1.638 @ 250 | 2.237 @ 1000 | 1.758 @ 1000 |

Shakespeare is about 1 MB of text, so these models overfit quickly, and sooner with more GPUs (a larger global batch
sees the data faster): DDP peaks at step 500 on 4 and 8 GPUs and at step 250 on 16. PowerSGD is still improving at
step 1000. Read these as "PowerSGD needs more steps", not as final quality. Job 62010559 stopped at 300 steps with one
evaluation, so its losses are not compared.

## Figures

- `figures/large_nodes_bars.png`: step time and bytes per step for every large-model run, 1 to 4 nodes.
- `figures/large_nodes_curves.png`: validation loss by step for the large model on 4, 8 and 16 GPUs (1000-step jobs).
- `figures/large_bars.png`, `figures/large_curves.png`: the 1-node medium and large runs.

## Files

`runs/<run>/` (`config.json`, `metrics.jsonl`, `step_times.csv`; the 4-node runs also `profile_table.txt` and
`trace_rank0.json.gz`), `runs.jsonl`, `slurm-logs/`, `gpu/` (`nvidia-smi` every second, one file per node),
`jobs.txt`, `gpu_usage_slurm.txt`. Copied with `perlmutter/bundle_results.sh` (scratch path, user name and account
redacted). Every `runs.jsonl` number was recomputed from the run's own files and matched. The traces were gzipped
after copying (8.3 MB to 0.5 MB for the largest); nothing else was changed.
