# Tensor vs data parallelism, measured (issue #19)

`train_tensor.py` (NanoGPT, 10.8M parameters, 2000 steps, seed 1) with Megatron-style tensor parallelism inside a node
and data parallelism (DDP, optionally PowerSGD rank 4) across the tensor-parallel groups. Submitted with
`scripts/nanogpt_tp.sbatch` from branch `experiments/tensor-parallel` (commit fe9e8ae) on 2026-11-25, after
`scripts/smoke_tp.sh` passed (`results/smoke_tests/`). All on 40 GB A100s.

- 1 node, 6 heads (the model in `results/`): tp 1 and tp 2, each with DDP and PowerSGD rank 4.
- 2 nodes, 12 heads: tp 1 and tp 4, each with DDP and rank 4. Tensor-parallel size must divide the head count, so tp 4
  uses 12 heads (head width 32; same parameter count) with its own 12-head tp 1 baseline. Here the two
  data-parallel replicas sit on different nodes, so the gradient all-reduce crosses the network between nodes while
  the activation all-reduces stay inside each node on NVLink.

Every config trains on 64 sequences per GPU per step (per-replica batch 64 x tp), so configs on the same number of
GPUs have the same global batch.

## Jobs

| Job | Runs | Nodes (GPUs) | Elapsed | State |
|---|---|---|---|---|
| 62006949 | tp1 / tp2 x DDP / r4, 6 heads | 1 (4) | 00:15:06 | COMPLETED |
| 62006950 | tp1 / tp4 x DDP / r4, 12 heads | 2 (8) | 00:16:49 | COMPLETED |

0.81 node-hours by elapsed time (0.25 + 2 x 0.28). Source: `jobs.txt` (`sacct`).

## Results (one seed)

Bytes are the logical all-reduce payload per GPU per step: DP = data-parallel gradients, TP = tensor-parallel activations.

| Setup | Config | Params per GPU | Median step | vs tp 1 DDP | DP bytes | TP bytes | Best val @ step |
|---|---|---|---|---|---|---|---|
| 1 node, 6 heads | tp 1, DDP | 10,795,776 | 79.7 ms | - | 43.18 MB | 0 | 1.481 @ 750 |
| 1 node, 6 heads | tp 1, rank 4 | 10,795,776 | 89.4 ms | +12.1% | 0.74 MB | 0 | 1.505 @ 2000 |
| 1 node, 6 heads | tp 2, DDP | 5,479,296 | 106.1 ms | +33.1% | 21.92 MB | 1,207.96 MB | 1.480 @ 750 |
| 1 node, 6 heads | tp 2, rank 4 | 5,479,296 | 114.9 ms | +44.2% | 0.48 MB | 1,207.96 MB | 1.497 @ 2000 |
| 2 nodes, 12 heads | tp 1, DDP | 10,795,776 | 83.7 ms | - | 43.18 MB | 0 | 1.479 @ 500 |
| 2 nodes, 12 heads | tp 1, rank 4 | 10,795,776 | 93.7 ms | +11.9% | 0.74 MB | 0 | 1.499 @ 2000 |
| 2 nodes, 12 heads | tp 4, DDP | 2,821,056 | 123.5 ms | +47.5% | 11.28 MB | 2,415.92 MB | 1.493 @ 500 |
| 2 nodes, 12 heads | tp 4, rank 4 | 2,821,056 | 131.8 ms | +57.4% | 0.36 MB | 2,415.92 MB | 1.470 @ 2000 |

- **Tensor parallelism moves far more data than data parallelism**: 28x more per GPU at tp 2 (1.21 GB of activations
  plus 21.9 MB of gradients, against 43.2 MB of gradients) and 56x more at tp 4. Activation traffic is
  4 x layers x batch x sequence x width x 4 bytes, and the measured counts equal that formula exactly. At this model
  size it only costs time: tp 2 is 33% slower per step than plain data parallelism, tp 4 48% slower.
- **PowerSGD only shrinks the gradient part.** At tp 2 it cuts the gradients from 21.9 MB to 0.48 MB (45x), but the
  1.21 GB of activations is untouched, and the step gets 8% slower (106.1 to 114.9 ms). At tp 4 on 2 nodes it cuts
  the gradient all-reduce between nodes from 11.3 MB to 0.36 MB (32x) for a 7% slower step.
- Measured numbers match the computed ones in `docs/tensor_parallelism.md`: 1.21 GB of activations and 21.9 MB of
  gradients per GPU per step at tp 2.
- The tp 4 + rank 4 run has the lowest best loss here (1.470), but that is one seed, at the last eval, against DDP runs
  that peak early and overfit; it is not evidence that the combination trains better.
- A 10.8M-parameter model fits on one GPU, so tensor parallelism is not needed here; this measures its communication
  cost, not a case where it pays off.

## Figures

`figures/tp_bars.png` (step time, and DP and TP bytes stacked on a log scale) and `figures/tp_curves.png` (validation
loss by step).

## Files

`runs/<run>/` (`config.json`, `metrics.jsonl`, `step_times.csv` with `dp_bytes` and `tp_bytes` per step), `runs.jsonl`
(`tp_size`, `dp_size`, `params_per_gpu`, `tp_bytes_per_step`), `slurm-logs/`, `gpu/`, `jobs.txt`,
`gpu_usage_slurm.txt`. Copied with `perlmutter/bundle_results.sh` (scratch path, user name and account redacted).
Every `runs.jsonl` number was recomputed from the run's own files and matched.
