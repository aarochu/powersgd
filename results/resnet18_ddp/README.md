# ResNet-18 on CIFAR-10 with our own trainer (issue #18)

`train_ddp.py --model resnet18` (11,173,962 parameters), 30 epochs, batch 128 per GPU (512 global), seed 1, on 1 node
(4 x A100 80 GB): DDP and PowerSGD rank 1, 2 and 4. Submitted with
`EPOCHS=30 sbatch -t 00:25:00 scripts/tier1_resnet18_1node.sbatch` from branch `experiments/tensor-parallel`
(commit fe9e8ae) on 2026-11-25. The two `timing_*` runs (2 epochs each) are the epoch-timing check from the interactive
smoke-test job, used to set the time limit.

**Not comparable with Table 3.** This trainer uses a 30-epoch cosine schedule, not the paper's 300-epoch schedule on
16 GPUs (`results/paper_reproduction/`). Compare DDP and PowerSGD within this job.

## Jobs

| Job | Runs | Nodes (GPUs) | Elapsed | State |
|---|---|---|---|---|
| 62006947 | DDP, r1, r2, r4, 30 epochs | 1 (4) | 00:08:07 | COMPLETED |
| 62006046 | interactive: smoke tests and the 2-epoch timing runs | 1 (4) | 00:29:20 | COMPLETED |

The 30-epoch job used 0.14 node-hours by elapsed time. Source: `jobs.txt` (`sacct`).

## Results (one seed)

| Method | Test accuracy, best (epoch) | Final | Median step | vs DDP | Bytes per step | vs DDP |
|---|---|---|---|---|---|---|
| DDP | 92.54% (29) | 92.49% | 15.3 ms | - | 44,695,848 | - |
| rank 1 | 87.33% (29) | 87.18% | 17.6 ms | +15.0% | 183,740 | 243.3x less |
| rank 2 | 91.09% (29) | 91.06% | 19.8 ms | +29.5% | 329,040 | 135.8x less |
| rank 4 | 91.88% (29) | 91.87% | 22.4 ms | +46.1% | 619,640 | 72.1x less |

- Best accuracy is 0.66 points below DDP for rank 4, 1.45 for rank 2 and 5.21 for rank 1, after the same 30 epochs.
- ResNet-18's steps are short (15 ms), so the compression work is a larger share of each step than for NanoGPT
  (where rank 4 costs +12% at 80 ms per step). Inside one NVLink node PowerSGD is slower per step at every rank.
- Rank 2 sends 135.8x less than DDP, the same ratio as the paper-code Gloo runs (`results/gloo_scaling/`, PR #8).
- ResNet-18's fp32 gradient is 44.70 MB per step (11,173,962 x 4 bytes), slightly more than the small NanoGPT's 43.18 MB.
- Mean GPU utilization was 71% (`gpu/`), lower than NanoGPT's 89-97%: with 15 ms steps, data loading and launch
  overhead show up.

## Figures

`figures/resnet18_curves.png` (test accuracy by step) and `figures/resnet18_bars.png` (step time and bytes per step).

## Files

`runs/<run>/` (`config.json`, `metrics.jsonl`, `step_times.csv`), `runs.jsonl`, `slurm-logs/`, `gpu/`, `jobs.txt`,
`gpu_usage_slurm.txt`. Copied with `perlmutter/bundle_results.sh` (scratch path, user name and account redacted). Every
`runs.jsonl` number was recomputed from the run's own files and matched.
