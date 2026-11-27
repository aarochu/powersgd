# NanoGPT with lower-precision gradients (issue #13)

`train_ddp.py --model nanogpt --precision ...` (10.8M parameters, 2000 steps, batch 64 per GPU, seed 1) on 1 node
(4 x A100 40 GB): DDP with fp32, fp16, bf16 and emulated fp8 (e4m3, e5m2) gradients, and PowerSGD rank 4 with fp32 and
fp16. Submitted with `scripts/nanogpt_precision.sbatch` from branch `experiments/tensor-parallel` (commit fe9e8ae) on
2026-11-25.

## Job

| Job | Runs | Nodes (GPUs) | Elapsed | State |
|---|---|---|---|---|
| 62006918 | 7 configs | 1 (4) | 00:22:20 | COMPLETED |

0.37 node-hours by elapsed time. Source: `jobs.txt` (`sacct`).

## Results (one seed, all in the same job)

| Method | Gradient precision | Bytes per step | vs DDP fp32 | Median step | vs DDP fp32 | Best val @ step |
|---|---|---|---|---|---|---|
| DDP | fp32 | 43,183,104 | - | 79.7 ms | - | 1.483 @ 750 |
| DDP | fp16 | 21,591,552 | 2.0x less | 79.8 ms | +0.0% | 1.479 @ 750 |
| DDP | bf16 | 21,591,552 | 2.0x less | 79.8 ms | +0.0% | 1.480 @ 750 |
| DDP | fp8 e4m3 (emulated) | 10,795,776 computed (43,183,104 logged) | 4.0x less, computed | 80.2 ms | +0.6% | 1.482 @ 750 |
| DDP | fp8 e5m2 (emulated) | 10,795,776 computed (43,183,104 logged) | 4.0x less, computed | 80.2 ms | +0.5% | 1.482 @ 750 |
| PowerSGD rank 4 | fp32 | 737,312 | 58.6x less | 89.4 ms | +12.1% | 1.505 @ 2000 |
| PowerSGD rank 4 | fp16 | 368,656 | 117.1x less | 88.0 ms | +10.4% | 1.503 @ 2000 |

- **fp16 and bf16 halve the payload with no measurable effect on loss** (1.479 and 1.480 against 1.483; the seed
  spread for DDP in `results/nanogpt_seeds/` is 1.475-1.488). They do not change step time, because on one NVLink
  node the all-reduce is not the bottleneck.
- **fp8 is emulated.** The A100 has no fp8: each gradient bucket is scaled, cast to fp8 and back, then sent as fp32.
  Its effect on the loss is real (none visible here, 1.482 for both formats); its logged bytes and step time are
  fp32's. The fp8 payload in the table is computed as one quarter of fp32, not measured.
- **PowerSGD on fp16 buckets works** and halves PowerSGD's payload again (117x less than DDP fp32), with the same loss
  as rank 4 on fp32. It is also 1.6% faster per step than rank 4 on fp32 (88.0 vs 89.4 ms).

## Figures

`figures/precision_bars.png` (step time and bytes per step) and `figures/precision_curves.png` (validation loss by step).

## Files

`runs/<run>/` (`config.json`, `metrics.jsonl`, `step_times.csv`), `runs.jsonl`, `slurm-logs/`, `gpu/` (`nvidia-smi`
every second), `jobs.txt`, `gpu_usage_slurm.txt`. Copied with `perlmutter/bundle_results.sh` (scratch path, user name
and account redacted). Every `runs.jsonl` number was recomputed from the run's own files and matched.
