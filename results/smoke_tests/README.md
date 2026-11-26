# GPU smoke tests for the new experiment code (issue #11)

One interactive node (job 62006046, 4 x A100, 2026-11-25, 29 minutes, 0.49 node-hours) before any batch job, on
branch `experiments/tensor-parallel`. Synthetic or short runs: these check the code, they are not results.

| File | What ran | Outcome |
|---|---|---|
| `smoke_next.txt` | `scripts/smoke_next.sh` at commit 3826918: 12 configs (fp32, fp16, bf16, fp8, PowerSGD on fp16, modeled 10 Gbit/s link, medium and large model), 40 steps | all 12 `ok`; **bug**: the modeled link added no time (78.2 ms with and without it; expected +51.8 ms) |
| `smoke_tp.txt` | `scripts/smoke_tp.sh` at 3826918 | tp 4 failed by design (6 heads cannot be split 4 ways); the loss checks could not fail (random tokens end at ln 65 = 4.174, and tp 1 and tp 2 were scored on different batches) |
| `smoke_tp2.txt` | `scripts/smoke_tp.sh` at fe9e8ae (rewritten) | all `PASS`: tp 1 matches `train_ddp.py` (5.4e-5); loss before training matches the unsplit model (1.2e-8 at tp 2, 3.6e-8 at tp 4); after 40 steps with dropout off (7.0e-7, 1.4e-4); loss fell 4.30 to 3.31-3.34; tensor- and data-parallel bytes equal their formulas exactly |
| `time_epoch.txt` | `scripts/time_epoch_1node.sh`: ResNet-18, real CIFAR-10, 2 epochs | DDP 15.6 ms per step, rank 2 19.8 ms; 7-8 s per 2 epochs, used to set the 30-epoch job's limit |
| `smoke_next2.txt` | `scripts/smoke_next.sh` at fe9e8ae | all 12 `ok`; `PASS` modeled 10 Gbit/s link: +54.2 ms per step (expected at least 51.8) |

The two fixes (commits 0cfc696 and fe9e8ae) are described on PRs #9 and #10. `smoke_*_runs.jsonl` are the runs'
summary records (`smoke_tp_runs_before_*.jsonl` is the first `smoke_tp.sh` run, kept when the second started a fresh
log). `queued_jobs_1126.txt` lists the batch jobs submitted after the tests passed. The interactive job's output went
to the terminal, not a Slurm log, so the `.txt` files are what `tee` saved.
