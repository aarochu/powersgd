# Next experiments: what they answer and how to run them

Code for the deck's next steps and the gaps in the current results. **None of these have been run yet.** Costs are
estimates from earlier jobs or time-limit caps, labeled as such. Anything that could exceed 2 node-hours is marked:
agree on it before submitting.

| # | Question | How | Nodes | Cap (time limit) | Expected use |
|---|---|---|---|---|---|
| 1 | Are the NanoGPT gaps bigger than seed noise? | `scripts/nanogpt_seeds.sbatch` | 1 (or 2) | 0.75 node-h (1.5) | about 0.4-0.5 (0.8-1.0) |
| 2 | Lower-precision gradients (fp16, bf16, fp8) | `scripts/nanogpt_precision.sbatch` | 1 | 0.67 node-h | about 0.35-0.45 |
| 3 | PowerSGD on a modeled 10 Gbit/s link | `scripts/nanogpt_slowlink.sbatch` | 1 | 0.75 node-h | about 0.35 |
| 4 | Larger models (25M and 85M parameters), up to 16 GPUs with one profiled step | `scripts/nanogpt_large.sbatch` | 1, 2 or 4 | 1-4 node-h per job | not measured; timing check first |
| 5 | Matched NCCL baseline for the Gloo runs | `perlmutter/submit_scaling.sh` | 2-4 | 2.0 node-h | small (Gloo versions took 50-73 s) |
| 6 | Rest of the scaling curve (1, 2, 4 GPUs) | `perlmutter/submit_scaling.sh` | 1 | 2.0 node-h | small |
| 7 | More seeds for the Gloo runs | `perlmutter/submit_scaling.sh` | 2-4 | 4.0 node-h | about 0.4 |
| 8 | Time to accuracy under Gloo (full 300 epochs) | `perlmutter/submit_scaling.sh` | 4 | 12 node-h | **about 3.6, over 2: agree first** |
| 9 | Data vs tensor parallelism | [`docs/tensor_parallelism.md`](tensor_parallelism.md) | - | - | write-up, no compute |
| 10 | ResNet-18 trained with our own DDP trainer | `scripts/time_epoch_1node.sh`, then `scripts/tier1_resnet18_1node.sbatch` | 1 | 0.33 (timing) + 1.0 (placeholder) | not measured; timing first |

Experiments 1-4 and 10 use `train_ddp.py` (NanoGPT, and ResNet-18 for 10). 5-8 use the paper code
(`paper-code/run_experiment.py`, CIFAR-10 ResNet-18), the same code as Table 3 and `results/gloo_scaling/`.

## 0. Before anything

On a Perlmutter login node:

```bash
cd $SCRATCH/powersgd && git pull
export SBATCH_ACCOUNT=<your project>     # read by sbatch for the scripts/*.sbatch jobs (they have no -A line)
export NERSC_ACCOUNT=<your project>      # read by perlmutter/submit_scaling.sh
```

Data is already in `$SCRATCH/powersgd/data` if `python tools/prepare_data.py` and `bash perlmutter/setup.sh` were run
once. Submit everything from `$SCRATCH/powersgd` (the scripts write `logs/` relative to it).

**Run the smoke test first.** The trainer patch (`best_step`, `lr_eff` logging) and the new `--sim-bandwidth-gbps` flag
have only been checked on a laptop without a GPU, and the fp16/bf16/fp8 hooks have not run on Perlmutter:

```bash
salloc -N 1 -C gpu -G 4 -q interactive -t 00:20:00 -A $SBATCH_ACCOUNT
bash scripts/smoke_next.sh
```

It runs 12 short configurations on synthetic data and prints a table. Check that every row says `ok`, that
`smoke_large_*` fit in memory, and the last line: the measured extra step time for the modeled 10 Gbit/s link should be
close to the expected value it prints. `smoke_r4_fp16` (PowerSGD on fp16 buckets) is allowed to fail; if it does, drop
`powersgd:4:fp16` from experiment 2.

## 1. NanoGPT seeds

```bash
sbatch scripts/nanogpt_seeds.sbatch                 # seeds 2 and 3: DDP, PowerSGD rank 1, 2, 4, 2000 steps
```

```bash
sbatch -N 2 scripts/nanogpt_seeds.sbatch            # the same on 2 nodes (8 GPUs)
```

Runs are named `gpt_<mode>_s<seed>_<nodes>node_<jobid>`. With seed 1 from `results/`, this gives 3 seeds per
configuration. 1-node jobs have run on 80 GB A100s and 2-node jobs on 40 GB A100s, matching the earlier runs of each
setup; check the `device` in each run's `metrics.jsonl` start line before pooling seeds.

## 2. Gradient precision

```bash
sbatch scripts/nanogpt_precision.sbatch             # DDP fp32/fp16/bf16/fp8_e4m3/fp8_e5m2, PowerSGD r4 fp32/fp16
```

fp16 and bf16 really halve the payload. **fp8 is emulated**: the gradient is quantized to fp8 and back, then sent as
fp32 (A100 has no fp8). The loss curve is a real test of fp8 numerics, but the logged bytes and step time are fp32's;
report fp8's payload as computed (one quarter of fp32), not measured.

## 3. Modeled slow link

```bash
sbatch scripts/nanogpt_slowlink.sbatch              # each method with no delay and at 10 Gbit/s, same job
SIM_BWS="0 1 10 25" sbatch scripts/nanogpt_slowlink.sbatch
```

With `--sim-bandwidth-gbps B`, `train_ddp.py` sleeps for `2(N-1)/N x bytes / B` after every all-reduce (plus
`--sim-latency-us`). That is the per-GPU traffic of a ring all-reduce on a link of that speed, with **no overlap with
compute and no contention**. Report it as "modeled 10 Gbit/s link", not as a measured network. At the default 0 the
trainer is unchanged. The modeled delay is recorded per run in `runs.jsonl` (`sim_link.modeled_delay_s_total`).

## 4. Larger NanoGPT

```bash
SIZE=large MAX_STEPS=50 EVAL=50 sbatch -q debug -t 00:15:00 scripts/nanogpt_large.sbatch   # timing check
SIZE=large sbatch scripts/nanogpt_large.sbatch                                              # DDP, r2, r4, 1000 steps
SIZE=medium sbatch scripts/nanogpt_large.sbatch
SIZE=large sbatch -N 2 scripts/nanogpt_large.sbatch                                         # 2 nodes
SIZE=large PROFILE_STEP=100 sbatch -N 4 scripts/nanogpt_large.sbatch                        # 16 GPUs, profile step 100
```

The 16-GPU job costs 4x the node-hours of a 1-node job, so it is the one most likely to exceed 2 node-hours: price it
from the timing check and agree on it first. `PROFILE_STEP` records one step on rank 0 with `torch.profiler`
(`profile_table.txt`, `trace_rank0.json` in the run folder), which shows how much of a step is compute and how much is
all-reduce; it works with any `-N`.

`medium` is 8 layers, width 512 (about 25M parameters, about 100 MB gradient); `large` is 12 layers, width 768 (about
85M, about 340 MB). From the timing check's median step time, set `-t` to about 3 x 1000 steps x step time plus 10
minutes, and check the node-hours before submitting (2-node jobs count double). Shakespeare is about 1 MB of text, so
these models overfit early: compare step time and bytes first, validation loss only with that caveat. 2-node jobs have
landed on 40 GB A100s, so compare step times within a job, not 1-node against 2-node.

## 5-8. Paper code: scaling and Gloo

`perlmutter/submit_scaling.sh` submits any slice of the scaling grid with short time limits, using the same run names
and folders as `submit_paper.sh`, so finished runs (`DONE`) and queued runs are skipped. It turns on a 1 Hz
`nvidia-smi` CSV per node (`GPU_MONITOR=1`), because Slurm's GPU accounting is unreliable for one-minute jobs. Always
dry-run first: it lists the jobs and the node-hour cap.

```bash
DRY_RUN=1 BACKENDS=nccl WORKERS="8 16" ./perlmutter/submit_scaling.sh

# 5. matched NCCL baseline at 8 and 16 GPUs (SGD and Rank 2, 10 epochs, seed 1)
BACKENDS=nccl WORKERS="8 16" ./perlmutter/submit_scaling.sh

# 6. rest of the scaling curve, both backends
BACKENDS="nccl gloo" WORKERS="1 2 4" ./perlmutter/submit_scaling.sh

# 7. more seeds for the Gloo comparison
BACKENDS=gloo WORKERS="8 16" SEEDS="2 3" ./perlmutter/submit_scaling.sh

# 8. full 300-epoch runs under Gloo on 16 GPUs: time to accuracy, comparable to Table 3 (NCCL)
DRY_RUN=1 GROUP=gloo16_full BACKENDS=gloo WORKERS=16 SEEDS="1 2 3" EPOCHS=300 TIME=00:30:00 ./perlmutter/submit_scaling.sh
```

Estimate for 8: the measured Gloo batch times (72.1 ms SGD, 38.4 ms Rank 2 at 16 GPUs) x about 24 batches per epoch x
300 epochs is about 9 and 5 minutes of training, plus about 2 minutes of evaluation, so about 11 and 7 minutes per job
and about 3.6 node-hours for all six. A different `EPOCHS` needs a different `GROUP`, because the run name does not
include the epoch count.

`PRESETS` takes a `;`-separated list (`PRESETS="SGD;Rank 2;Rank 4"`), and `TASK=lm` runs the WikiText-2 LSTM instead.

## 10. ResNet-18 with our own trainer

The scripts already exist (`scripts/time_epoch_1node.sh`, `scripts/tier1_resnet18_1node.sbatch`,
`scripts/tier3_resnet18_2node.sbatch`); their time limits are placeholders, so time an epoch first.

```bash
salloc -N 1 -C gpu -G 4 -q interactive -t 00:20:00 -A $SBATCH_ACCOUNT
bash scripts/time_epoch_1node.sh        # 2 epochs each of DDP and PowerSGD rank 2; prints seconds per epoch; then exit
EPOCHS=30 sbatch -t <limit> scripts/tier1_resnet18_1node.sbatch     # DDP vs PowerSGD rank 1, 2, 4 on 4 GPUs
```

Set `<limit>` (HH:MM:SS) to about 4 configs x 15 x the 2-epoch wall time the timing script prints, plus 10 minutes. On
`master` the tier scripts still have an `-A` line; `SBATCH_ACCOUNT` overrides it (sbatch lets environment variables
override `#SBATCH` lines). The 2-node `tier3_resnet18_2node.sbatch` (DDP vs rank 2 on 8 GPUs) is optional and costs 2x
per hour.

This trainer uses a 30-epoch cosine schedule with batch 128 per GPU, not the paper's 300-epoch schedule, so its
accuracy is not comparable with Table 3. What it adds is the same comparison as NanoGPT (same hooks, byte counting and
timing) on a CNN: compare DDP and PowerSGD within the job with `summarize_experiments.py runs --match resnet18`.

## After the runs: copy, then summarize

`perlmutter/bundle_results.sh` copies the outputs (never moves them), adds `sacct` job and GPU accounting files, redacts
the scratch path, user name and Slurm account in the copies, and writes a `.tgz`:

```bash
# NanoGPT jobs (run folders, the shared runs.jsonl, job logs, nvidia-smi CSVs)
bash perlmutter/bundle_results.sh nanogpt_next <jobid1>,<jobid2> \
  results/runs/*_<jobid1> results/runs/*_<jobid2> results/runs.jsonl logs/*_<jobid1>.out logs/*_<jobid2>.out logs/gpu_<jobid1>_* logs/gpu_<jobid2>_*

# paper-code jobs
bash perlmutter/bundle_results.sh scaling_next <jobid>,<jobid>,... results/scaling results/slurm-logs/scaling_*
```

Then, on your laptop: `scp <user>@perlmutter.nersc.gov:<your scratch>/<name>.tgz ~/Downloads/`, extract into a new
folder, and run the check line the script prints before committing anything.

```bash
python tools/summarize_experiments.py runs --jsonl <extracted>/results/runs.jsonl --match '_<jobid>$'
python tools/summarize_experiments.py paper <extracted>/results/scaling/*
python tools/summarize_experiments.py gpu <extracted>/logs/gpu_<jobid>_*.csv
```

`runs` compares each run with the DDP / fp32 run on the same modeled link (or none) in the same job, with the same seed,
model size and step count; the full table above it has every run's absolute step time, including DDP with and without
the link.
`paper` compares each method with SGD in the same task, backend, GPU count, seed and epoch count, and gives
mean [min, max] wherever a setting has more than one seed.

## Reporting rules

- One seed is indicative. Report spreads once there are seeds 2 and 3.
- `comm_bytes_per_step` and `bits` are the logical payload handed to all-reduce per GPU, not wire traffic.
- fp8 is emulated, and the slow link is a model; say so wherever those numbers appear.
- Gloo on Perlmutter is a slower software path on a fast network, not the paper's 10 Gbit/s Ethernet.
- Compare step times within a job. 1-node and 2-node jobs have run on different A100 models (80 GB and 40 GB).
- A faster batch is not faster training: use experiment 8 (or best validation loss against wall time) for that claim.
