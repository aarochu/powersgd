# PowerSGD: reproduction and NanoGPT extension on Perlmutter

A course project for distributed deep learning (Rutgers) by Aaron Chu and Nayan Yadav. We reproduce
[PowerSGD](https://arxiv.org/abs/1905.13727) (Vogels, Karimireddy, Jaggi, NeurIPS 2019) on NERSC Perlmutter (A100 GPUs, NCCL)
and extend it to a small transformer. This repository is a fork of [epfml/powersgd](https://github.com/epfml/powersgd);
the original library and paper code are kept (see [Upstream PowerSGD](#upstream-powersgd) below).

PowerSGD replaces each layer's gradient matrix (n x m) with a rank-r approximation P Qᵀ, where P is n x r and Q is m x r,
so each worker all-reduces r(n + m) numbers instead of n x m.

| Part | Who | What | Where |
|---|---|---|---|
| Paper reproduction | Nayan Yadav | Table 3: ResNet-18 on CIFAR-10 and a 3-layer LSTM on WikiText-2; SGD and PowerSGD rank 1, 2, 4; 16 GPUs; 3 seeds | [`paper-code/`](paper-code), [`perlmutter/`](perlmutter), [`results/paper_reproduction/`](results/paper_reproduction) |
| NanoGPT extension | Aaron Chu | NanoGPT on Shakespeare (char-level), DDP vs PowerSGD rank 1, 2, 4, on 1 node (4 GPUs) and 2 nodes (8 GPUs) | [`train_ddp.py`](train_ddp.py), [`models.py`](models.py), [`scripts/`](scripts), [`results/`](results) |

## Results

### Paper reproduction (16 A100s, 3 seeds, mean)

Full tables, per-run values and curves: [`results/paper_reproduction/table3_results.md`](results/paper_reproduction/table3_results.md).

| | CIFAR-10 acc. (ours / paper) | LSTM perplexity (ours / paper) | MB sent per epoch, CIFAR / LSTM | Time per batch vs SGD, CIFAR / LSTM (paper) |
|---|---|---|---|---|
| SGD | 94.1 / 94.3 | 99.6 / 91 | 1073 / 7882 | 0% |
| Rank 1 | 93.3 / 93.6 | 111.2 / 102 | 4 / 25 | +8% / -8% (paper -26% / -56%) |
| Rank 2 | 94.4 / 94.4 | 101.4 / 93 | 8 / 39 | +13% / -6% (paper -23% / -53%) |
| Rank 4 | 94.5 / 94.5 | 98.9 / 91 | 15 / 65 | +27% / -3% (paper -16% / -55%) |

- CIFAR-10 accuracy matches the paper. LSTM perplexity is 8 to 9 points worse than the paper for every method, SGD included.
- The data sent drops by two orders of magnitude, but **the paper's wall-clock speedups did not reproduce** on Perlmutter. CIFAR-10 steps were slower with compression;
  the LSTM was only a few percent faster.
- Our reading, not tested here: Perlmutter's interconnect is fast enough that communication is not the bottleneck, so the compression overhead is not repaid.

### NanoGPT (10.8M parameters, seed 1, fp32, 1 node of 4 A100-80GB)

| | Logical bytes per step, per rank | vs DDP | Median step time vs DDP |
|---|---|---|---|
| DDP | 43,183,104 | 1x | 0% |
| PowerSGD rank 1 | 276,488 | 156x fewer | +4.0% |
| PowerSGD rank 2 | 430,096 | 100x fewer | +6.8% |
| PowerSGD rank 4 | 737,312 | 59x fewer | +12.5% |

- DDP's best validation loss is 1.481 at step 750; it then overfits (2.209 at step 2000).
- Rank 4 needs about 2500 steps (6000-step run) to reach 1.475, then overfits too (2.135 at step 6000). Rank 2 reaches 1.503 at step 4500.
  So PowerSGD slows convergence per step and does not improve the best loss here.
- Everything above is a **single seed**; differences of a few hundredths in loss are not established. Repeat runs of the same seed agree within about 0.01.

Details, caveats and the full run table: [`results/README.md`](results/README.md), [`results/MANIFEST.md`](results/MANIFEST.md), [`docs/RUNS.md`](docs/RUNS.md).

### Caveats

- `comm_bytes_per_step` is the logical payload given to `dist.all_reduce` on each rank, not bytes on the wire.
- 1-node NanoGPT jobs ran on A100-80GB and 2-node jobs on A100-40GB, so compare step times only within one setup.
- The NanoGPT learning rate is not scaled with world size, so 1-node (global batch 256) and 2-node (512) losses are not comparable.
- Not run: ResNet-18 through `train_ddp.py`, real FP8 (A100 has none), tensor parallelism.

## Repository layout

| Path | Contents |
|---|---|
| `powersgd/` | upstream PowerSGD library (PyTorch) |
| `paper-code/` | upstream research code for the paper, ported to run on Perlmutter; `run_experiment.py` launches one run, `analyze_results.py` builds the tables |
| `perlmutter/` | `env.sh`, `setup.sh`, `run.sbatch`, `submit_paper.sh` for the paper reproduction |
| `train_ddp.py`, `models.py`, `common/` | DDP and PowerSGD trainer (PyTorch DDP communication hooks), NanoGPT and ResNet-18 models, run logger |
| `scripts/` | `common.sh` (torchrun launcher, GPU monitor) and the sbatch/shell scripts for the NanoGPT runs |
| `tools/` | `prepare_data.py`, `make_results.py`, `organize_results.py`, `update_budget.py` |
| `results/` | run outputs, `MANIFEST.md`, `RESULTS.md`, and `paper_reproduction/` |
| `figures/`, `presentation/` | plots and the slide deck |
| `docs/`, `NOTES.md` | run documentation, risks, running log |

## Running it

Both parts expect Perlmutter with a PyTorch module and an allocation. Compute nodes have no internet, so download data on a login node first.

**Paper reproduction** (4 nodes x 4 GPUs per run):

```bash
export NERSC_ACCOUNT=<your project>     # e.g. m1234
bash perlmutter/setup.sh                # login node: checks the stack, downloads CIFAR-10 and WikiText-2
./perlmutter/submit_paper.sh smoke      # short debug-queue check, run this first
./perlmutter/submit_paper.sh table3     # SGD, rank 1, 2, 4 on CIFAR-10 and WikiText-2
```

Runs already finished are skipped, so the script is safe to re-run. Tables are rebuilt with `paper-code/analyze_results.py` (see
`results/paper_reproduction/table3_results.md`).

**NanoGPT** (inside a Slurm allocation; the scripts do not hard-code one, so set yours first):

```bash
export SBATCH_ACCOUNT=<your project>    # read by sbatch; for salloc use SALLOC_ACCOUNT or -A
python tools/prepare_data.py            # login node: downloads tiny Shakespeare
source scripts/common.sh
start_gpu_monitor
COMMON="--model nanogpt --seed 1 --max-steps 2000 --eval-interval 250 --start-iter 10"
launch $COMMON --mode ddp               --run-name gpt_ddp
launch $COMMON --mode powersgd --rank 2 --run-name gpt_powersgd_r2
stop_gpu_monitor
```

`--rank` is the PowerSGD rank, not the distributed process rank. Error feedback and warm start are always on in `--mode powersgd`.
The exact batch scripts that produced the reported results are `scripts/nanogpt_1node_long.sbatch` and `scripts/nanogpt_2node.sbatch`.

---

# Upstream PowerSGD

The rest of this file is the original README from [epfml/powersgd](https://github.com/epfml/powersgd), kept for the library
usage and the citation.

## Reference implementation

This is a reference implementation for the PowerSGD algorithm.

Installation:

```bash
pip install git+https://github.com/epfml/powersgd.git
```

Usage:

```diff
+ from powersgd import PowerSGD, Config, optimizer_step

  model = torchvision.models.resnet50(pretrained=True)
  params = list(model.parameters())
  optimizer = torch.optim.SGD(params, lr=0.1)

+ powersgd = PowerSGD(params, config=Config(
+     rank=1,  # lower rank => more aggressive compression
+     min_compression_rate=10,  # don't compress gradients with less compression
+     num_iters_per_step=2,  #   # lower number => more aggressive compression
+     start_compressing_after_num_steps=0,
+ ))

  for each batch:
      loss = ...
-     optimizer.zero_grad()
      loss.backward()
-     optimizer.step()
+     optimizer_step(optimizer, powersgd)
```

## Differences with the paper version

The version in this code base is a slight improvement over the version in the PowerSGD paper.
It looks a bit like Algorithm 2 in [this follow-up paper](https://arxiv.org/pdf/2008.01425.pdf).

We found that there are two ways to control the approximation quality in PowerSGD: the first is the 'rank' of the approximation, and the second is the 'number of powerSGD iterations' in between gradient steps, while keeping the rank 1. Because the cost of orthogonalisation grows as $O(\text{rank}^2)$, increasing the rank can become inefficient, leaving changing the number of iterations as the best option.

In the original PowerSGD paper, more iterations only improves the quality of the rank-k approximation, as the approximation converges to the "best rank k approximation". In the [follow-up paper](https://arxiv.org/pdf/2008.01425.pdf), intermediate results from these rank 1 power iterations are all used and communicated, effectively increasing the rank as the number of iterations grows.

In the original PowerSGD paper, we used two iterations per SGD step (a left and a right iteration). In this setting, there is not much of a difference. The difference appears when you use more power iteration steps per SGD step.



## PyTorch implementation
PyTorch features an implementation of PowerSGD as a [communucation hook](https://pytorch.org/docs/stable/ddp_comm_hooks.html) for `DistributedDataParallel` models.
Because of the integration with DDP, the code is more involved than the code in this repository.
## Research code

Research code for the experiments in the [PowerSGD paper](https://arxiv.org/abs/1905.13727) is located under [paper-code](./paper-code/README.md).

## Selected follow-up work 
- [(Cho et al., 2019)](http://learningsys.org/neurips19/assets/papers/1_CameraReadySubmission_mlsys_grz_camera_ready.pdf) concurrently developed an algorithm that is fundamentally very similar to PowerSGD.
- [(Ramesh et al., 2021 - DALL-E)](https://arxiv.org/abs/2102.12092) share valuable recommendations in using PowerSGD for large-scale transformer training.
- [(Agarwal et al., 2020)](https://arxiv.org/pdf/2010.16248.pdf) share insights into adaptive compression with PowerSGD.
- [(Vogels et al., 2020)](https://arxiv.org/abs/2008.01425) adapt PowerSGD to work in a decentralized setting (with sparse connectivity between workers.)
- [(Wang, 2021)](https://medium.com/pytorch/accelerating-pytorch-ddp-by-10x-with-powersgd-585aef12881d) introduces a variation to PowerSGD and describes his experience with PowerSGD on large language models.
- [(Song et al., 2023)](https://arxiv.org/abs/2301.09830) utilizes PowerSGD (and its slight variant) to compressing pipeline-/data-parallelism gradients in 3D parallelism-based LLM training.
- (Please submit a PR if you want your work to be included here.)


# Reference

If you use this code, please cite the following [paper](https://arxiv.org/abs/1905.13727)

    @inproceedings{vkj2019powersgd,
      author = {Vogels, Thijs and Karimireddy, Sai Praneeth and Jaggi, Martin},
      title = "{{PowerSGD}: Practical Low-Rank Gradient Compression for Distributed Optimization}",
      booktitle = {NeurIPS 2019 - Advances in Neural Information Processing Systems},
      year = 2019,
      url = {https://arxiv.org/abs/1905.13727}
    }
