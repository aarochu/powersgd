# Experiments index

One row per experiment folder. Every number in a folder's README comes from the files in that folder. Node-hours are
by elapsed time from `sacct`; charged hours may differ. Single-seed results, emulated fp8 and the modeled slow link are
labeled as such in each README.

| Folder | Issue | What | Jobs | Node-hours | Headline |
|---|---|---|---|---|---|
| [`paper_reproduction/`](paper_reproduction/) | - | Table 3: CIFAR-10 ResNet-18 and WikiText-2 LSTM, paper code, NCCL, 16 GPUs, 3 seeds | 24 jobs, see folder | see folder | CIFAR-10: SGD 94.1%, Rank 2 94.4%, Rank 4 94.5%; Rank 2 is 13% slower per batch than SGD on NCCL |
| [`organized/`](organized/) | - | First NanoGPT runs with `train_ddp.py` (seed 1, 1 and 2 nodes, a 6000-step run) | 60622124, 60623329, 60623330, 60626982, 60627319 | see `MANIFEST.md` | PowerSGD sends 59-156x less, 4-13% slower per step |
| `gloo_scaling/` (PR #8) | - | Paper code with the Gloo backend, 8 and 16 GPUs, 10 epochs | 60949651, 60951117-60951120 | 0.21 | Rank 2 about half the time per batch of SGD when communication is slow |
| [`smoke_tests/`](smoke_tests/) | #11 | GPU smoke tests of the new code | 62006046 | 0.49 | Found and fixed two bugs; all checks pass |
| [`nanogpt_seeds/`](nanogpt_seeds/) | #12 | NanoGPT seeds 2-3 (seed 1 from `organized/`), 1 and 2 nodes | 62006912, 62006917 | 1.31 | Ordering holds over 3 seeds: DDP 1.481 [1.475, 1.488], rank 4 1.509, rank 2 1.765, rank 1 2.307 (1 node) |
| [`nanogpt_precision/`](nanogpt_precision/) | #13 | fp16, bf16, emulated fp8 gradients; PowerSGD rank 4 on fp16 | 62006918 | 0.37 | fp16/bf16 halve the payload at no loss; rank 4 on fp16 sends 117x less than DDP fp32 |
| [`nanogpt_slowlink/`](nanogpt_slowlink/) | #14 | Modeled 10 Gbit/s link | 62006920 | 0.36 | PowerSGD steps 32-34% faster than DDP, but DDP still reaches its best loss sooner (101 s vs 185 s for rank 4) |
| [`nanogpt_large/`](nanogpt_large/) | #15 | 25M and 85M-parameter NanoGPT; the 85M model on 4, 8 and 16 GPUs with a profiled step | 62006924, 62006927, 62010554, 62010559, 62010560 | 4.18 | Compression overhead shrinks with model size (rank 4: +12%, +7.3%, +3.7%); at 16 GPUs the all-reduce is 2.5% of GPU time, so PowerSGD stays 2-4% slower |
| [`resnet18_ddp/`](resnet18_ddp/) | #18 | ResNet-18, 30 epochs, `train_ddp.py` | 62006947 | 0.14 | DDP 92.54%, rank 4 91.88%, rank 2 91.09%, rank 1 87.33%; 15-46% slower per step |
| [`nanogpt_tp/`](nanogpt_tp/) | #19 | Tensor vs data parallelism (`train_tensor.py`) | 62006949, 62006950 | 0.81 | Tensor parallelism moves 28-56x more data; PowerSGD shrinks only the gradient part |

Not run: #16 (scaling grid), #17 (300-epoch Gloo runs), #20 (LSTM tokenizer check).
