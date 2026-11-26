# Data vs tensor parallelism: where gradient compression applies

A design note for the "data vs tensor parallelism" next step. The traffic numbers below are computed from the model
sizes we used; `train_tensor.py` (experiment 11 in `NEXT_EXPERIMENTS.md`) measures them. **It has not been run yet.**

## Data parallelism (what we ran)

Every GPU holds the whole model, trains on its own slice of the batch, and the GPUs all-reduce their **gradients** once
per step. That one all-reduce is the only communication. PowerSGD compresses exactly this message:

- the payload is a set of weight-gradient matrices, which tend to be close to low rank, so a rank-r approximation loses
  little;
- the same matrices come back every step, so warm start (reusing Q) and error feedback (adding back what was missed
  last step) work: an error made in one step is corrected in later steps;
- the result only feeds the optimizer, so a slightly wrong gradient slows training a little instead of breaking it.

For our NanoGPT (10.8M parameters) that message is 43.2 MB per GPU per step in fp32 (`comm_bytes_per_step` in
`results/runs.jsonl`); rank 2 sends 0.43 MB.

## Tensor parallelism (Megatron-style)

Each weight matrix is split across GPUs (column-wise in the first linear layer of a block, row-wise in the second), so
each GPU computes part of every layer. The GPUs must then combine **activations** inside every forward and backward
pass: in the usual layout, two all-reduces per transformer layer in the forward pass (after attention and after the
MLP) and two in the backward pass.

Computed for our NanoGPT shape (batch 64, sequence 256, width 384, fp32, 6 layers):

| | Messages per step | Size of each | Total per GPU per step |
|---|---|---|---|
| Data parallel gradient all-reduce | 1 (in buckets) | 43.2 MB in total | 43.2 MB |
| Tensor parallel activation all-reduces | 4 per layer x 6 = 24 | 64 x 256 x 384 x 4 B = 25.2 MB | about 604 MB |

So tensor parallelism moves much more data, more often, and it sits on the critical path of every layer.

Why PowerSGD does not carry over to these messages:

- **They change every micro-batch.** Activations depend on the input, so there is no stable matrix for warm start to
  track, and no next step in which error feedback could repay this step's error.
- **Errors compound.** A compressed activation feeds the next layer immediately; the error propagates through the rest
  of the forward and backward pass instead of only nudging the optimizer.
- **Low rank is not given.** Weight gradients summed over a batch tend to be low rank; a batch of activations has no
  such guarantee.
- **The tooling does not exist here.** PyTorch's `powerSGD_hook` is a DDP communication hook that works on gradient
  buckets. Using PowerSGD inside tensor-parallel collectives (or FSDP's reduce-scatter) would need new code.

## Where compression still applies in a hybrid setup

Large models usually combine both: tensor parallelism inside a node and data parallelism across nodes. In that layout:

- the **data-parallel gradient all-reduce across nodes** is still a gradient message. Each GPU's shard of a weight
  gradient is still a matrix (n x m/t for t-way tensor parallelism), so PowerSGD applies to it, and it is the message
  that crosses the slower inter-node network;
- the **tensor-parallel activation all-reduces** stay uncompressed, and they stay inside a node on NVLink, where
  bandwidth is highest.

This is the setting of the follow-up work listed in the upstream README (Song et al., 2023), which applies PowerSGD-style
compression to the data-parallel and pipeline-parallel parts of 3D-parallel LLM training.

## How we measure it: `train_tensor.py`

A separate trainer, so `train_ddp.py` (which produced every result so far) and `models.py` stay unchanged:

- It builds the normal `models.GPT` with the usual init, then replaces each block's attention `qkv` and MLP first layer
  with column slices and the attention `proj` and MLP second layer with row slices, Megatron-style. The split layers
  are written by hand with plain PyTorch (explicit all-reduces in two small autograd functions), so DDP, the PowerSGD
  hook and byte counting work as before. Because the fused `qkv` layer is split by heads (each rank keeps the q, k and v
  rows of its own heads), a replacement attention forward runs the same math on the local heads.
- `--tp-size` GPUs of a node form a tensor-parallel group; data parallelism (DDP, optionally PowerSGD) runs across the
  groups, only on gradients. Every GPU of a group sees the same batch, and evaluation runs on the whole group.
- Bytes are counted per group: gradient all-reduces (data-parallel) and activation all-reduces (tensor-parallel).
- With the default per-replica batch of 64 x tp, every config trains on the same global batch as `train_ddp.py` on the
  same GPUs, so the table above becomes: tp 2 moves about 1.21 GB of activations and 21.9 MB of gradients per GPU per
  step, against 43.2 MB of gradients for plain data parallelism.

`scripts/smoke_tp.sh` checks it before any real run: tp 1 must reproduce `train_ddp.py`; on real data, the loss before
training and after 40 steps with dropout off must match an unsplit run with the same data-parallel layout; and the
counted bytes must equal the formulas. `scripts/nanogpt_tp.sbatch` runs the comparison. The tensor-parallel size must
divide the number of attention heads, so the 6-head model splits 2 ways; tp 4 runs use 12 heads.

What it will and will not show: how much more a tensor-parallel step communicates and what that costs in step time on
NVLink, and that PowerSGD shrinks only the gradient part. It does not test compressing activations, which this note
argues against.
