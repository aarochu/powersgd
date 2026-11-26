# Data vs tensor parallelism: where gradient compression applies

A design note for the "data vs tensor parallelism" next step. **Nothing here was run.** The traffic numbers are computed
from the model sizes we used, not measured.

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

## What an experiment would need

To measure this instead of computing it: a tensor-parallel NanoGPT (for example with
`torch.distributed.tensor.parallel` column/row-wise plans), byte counting on its collectives, and a run with
2-way tensor parallelism x 2-way data parallelism on one node, then across nodes. That changes `models.py`, so it needs
an explicit go-ahead; it is not part of this change.
