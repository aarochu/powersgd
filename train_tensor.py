#!/usr/bin/env python3
"""NanoGPT with tensor parallelism inside each node and data parallelism across, for the "data vs tensor parallelism"
experiment (docs/tensor_parallelism.md). Kept separate on purpose: train_ddp.py and models.py are not changed. The
NanoGPT task (data, AdamW, LR schedule, evaluation) is imported from train_ddp.py, so with --tp-size 1 this trainer
does the same thing as `train_ddp.py --model nanogpt`; scripts/smoke_tp.sh checks that.

    torchrun --nproc_per_node=4 train_tensor.py --tp-size 2 --mode ddp
    torchrun --nproc_per_node=4 train_tensor.py --tp-size 2 --mode powersgd --rank 4

Layout: global ranks form tensor-parallel (TP) groups of --tp-size consecutive ranks, which stay inside one node, and
data-parallel (DP) groups of the ranks that hold the same slice. Each TP group holds one copy of the model, split
Megatron-style across its GPUs; DDP (and PowerSGD) all-reduce gradients only inside a DP group, so PowerSGD never
touches the TP traffic.

Split of the unchanged models.GPT, after its normal init (so a split model starts from the same weights as a full one):
    attention qkv  column-parallel: each rank keeps the q, k and v rows of its heads
    attention proj row-parallel: each rank keeps the input columns of its heads; partial outputs are all-reduced
    MLP fc1        column-parallel
    MLP fc2        row-parallel
    embeddings, layer norms, LM head: replicated (small here); their gradients come out the same on every TP rank
Per block: one all-reduce after attention and one after the MLP going forward, one before each going backward
(Megatron's g and f operators), each of batch x sequence x width fp32 values. TP and DP bytes are counted separately.
Dropout masks come from the same seed on every TP rank, as in train_ddp.py; --dropout 0 makes runs with different
--tp-size comparable step by step.
"""
import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook as powerSGD
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.run_logger import ROOT, log_run  # noqa: E402
from models import GPT, GPTConfig  # noqa: E402
from train_ddp import GPTTask  # noqa: E402


# ----------------------------------------------------------------------------- args
def parse_args():
    p = argparse.ArgumentParser(description="NanoGPT with tensor x data parallelism (see module docstring)")
    p.add_argument("--tp-size", type=int, default=1, help="GPUs per tensor-parallel group (must divide GPUs per node)")
    p.add_argument("--mode", choices=["ddp", "powersgd"], default="ddp", help="gradient all-reduce across DP groups")
    p.add_argument("--rank", type=int, default=2, help="PowerSGD rank R (not the process rank)")
    p.add_argument("--start-iter", type=int, default=10, help="PowerSGD start_powerSGD_iter")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=None,
                   help="sequences per DP replica per step (default 64 x tp-size: the same global batch as train_ddp.py)")
    p.add_argument("--max-steps", type=int, default=None, help="total steps (default 2000)")
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", str(ROOT / "data")))
    p.add_argument("--out-dir", default=str(ROOT / "results" / "runs"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--synthetic", action="store_true", help="random data, for pipeline smoke tests only")
    p.add_argument("--profile-step", type=int, default=0, help=">0: torch.profiler trace of that single step (rank 0)")
    p.add_argument("--timing-warmup", type=int, default=None, help="steps ignored in step-time stats")
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1, help="as models.GPTConfig (0.1)")
    a = p.parse_args()
    a.model, a.precision = "nanogpt", "fp32"  # fields GPTTask and the run record expect
    return a


# ----------------------------------------------------------------------------- layout and slices (pure Python)
def group_layout(world_size, tp):
    """(tp_groups, dp_groups) as lists of global ranks. TP groups are consecutive ranks, so they stay inside a node
    when tp divides the GPUs per node; DP groups are the ranks with the same position in their TP group."""
    tp_groups = [list(range(s, s + tp)) for s in range(0, world_size, tp)]
    dp_groups = [list(range(i, world_size, tp)) for i in range(tp)]
    return tp_groups, dp_groups


def block(n, tp, r):
    """Indices of TP rank r's contiguous 1/tp share of n features."""
    c = n // tp
    return list(range(r * c, (r + 1) * c))


def qkv_rows(n_embd, tp, r):
    """Rows of the fused qkv weight (q, k, v stacked, n_embd rows each, heads contiguous) that TP rank r keeps:
    its share of the q rows, then of the k rows, then of the v rows, so its local output splits into its own q, k, v."""
    return [part * n_embd + i for part in range(3) for i in block(n_embd, tp, r)]


# ----------------------------------------------------------------------------- Megatron f / g and split layers
class _CopyToTP(torch.autograd.Function):
    """f: identity going forward; all-reduce of the input gradient over the TP group going backward."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTP(torch.autograd.Function):
    """g: all-reduce over the TP group going forward; identity going backward."""

    @staticmethod
    def forward(ctx, x, group):
        x = x.contiguous().clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class ColumnParallelLinear(nn.Module):
    """Keeps the given output rows of a full nn.Linear; input replicated, output split."""

    def __init__(self, full, rows, group):
        super().__init__()
        idx = torch.tensor(rows, device=full.weight.device)
        self.weight = nn.Parameter(full.weight.detach().index_select(0, idx).clone())
        self.bias = nn.Parameter(full.bias.detach().index_select(0, idx).clone())
        self.group = group

    def forward(self, x):
        return F.linear(_CopyToTP.apply(x, self.group), self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Keeps the given input columns of a full nn.Linear; input split, partial outputs summed, bias added once."""

    def __init__(self, full, cols, group):
        super().__init__()
        idx = torch.tensor(cols, device=full.weight.device)
        self.weight = nn.Parameter(full.weight.detach().index_select(1, idx).clone())
        self.bias = nn.Parameter(full.bias.detach().clone())
        self.group = group

    def forward(self, x):
        return _ReduceFromTP.apply(F.linear(x, self.weight), self.group) + self.bias


def _tp_attention_forward(self, x):
    """models.CausalSelfAttention.forward on this rank's heads (qkv column-split, proj row-split)."""
    B, T, _ = x.shape
    qkv = self.qkv(x)
    c = qkv.shape[-1] // 3
    h = self.n_head_local
    q, k, v = (t.view(B, T, h, c // h).transpose(1, 2) for t in qkv.split(c, dim=2))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
    return self.resid_drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, c)))


def split_model(model, tp, tp_rank, group):
    """Replace each block's attention and MLP linears with this rank's column/row slices, in place."""
    for b in model.blocks:
        att = b.attn
        n_embd = att.qkv.in_features
        att.qkv = ColumnParallelLinear(att.qkv, qkv_rows(n_embd, tp, tp_rank), group)
        att.proj = RowParallelLinear(att.proj, block(n_embd, tp, tp_rank), group)
        att.n_head_local = att.n_head // tp
        att.forward = types.MethodType(_tp_attention_forward, att)
        fc1, fc2 = b.mlp[0], b.mlp[2]
        b.mlp[0] = ColumnParallelLinear(fc1, block(fc1.out_features, tp, tp_rank), group)
        b.mlp[2] = RowParallelLinear(fc2, block(fc2.in_features, tp, tp_rank), group)
    return model


class GroupCommCounter:
    """Logical payload bytes passed to dist.all_reduce, split into TP (calls on the TP group) and DP (all others)."""

    def __init__(self, tp_group):
        self.dp_bytes, self.tp_bytes, self.enabled = 0, 0, True
        orig = dist.all_reduce

        def counted(tensor, *a, **k):
            if self.enabled:
                n = tensor.numel() * tensor.element_size()
                group = k.get("group", a[1] if len(a) > 1 else None)
                if tp_group is not None and group is tp_group:
                    self.tp_bytes += n
                else:
                    self.dp_bytes += n
            return orig(tensor, *a, **k)

        dist.all_reduce = counted


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
    t = a.tp_size
    if t < 1 or local_ws % t or ws % t or a.n_head % t:
        raise SystemExit(f"--tp-size {t} must divide the GPUs per node ({local_ws}), the world size ({ws}) "
                         f"and --n-head ({a.n_head})")
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.set_device(local)
    dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    dist.init_process_group("nccl" if cuda else "gloo")
    is0 = rank == 0
    tp_rank, dp_rank, dp_size = rank % t, rank // t, ws // t
    tp_group = dp_group = None  # None = the default (world) group, exactly as in train_ddp.py
    if t > 1:
        tp_groups, dp_groups = group_layout(ws, t)
        for ranks in tp_groups:  # every rank must create every group, in the same order
            g = dist.new_group(ranks)
            if rank in ranks:
                tp_group = g
        for ranks in dp_groups:
            g = dist.new_group(ranks)
            if rank in ranks:
                dp_group = g
    torch.manual_seed(a.seed)  # same seed everywhere -> identical full init on every rank before the split
    np.random.seed(a.seed)

    a.batch_size = a.batch_size or 64 * t
    # data follows the DP rank, so every GPU of a TP group trains on the same batch (with tp 1 it is train_ddp.py's)
    task = GPTTask(a, {"device": dev, "rank": dp_rank})

    rid = [a.run_name or f"gpt_tp{t}_{a.mode}{a.rank if a.mode == 'powersgd' else ''}_w{ws}_s{a.seed}_{time.strftime('%m%d-%H%M%S')}"]
    dist.broadcast_object_list(rid, src=0)
    run_id = rid[0]
    run_dir = Path(a.out_dir) / run_id
    global_batch = a.batch_size * dp_size
    lr_eff = task.base_lr()
    if is0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(
            vars(a) | {"world_size": ws, "run_id": run_id, "trainer": "train_tensor.py",
                       "resolved": {"tp_size": t, "dp_size": dp_size, "batch_size_per_replica": a.batch_size,
                                    "global_batch": global_batch, "lr_eff": lr_eff}}, indent=1))
    metrics_f = open(run_dir / "metrics.jsonl", "a") if is0 else None

    def emit(rec):
        if is0:
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(json.dumps(rec), flush=True)

    model = GPT(GPTConfig(vocab_size=task.vocab, block_size=a.block_size, n_layer=a.n_layer, n_head=a.n_head,
                          n_embd=a.n_embd, dropout=a.dropout)).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    if t > 1:
        split_model(model, t, tp_rank, tp_group)
    nlocal = sum(p.numel() for p in model.parameters())
    model = DDP(model, device_ids=[local] if cuda else None, process_group=dp_group)
    counter = GroupCommCounter(tp_group)
    if a.mode == "powersgd":
        state = powerSGD.PowerSGDState(process_group=dp_group, matrix_approximation_rank=a.rank,
                                       start_powerSGD_iter=a.start_iter, use_error_feedback=True, warm_start=True,
                                       random_seed=a.seed)
        model.register_comm_hook(state, powerSGD.powerSGD_hook)
    else:
        model.register_comm_hook(dp_group, default_hooks.allreduce_hook)
    opt = task.make_optimizer(model)

    # validation loss before any training: the same on every --tp-size if the split is right (dropout is off in eval)
    counter.enabled = False
    init_ev = task.evaluate(model.module)  # all ranks: the TP all-reduces need the whole group
    counter.enabled = True
    warm_ignore = a.timing_warmup if a.timing_warmup is not None else max(20, a.start_iter + 10)
    emit({"type": "start", "run_id": run_id, "params": nparams, "params_per_gpu": nlocal, "world_size": ws, "tp_size": t,
          "dp_size": dp_size, "total_steps": task.total_steps, "lr_eff": lr_eff, "global_batch": global_batch,
          "init_val_loss": init_ev["val_loss"], "device": torch.cuda.get_device_name(local) if cuda else "cpu",
          "torch": torch.__version__})

    def sync():
        if cuda:
            torch.cuda.synchronize()

    step_times, data_times, dp_per_step, tp_per_step = [], [], [], []
    run_loss, run_n = 0.0, 0
    best, best_step, last_ev = None, None, None
    status = "ok"
    dist.barrier()
    sync()
    t_start = time.perf_counter()
    train_time = 0.0
    try:
        for step in range(task.total_steps):
            for g in opt.param_groups:
                g["lr"] = task.lr_at(step, lr_eff)
            td0 = time.perf_counter()
            batch = task.next_batch(step)
            sync()
            td1 = time.perf_counter()
            prof = None
            if a.profile_step and step == a.profile_step and is0:
                acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if cuda else [])
                prof = torch.profiler.profile(activities=acts)
                prof.__enter__()
            d0, k0 = counter.dp_bytes, counter.tp_bytes
            opt.zero_grad(set_to_none=True)
            loss = task.loss(model, batch)
            loss.backward()
            opt.step()
            sync()
            td2 = time.perf_counter()
            if prof is not None:
                prof.__exit__(None, None, None)
                (run_dir / "profile_table.txt").write_text(
                    prof.key_averages().table(sort_by="cuda_time_total" if cuda else "cpu_time_total", row_limit=30))
                prof.export_chrome_trace(str(run_dir / "trace_rank0.json"))

            lv = loss.item()
            if not math.isfinite(lv):
                status = "diverged"
                emit({"type": "diverged", "step": step})
                break
            step_times.append(td2 - td1)
            data_times.append(td1 - td0)
            dp_per_step.append(counter.dp_bytes - d0)
            tp_per_step.append(counter.tp_bytes - k0)
            train_time += td2 - td0
            run_loss += lv
            run_n += 1

            if task.is_eval_step(step):
                counter.enabled = False
                ev = task.evaluate(model.module)  # all ranks (TP all-reduces); only rank 0 records it
                if is0:
                    last_ev = ev
                    if best is None or ev["val_loss"] < best["val_loss"]:
                        best, best_step = ev, step + 1
                    steady = step_times[warm_ignore:] or step_times
                    emit({"type": "eval", "step": step + 1, "epoch": None, "train_loss": run_loss / run_n,
                          "train_time_s": train_time, "elapsed_s": time.perf_counter() - t_start,
                          "median_step_s": float(np.median(steady)), **ev})
                run_loss, run_n = 0.0, 0
                dist.barrier()
                counter.enabled = True
    except Exception as e:  # log failure then re-raise
        status = f"error: {type(e).__name__}: {e}"[:300]
        raise
    finally:
        sync()
        wall = time.perf_counter() - t_start
        if is0:
            steady = step_times[warm_ignore:] if len(step_times) > warm_ignore else step_times
            dp_steady = dp_per_step[a.start_iter + 2:] or dp_per_step  # after PowerSGD kicks in
            tokens = a.batch_size * dp_size * a.block_size
            np.savetxt(run_dir / "step_times.csv",
                       np.c_[np.arange(len(step_times)), step_times, data_times, dp_per_step, tp_per_step],
                       delimiter=",", header="step,step_time_s,data_time_s,dp_bytes,tp_bytes", comments="",
                       fmt=["%d", "%.6f", "%.6f", "%d", "%d"])
            rec = {
                "run_id": run_id, "trainer": "train_tensor.py", "model": "nanogpt", "dataset": task.dataset,
                "world_size": ws, "nodes": ws // local_ws, "gpus_used": ws if cuda else 0, "tp_size": t, "dp_size": dp_size,
                "rank_R": a.rank if a.mode == "powersgd" else None, "mode": a.mode, "precision": "fp32",
                "precision_note": None, "seed": a.seed, "batch_size_per_replica": a.batch_size, "global_batch": global_batch,
                "dropout": a.dropout, "steps": len(step_times), "planned_steps": task.total_steps, "epochs": None,
                "start_powerSGD_iter": a.start_iter if a.mode == "powersgd" else None,
                "init_val_loss": init_ev["val_loss"], "final_metric": last_ev, "best_metric": best, "best_step": best_step,
                "lr_eff": lr_eff, "wall_time_s": wall, "train_time_s": train_time,
                "median_step_time_s": float(np.median(steady)) if len(steady) else None,
                "mean_step_time_s": float(np.mean(steady)) if len(steady) else None,
                "throughput_per_s": tokens / float(np.mean(steady)) if len(steady) else None,
                "throughput_unit": "tokens/s",
                "comm_bytes_per_step": float(np.mean(dp_steady)) if len(dp_steady) else None,
                "tp_bytes_per_step": float(np.mean(tp_per_step)) if len(tp_per_step) else None,
                "comm_bytes_note": "logical payload per rank passed to dist.all_reduce, steady state; not wire bytes; "
                                   "comm_bytes_per_step is the data-parallel gradient traffic, tp_bytes_per_step the "
                                   "tensor-parallel activation traffic",
                "sim_link": None, "status": status, "run_dir": str(run_dir), "params": nparams, "params_per_gpu": nlocal,
                "torch": torch.__version__,
            }
            if not cuda:
                rec["note"] = "CPU/gloo run: timing not representative"
            if a.synthetic:
                rec["note"] = "SYNTHETIC DATA smoke test - not a result"
            log_run(rec)
            emit({"type": "done", **{k: rec[k] for k in ("wall_time_s", "median_step_time_s", "comm_bytes_per_step",
                                                          "tp_bytes_per_step", "status")}})
        dist.barrier() if status == "ok" else None
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
