#!/usr/bin/env python3
"""Figures for the next-step experiments (needs matplotlib: on Perlmutter `module load pytorch` has it).

  python tools/plot_experiments.py curves  --match '_61000001$' --out figures/slowlink_curves.png        # val loss vs step
  python tools/plot_experiments.py curves  --match 'gpt_large' --x time --out figures/large_vs_time.png  # vs wall time
  python tools/plot_experiments.py bars    --match '_61000001$' --out figures/slowlink_bars.png          # step time, bytes
  python tools/plot_experiments.py seeds   --match 'gpt_.*_s[123]_1node' --out figures/seeds.png         # spread over seeds
  python tools/plot_experiments.py scaling results/scaling/* --out figures/scaling.png                  # paper-code runs

--runs-dir and --jsonl default to results/runs and results/runs.jsonl (train_ddp.py and train_tensor.py runs); point
them at an extracted bundle instead. Every figure says in its title what it shows; single-seed figures say so.
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_experiments import load_paper_run  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# the deck's palette first
COLORS = ["#1B2A41", "#2E86AB", "#E07A5F", "#3D9970", "#E0A458", "#7B6D8D", "#B8B8B8"]


# ----------------------------------------------------------------------------- data (no matplotlib needed)
def load_curves(runs_dir, match):
    """{run: [(step, elapsed_s, metric_name, value), ...]} from each matching run's metrics.jsonl eval records."""
    out = {}
    for run in sorted(os.listdir(runs_dir)):
        path = os.path.join(runs_dir, run, "metrics.jsonl")
        if not re.search(match, run) or not os.path.exists(path):
            continue
        pts = []
        for line in open(path):
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("type") != "eval":
                continue
            name = "val_loss" if "val_loss" in m else "test_acc"
            pts.append((m["step"], m.get("elapsed_s"), name, m[name]))
        if pts:
            out[run] = pts
    return out


def load_records(jsonl, match):
    rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    return [r for r in rows if re.search(match, r["run_id"])]


def best_value(r):
    b = r.get("best_metric") or {}
    return b.get("val_loss", b.get("test_acc"))


def seed_groups(rows):
    """{config: [(seed, best)]}: run names with the _s<seed> token and a trailing _<jobid> removed."""
    groups = defaultdict(list)
    for r in rows:
        cfg = re.sub(r"_\d{5,}$", "", re.sub(r"_s\d+(?=_|$)", "", r["run_id"]))
        if best_value(r) is not None:
            groups[cfg].append((r["seed"], best_value(r)))
    return dict(groups)


def scaling_series(dirs):
    """{(method, backend): sorted [(gpus, batch_ms, comm_share)]} from paper-code run folders."""
    series = defaultdict(list)
    for d in dirs:
        r = load_paper_run(d) if os.path.isdir(d) else None
        if r and r["batch"]:
            share = r["reduce"] / r["batch"] if r["reduce"] else None
            series[(r["method"], r["backend"])].append((r["workers"], r["batch"], share))
    return {k: sorted(v) for k, v in series.items()}


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    return plt


def _save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def cmd_curves(a):
    data = load_curves(a.runs_dir, a.match)
    if not data:
        sys.exit(f"no runs in {a.runs_dir} match {a.match!r}")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (run, pts) in enumerate(data.items()):
        xs = [p[0] if a.x == "step" else p[1] for p in pts]
        ax.plot(xs, [p[3] for p in pts], marker="o", ms=3, color=COLORS[i % len(COLORS)], label=run)
    metric = next(iter(data.values()))[0][2]
    ax.set_xlabel("step" if a.x == "step" else "wall time (s, includes evaluation)")
    ax.set_ylabel("validation loss" if metric == "val_loss" else "test accuracy")
    ax.set_title(a.title or f"{'Validation loss' if metric == 'val_loss' else 'Test accuracy'} by "
                            f"{'step' if a.x == 'step' else 'wall time'}")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, a.out)


def cmd_bars(a):
    rows = load_records(a.jsonl, a.match)
    if not rows:
        sys.exit(f"no records in {a.jsonl} match {a.match!r}")
    plt = _plt()
    names = [r["run_id"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.45 * len(rows) + 1.5))
    ax1.barh(names, [(r.get("median_step_time_s") or 0) * 1000 for r in rows], color=COLORS[1])
    ax1.set_xlabel("median step time (ms)")
    dp = [(r.get("comm_bytes_per_step") or 0) / 1e6 for r in rows]
    tp = [(r.get("tp_bytes_per_step") or 0) / 1e6 for r in rows]
    ax2.barh(names, dp, color=COLORS[0], label="data-parallel (gradients)")
    if any(tp):
        ax2.barh(names, tp, left=dp, color=COLORS[2], label="tensor-parallel (activations)")
    ax2.set_xscale("log")
    ax2.set_xlabel("all-reduce payload per GPU per step (MB, log scale)")
    ax2.set_yticks([])
    ax2.legend(fontsize=7, frameon=False)
    fig.suptitle(a.title or "Step time and communication per step (logical bytes, not wire traffic)")
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    _save(fig, a.out)


def cmd_seeds(a):
    groups = seed_groups(load_records(a.jsonl, a.match))
    if not groups:
        sys.exit(f"no records in {a.jsonl} match {a.match!r}")
    plt = _plt()
    names = sorted(groups)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.5))
    for i, name in enumerate(names):
        vals = [v for _, v in groups[name]]
        ax.barh(i, statistics.mean(vals), color=COLORS[6], height=0.6)
        ax.scatter(vals, [i] * len(vals), color=COLORS[0], zorder=3, s=18)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} (n={len(groups[n])})" for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("best validation metric (bar = mean over seeds, dots = each seed)")
    ax.set_title(a.title or "Spread over seeds")
    _save(fig, a.out)


def cmd_scaling(a):
    series = scaling_series(a.dirs)
    if not series:
        sys.exit("no paper-code runs with timer_summary.json in the given folders")
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, ((method, backend), pts) in enumerate(sorted(series.items())):
        style = "-" if backend == "nccl" else "--"
        kw = dict(color=COLORS[i % len(COLORS)], ls=style, marker="o", label=f"{method}, {backend}")
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], **kw)
        ax2.plot([p[0] for p in pts if p[2] is not None], [100 * p[2] for p in pts if p[2] is not None], **kw)
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("GPUs")
    ax1.set_ylabel("time per batch (ms)")
    ax2.set_ylabel("communication share of a batch (%)")
    ax1.legend(fontsize=7, frameon=False)
    fig.suptitle(a.title or "Paper code, CIFAR-10 ResNet-18: time per batch by GPU count and backend")
    _save(fig, a.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("curves", help="validation loss (or test accuracy) per run, by step or wall time")
    s.add_argument("--runs-dir", default=os.path.join(ROOT, "results", "runs"))
    s.add_argument("--match", default=".", help="regex on the run folder name")
    s.add_argument("--x", choices=["step", "time"], default="step")
    s.set_defaults(fn=cmd_curves)
    for name, fn, hlp in (("bars", cmd_bars, "median step time and bytes per step, one bar per run"),
                          ("seeds", cmd_seeds, "best validation metric per configuration over seeds")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--jsonl", default=os.path.join(ROOT, "results", "runs.jsonl"))
        s.add_argument("--match", default=".", help="regex on run_id")
        s.set_defaults(fn=fn)
    s = sub.add_parser("scaling", help="paper-code run folders: time per batch and communication share by GPU count")
    s.add_argument("dirs", nargs="+")
    s.set_defaults(fn=cmd_scaling)
    for s in sub.choices.values():
        s.add_argument("--out", required=True, help="PNG path")
        s.add_argument("--title", default=None)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
