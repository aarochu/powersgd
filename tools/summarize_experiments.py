#!/usr/bin/env python3
"""Summarize experiment outputs as markdown tables (prints to stdout; writes nothing).

  python tools/summarize_experiments.py paper results/scaling/*            # paper-code run folders
  python tools/summarize_experiments.py runs --match 'gpt_.*_s[23]_1node'  # train_ddp.py records in results/runs.jsonl
  python tools/summarize_experiments.py gpu logs/gpu_61000001_*.csv        # nvidia-smi CSVs

Numbers are copied from the files. The only arithmetic is per-run ratios, changes against a reference run, and the
mean [min, max] over seeds where a table says so.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict


def fmt(x, spec=".2f", none="-"):
    return none if x is None else format(x, spec)


def change(new, ref):
    return None if new is None or not ref else 100 * (new - ref) / ref


# ----------------------------------------------------------------------------- nvidia-smi CSVs
def gpu_stats(paths):
    """Mean utilization, mean over active samples (>10%), active fraction and max memory, over all GPUs in the files."""
    util, mem = [], []
    gpus = set()
    for path in paths:
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) < 4 or not row[0].strip() or row[0].startswith("timestamp"):
                    continue
                try:
                    u = float(row[2].strip().rstrip("%").strip())
                    m = float(row[3].strip().split()[0])
                except ValueError:  # e.g. a half-written last line
                    continue
                gpus.add((path, row[1].strip()))
                util.append(u)
                mem.append(m)
    if not util:
        return None
    active = [u for u in util if u > 10]
    return {"samples": len(util), "gpus": len(gpus), "mean": statistics.mean(util),
            "active_mean": statistics.mean(active) if active else 0.0, "active_frac": len(active) / len(util),
            "max_mem_mib": max(mem)}


def cmd_gpu(a):
    print("| file | GPUs | samples | util mean | util when active (>10%) | active share | max memory |")
    print("|---|---|---|---|---|---|---|")
    for p in a.csv:
        s = gpu_stats([p])
        if s is None:
            print(f"| {os.path.basename(p)} | - | 0 | - | - | - | - |")
            continue
        print(f"| {os.path.basename(p)} | {s['gpus']} | {s['samples']} | {s['mean']:.1f}% | {s['active_mean']:.1f}% | "
              f"{100 * s['active_frac']:.0f}% | {s['max_mem_mib']:.0f} MiB |")


# ----------------------------------------------------------------------------- paper-code run folders
def load_paper_run(d):
    if not os.path.exists(os.path.join(d, "timer_summary.json")) or not os.path.exists(os.path.join(d, "config.json")):
        return None
    cfg = json.load(open(os.path.join(d, "config.json")))
    t = json.load(open(os.path.join(d, "timer_summary.json")))
    avg = lambda k: t[k]["average_duration"] * 1000 if k in t else None
    final, bits = {}, None
    final_epoch = -1.0
    with open(os.path.join(d, "metrics.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("bits") is not None:
                bits = max(bits or 0, m["bits"])
            if m.get("split") != "test" or m.get("epoch") is None:
                continue
            if m["epoch"] > final_epoch:
                final_epoch, final = m["epoch"], {}
            if m["epoch"] == final_epoch:
                final[m["name"]] = m["value"]
    done = os.path.join(d, "DONE")
    fb = (avg("batch.forward") or 0) + (avg("batch.backward") or 0)
    return {
        "dir": d, "run": os.path.basename(d.rstrip("/")), "task": cfg.get("task_name"),
        "backend": cfg.get("distributed_backend"), "workers": cfg.get("n_workers") or cfg.get("world_size"),
        "seed": cfg.get("seed"), "epochs": cfg.get("num_epochs"), "method": cfg.get("reducer_name") or "?",
        "batch": avg("batch"), "reduce": avg("batch.reduce"), "fwd_bwd": fb or None, "final_epoch": final_epoch,
        "metrics": final, "bits": bits, "done": json.load(open(done))["wallclock_seconds"] if os.path.exists(done) else None,
        "gpu": gpu_stats(sorted(glob.glob(os.path.join(d, "gpu_*.csv")))),
    }


def quality(r, prefix):
    m = r["metrics"]
    if f"{prefix}_accuracy" in m:
        return f"{100 * m[prefix + '_accuracy']:.1f}%"
    if f"{prefix}_perplexity" in m:
        return f"{m[prefix + '_perplexity']:.1f} ppl"
    return "-"


def cmd_paper(a):
    runs, skipped = [], []
    for d in a.dirs:
        if os.path.isdir(d):
            r = load_paper_run(d)
            (runs if r else skipped).append(r or d)
    runs.sort(key=lambda r: (r["task"] or "", r["backend"] or "", r["workers"] or 0, r["epochs"] or 0, r["seed"] or 0, r["method"]))
    print("| run | method | backend | GPUs | seed | epochs | batch | comm (`batch.reduce`) | comm / batch | fwd+bwd | data sent | "
          "last | runavg | GPU util when active |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        share = 100 * r["reduce"] / r["batch"] if r["reduce"] and r["batch"] else None
        gb = r["bits"] / 8e9 if r["bits"] else None
        g = r["gpu"]
        print(f"| {r['run']} | {r['method']} | {r['backend']} | {r['workers']} | {r['seed']} | {r['epochs']} | "
              f"{fmt(r['batch'])} ms | {fmt(r['reduce'])} ms | {fmt(share, '.1f')}% | {fmt(r['fwd_bwd'])} ms | "
              f"{fmt(gb, '.3f')} GB | {quality(r, 'last')} | {quality(r, 'runavg')} | "
              f"{fmt(g['active_mean'] if g else None, '.0f')}{'%' if g else ''} |")
    for d in skipped:
        print(f"\nskipped (no timer_summary.json/config.json yet): {d}")

    # time per batch against SGD with the same task, backend, GPU count, seed and epochs
    groups = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["backend"], r["workers"], r["seed"], r["epochs"])].append(r)
    lines = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ref = next((r for r in rs if r["method"] == "SGD"), None)
        for r in rs:
            if ref is None or r is ref:
                continue
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {r['method']} | {fmt(ref['batch'])} ms | "
                         f"{fmt(r['batch'])} ms | {fmt(change(r['batch'], ref['batch']), '+.1f')}% | "
                         f"{fmt(ref['bits'] / r['bits'] if ref['bits'] and r['bits'] else None, '.1f')}x |")
    if lines:
        print("\nAgainst SGD in the same setting (task, backend, GPUs, seed, epochs):\n")
        print("| task | backend | GPUs | seed | epochs | method | SGD batch | method batch | change | less data |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        print("\n".join(lines))

    # mean [min, max] over seeds, only where a setting has more than one seed
    seeds = defaultdict(list)
    for r in runs:
        seeds[(r["task"], r["backend"], r["workers"], r["epochs"], r["method"])].append(r)
    multi = {k: v for k, v in seeds.items() if len(v) > 1}
    if multi:
        print("\nOver seeds, mean [min, max]:\n")
        print("| task | backend | GPUs | epochs | method | seeds | batch (ms) | last accuracy / perplexity |")
        print("|---|---|---|---|---|---|---|---|")
        for k, rs in sorted(multi.items(), key=lambda kv: str(kv[0])):
            b = [r["batch"] for r in rs if r["batch"]]
            q = [r["metrics"].get("last_accuracy", r["metrics"].get("last_perplexity")) for r in rs]
            q = [x for x in q if x is not None]
            qs = "-" if not q else (f"{100 * statistics.mean(q):.2f}% [{100 * min(q):.2f}, {100 * max(q):.2f}]"
                                    if "last_accuracy" in rs[0]["metrics"] else
                                    f"{statistics.mean(q):.2f} [{min(q):.2f}, {max(q):.2f}]")
            print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {len(rs)} | "
                  f"{statistics.mean(b):.2f} [{min(b):.2f}, {max(b):.2f}] | {qs} |")


# ----------------------------------------------------------------------------- train_ddp.py runs.jsonl
def best_of(r):
    b = r.get("best_metric") or {}
    at = f" @ {r['best_step']}" if r.get("best_step") else ""  # best_step is only logged by the patched trainer
    if "val_loss" in b:
        return f"{b['val_loss']:.3f}{at}"
    if "test_acc" in b:
        return f"{100 * b['test_acc']:.2f}%{at}"
    return "-"


def final_of(r):
    f = r.get("final_metric") or {}
    return f"{f['val_loss']:.3f}" if "val_loss" in f else (f"{100 * f['test_acc']:.2f}%" if "test_acc" in f else "-")


def cmd_runs(a):
    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    rows = [r for r in rows if re.search(a.match, r["run_id"])]
    if not rows:
        sys.exit(f"no records in {a.jsonl} match {a.match!r}")
    print("| run | job | mode | R | precision | modeled link | GPUs | params | steps | best val | final | median step | "
          "bytes/step | GPU-h | status |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sim = r.get("sim_link")
        link = f"{sim['bandwidth_gbps']:g} Gbit/s, {sim['latency_us']:g} us" if sim else "none"
        step = r["median_step_time_s"] * 1000 if r.get("median_step_time_s") else None
        mb = r["comm_bytes_per_step"] / 1e6 if r.get("comm_bytes_per_step") else None
        print(f"| {r['run_id']} | {r.get('slurm_job_id') or '-'} | {r['mode']} | {r.get('rank_R') or '-'} | {r['precision']} | "
              f"{link} | {r['world_size']} | {r['params'] / 1e6:.1f}M | {r['steps']} | {best_of(r)} | {final_of(r)} | "
              f"{fmt(step, '.1f')} ms | {fmt(mb, '.3f')} MB | {fmt(r.get('gpu_hours'), '.3f')} | {r['status']} |")

    # within one Slurm job (same hardware): each run against DDP / fp32 on the same modeled link (or none), with the
    # same seed, model size, GPU count and planned steps
    link = lambda r: (r["sim_link"]["bandwidth_gbps"], r["sim_link"]["latency_us"]) if r.get("sim_link") else None
    key = lambda r: (r.get("slurm_job_id"), r["seed"], r["params"], r["world_size"], r.get("planned_steps"), link(r))
    refs = {key(r): r for r in rows if r["mode"] == "ddp" and r["precision"] == "fp32"}
    lines = []
    for r in rows:
        ref = refs.get(key(r))
        if ref is None or r is ref:
            continue
        rs, fs = r.get("median_step_time_s"), ref.get("median_step_time_s")
        rb, fb = r.get("comm_bytes_per_step"), ref.get("comm_bytes_per_step")
        lines.append(f"| {r['run_id']} | {ref['run_id']} | {fmt(change(rs, fs), '+.1f')}% | "
                     f"{fmt(fb / rb if rb and fb else None, '.1f')}x | {best_of(r)} vs {best_of(ref)} |")
    if lines:
        print("\nWithin each job, against DDP / fp32 on the same modeled link (same seed, model size, GPU count and steps):\n")
        print("| run | reference | median step change | less data per step | best val (run vs reference) |")
        print("|---|---|---|---|---|")
        print("\n".join(lines))
    print("\nbytes/step is the logical payload handed to all-reduce per GPU, not wire traffic; fp8 runs are emulated, "
          "so their bytes and step times are fp32's.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("paper", help="paper-code run folders (config.json, timer_summary.json, metrics.jsonl)")
    s.add_argument("dirs", nargs="+")
    s.set_defaults(fn=cmd_paper)
    s = sub.add_parser("runs", help="train_ddp.py records in a runs.jsonl")
    s.add_argument("--jsonl", default=os.path.join(os.path.dirname(__file__), "..", "results", "runs.jsonl"))
    s.add_argument("--match", default=".", help="regex on run_id")
    s.set_defaults(fn=cmd_runs)
    s = sub.add_parser("gpu", help="nvidia-smi CSVs")
    s.add_argument("csv", nargs="+")
    s.set_defaults(fn=cmd_gpu)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
