#!/usr/bin/env python
"""Organize raw scp copies of Perlmutter results into results/organized/ and write results/MANIFEST.md.

COPY ONLY: raw files are never modified, moved or deleted. Everything is derived from the files themselves
(config.json, metrics.jsonl, step_times.csv, runs.jsonl, logs/*.out, gpu_*.csv). The only hand-typed provenance is
STATED below (what Aaron said ran where); it is used to cross-check the files and any disagreement is reported.

    python tools/organize_results.py --raw powersgd_results --out results
Re-run it after re-copying (e.g. results_batch/, logs_batch/ inside --raw); output is rebuilt from scratch in --out/organized.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict

BUSY_PCT = 10  # a GPU-second counts as "active" above this utilization
RUN_FILES = ("config.json", "metrics.jsonl", "step_times.csv")

# Stated by Aaron (not derived from files); used only to cross-check. (job, nodes, world_size, seed, global_batch, steps)
S1 = ("60622124", 1, 4, 1, 256)
STATED = {
    "smoke_gpt_ddp": S1 + (50,), "smoke_gpt_r2": S1 + (50,),
    "gpt_ddp_r0_1node": S1 + (2000,), "gpt_powersgd_r1_1node": S1 + (2000,),
    "gpt_powersgd_r2_1node": S1 + (2000,), "gpt_powersgd_r4_1node": S1 + (2000,),
    "rerun_ddp_s1": ("60626982", 1, 4, 1, 256, 2000), "rerun_r4_s1": ("60626982", 1, 4, 1, 256, 2000),
    "gpt_ddp_2node": ("60627319", 2, 8, 1, 512, 2000), "gpt_r2_2node": ("60627319", 2, 8, 1, 512, 2000),
    "gpt_r4_2node": ("60627319", 2, 8, 1, 512, 2000),
    "gpt_ddp_1000_1node": ("60623330", 1, 4, 1, 256, 1000), "gpt_r2_6000_1node": ("60623330", 1, 4, 1, 256, 6000),
    "gpt_r4_6000_1node": ("60623330", 1, 4, 1, 256, 6000),
}
REUSED = {"gpt_ddp_2node", "gpt_r2_2node", "gpt_r4_2node"}
STATED_BATCH_NAME_REUSE = "jobs 60623329 (2-node sbatch) and 60623330 (1-node sbatch) were stated to still be running when copied"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def profile(run_dir):
    """Facts about one run folder, straight from its files."""
    p = {"files": {f: os.path.exists(os.path.join(run_dir, f)) for f in RUN_FILES},
         "hash": {f: md5(os.path.join(run_dir, f)) for f in RUN_FILES if os.path.exists(os.path.join(run_dir, f))}}
    cfg = json.load(open(os.path.join(run_dir, "config.json"))) if p["files"]["config.json"] else {}
    ml = jl(os.path.join(run_dir, "metrics.jsonl"))
    p["cfg"] = cfg
    p["starts"] = [m for m in ml if m["type"] == "start"]
    p["dones"] = [m for m in ml if m["type"] == "done"]
    p["evals"] = [m for m in ml if m["type"] == "eval"]
    p["other"] = [m for m in ml if m["type"] not in ("start", "eval", "done")]
    p["step_rows"] = sum(1 for _ in open(os.path.join(run_dir, "step_times.csv"))) - 1 if p["files"]["step_times.csv"] else None
    # split metrics into blocks at each `start` line (more than one block = more than one run wrote this file)
    blocks, cur = [], None
    for m in ml:
        if m["type"] == "start":
            cur = {"start": m, "evals": [], "done": None}
            blocks.append(cur)
        elif cur is not None and m["type"] == "eval":
            cur["evals"].append(m)
        elif cur is not None and m["type"] == "done":
            cur["done"] = m
    p["blocks"] = blocks
    return p


def out_segments(txt):
    """Runs printed to a batch job's .out: one segment per `start` line, with the eval/done JSON lines that follow."""
    segs, cur = [], None
    for line in txt.splitlines():
        if not line.startswith('{"type"'):
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue  # line cut off mid-write
        if m["type"] == "start":
            cur = {"run_id": m["run_id"], "evals": [], "done": None}
            segs.append(cur)
        elif cur is not None and m["type"] == "eval":
            cur["evals"].append(m)
        elif cur is not None and m["type"] == "done":
            cur["done"] = m
    return segs


def vals(evals):
    return [e["val_loss"] for e in evals]


def best(evals):
    return min(evals, key=lambda e: e["val_loss"]) if evals else None


def gpu_stats(path):
    rows = list(csv.reader(open(path)))
    hdr, rows = rows[0], rows[1:]
    util = [int(r[2].replace("%", "").strip()) for r in rows]
    busy = [u for u in util if u > BUSY_PCT]
    gpus = sorted({r[1].strip() for r in rows})
    return {"hdr_ok": hdr[:3] == ["timestamp", " index", " utilization.gpu [%]"], "samples": len(util), "gpus": len(gpus),
            "first": rows[0][0].strip(), "last": rows[-1][0].strip(), "mean_all": sum(util) / len(util),
            "mean_busy": sum(busy) / len(busy) if busy else None, "busy_frac": len(busy) / len(util),
            "seconds": len(util) / max(len(gpus), 1)}


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    raw, out = os.path.abspath(a.raw), os.path.abspath(a.out)
    org = os.path.join(out, "organized")
    if os.path.exists(org):
        shutil.rmtree(org)  # only ever the generated output dir, never --raw

    copies = sorted(d for d in os.listdir(raw) if os.path.exists(os.path.join(raw, d, "runs.jsonl")))
    logdirs = sorted(d for d in os.listdir(raw) if d.startswith("logs") and os.path.isdir(os.path.join(raw, d)))
    records = defaultdict(list)  # run_id -> [(copy, record)]
    variants = defaultdict(list)  # run_id -> [{copy, dir, profile}]
    for c in copies:
        for r in jl(os.path.join(raw, c, "runs.jsonl")):
            records[r["run_id"]].append((c, r))
        for d in sorted(glob.glob(os.path.join(raw, c, "runs", "*"))):
            variants[os.path.basename(d)].append({"copy": c, "dir": d, "p": profile(d)})

    outs = {}
    for d in logdirs:
        for p in sorted(glob.glob(os.path.join(raw, d, "*.out"))):
            m = re.search(r"_(\d+)\.out$", p)
            if m:
                txt = open(p, errors="replace").read()
                outs[m.group(1)] = {"path": p, "dir": d, "finished": "JobID" in txt and "State" in txt, "segs": out_segments(txt)}
    # ---- per-run analysis -------------------------------------------------------------------------------------
    runs, flags = [], []
    for rid in sorted(variants):
        vs = variants[rid]
        groups = defaultdict(list)  # identical file content -> copies that hold it
        for v in vs:
            groups[tuple(sorted(v["p"]["hash"].items()))].append(v)
        distinct = list(groups.values())
        # records: identical lines across copies count once
        uniq_rec = {json.dumps(r, sort_keys=True): (c, r) for c, r in records.get(rid, [])}
        recs = [r for _, r in uniq_rec.values()]
        for grp in distinct:
            v = grp[0]
            p = v["p"]
            cfg = p["cfg"]
            st = p["starts"][0] if p["starts"] else {}
            planned = st.get("total_steps") or cfg.get("max_steps")
            ei = cfg.get("eval_interval")
            notes, validity = [], "valid"
            n_blocks = len(p["blocks"])
            steps_ev = [e["step"] for e in p["evals"]]
            collision = n_blocks > 1 or len(steps_ev) != len(set(steps_ev)) or (planned and (p["step_rows"] or 0) > planned) or len(recs) > 1
            if len(distinct) > 1:
                collision = True
                notes.append(f"folder content differs between raw copies ({', '.join(x['copy'] for g in distinct for x in g[:1])}); each version kept separately")
            expected_evals = (planned // ei) if planned and ei else None
            partial = (not p["dones"]) or (expected_evals is not None and len(p["evals"]) < expected_evals) or not p["files"]["step_times.csv"] or not recs
            is_smoke = rid.startswith("smoke_") or (planned is not None and planned <= 50)
            if collision:
                validity = "suspect collision"
            elif partial:
                validity = "partial"
            elif is_smoke:
                validity = "smoke test"
            rec = recs[0] if len(recs) == 1 else None
            r_extra = {}
            if n_blocks > 1:
                attr = []
                for i, b in enumerate(p["blocks"], 1):
                    bb = best(b["evals"])
                    fin = b["evals"][-1]["val_loss"] if b["evals"] else None
                    why = []
                    if recs and fin is not None and any(abs(fin - x["final_metric"]["val_loss"]) < 1e-9 for x in recs):
                        why.append("job " + next(x["slurm_job_id"] for x in recs if abs(fin - x["final_metric"]["val_loss"]) < 1e-9) + " (final val equals its runs.jsonl record)")
                    for jid, o in outs.items():
                        for sg in o["segs"]:
                            if sg["run_id"] == rid and vals(sg["evals"]) == vals(b["evals"]) and b["evals"]:
                                why.append(f"job {jid} (all {len(b['evals'])} eval val_loss values identical to the lines in {os.path.basename(o['path'])})")
                    attr.append({"i": i, "final": fin, "best": bb, "n": len(b["evals"]), "done": b["done"], "why": why or ["no match in runs.jsonl or .out logs"]})
                    notes.append(f"metrics block {i}: {len(b['evals'])} evals, final val {fin:.4f}, best {bb['val_loss']:.4f}@{bb['step']}, done={'yes' if b['done'] else 'no'}; attributed to: {'; '.join(attr[-1]['why'])}")
                notes.append("two complete runs wrote this one metrics.jsonl; each block is attributed above by exact match. config.json is identical either way")
                if p["files"]["step_times.csv"]:
                    rows = list(csv.reader(open(os.path.join(v["dir"], "step_times.csv"))))[1:]
                    warm = max(20, (cfg.get("start_iter") or 0) + 10) if cfg.get("timing_warmup") is None else cfg["timing_warmup"]
                    med = sorted(float(x[1]) for x in rows[warm:])
                    med = (med[len(med) // 2 - 1] + med[len(med) // 2]) / 2 if len(med) % 2 == 0 else med[len(med) // 2]
                    hit = [x["i"] for x in attr if x["done"] and abs(x["done"]["median_step_time_s"] - med) < 2e-6]
                    notes.append("step_times.csv median step time " + (f"matches block {hit[0]}'s `done` line only (the run that wrote it last)" if len(hit) == 1 else "does not single out one block") + f" (csv median {med:.6f} s)")
                r_extra["attr"] = attr
            if len(recs) > 1:
                notes.append(f"{len(recs)} runs.jsonl records for this run_id")
            out_job = None
            if not recs:
                notes.append("no runs.jsonl record (run unfinished or record not yet copied)")
                if p["evals"] and n_blocks <= 1:
                    for jid, o in outs.items():
                        for sg in o["segs"]:
                            if sg["run_id"] == rid and vals(sg["evals"])[:len(p["evals"])] == vals(p["evals"]):
                                out_job = jid
                                notes.append(f"job {jid} attributed from {os.path.basename(o['path'])}: its eval lines for this run_id begin with the same {len(p['evals'])} val_loss values")
            if not p["dones"] and not collision:
                notes.append("no `done` line in metrics.jsonl")
            if not p["files"]["step_times.csv"]:
                notes.append("no step_times.csv (written only when a run ends)")
            if expected_evals and len(p["evals"]) < expected_evals and not collision:
                notes.append(f"{len(p['evals'])}/{expected_evals} evals")
            if rid in REUSED and not collision:
                notes.append("run name is reused by sbatch job 60623329 (stated); later copies may show this folder overwritten or doubled")
            if cfg.get("mode") == "ddp":
                notes.append("config.json `rank` is a CLI default for ddp, not a PowerSGD rank")
            # file-derived vs run record cross-checks
            if rec and not collision:
                if p["evals"] and abs(p["evals"][-1]["val_loss"] - rec["final_metric"]["val_loss"]) > 1e-6:
                    notes.append("final val in metrics.jsonl differs from runs.jsonl")
                if p["step_rows"] is not None and p["step_rows"] != rec["steps"]:
                    notes.append("step_times rows differ from runs.jsonl steps")
            # check against stated provenance
            stated = STATED.get(rid)
            if stated and rec:
                got = (rec["slurm_job_id"], rec["nodes"], rec["world_size"], cfg.get("seed"), st.get("global_batch"), planned)
                if got != stated:
                    msg = f"DISAGREES with stated provenance (job,nodes,ws,seed,global_batch,steps): stated {stated}, files {got}"
                    notes.append(msg)
                    flags.append(f"`{rid}`: {msg}")
            if stated and not rec:
                notes.append(f"stated job {stated[0]}; " + ("confirmed by the .out log" if out_job == stated[0] else "not confirmed by any file"))
            b0 = best(p["evals"]) if n_blocks <= 1 else None
            runs.append({**r_extra, "id": rid, "copies": [x["copy"] for x in grp], "dir": v["dir"], "p": p, "rec": rec, "recs": recs,
                         "validity": validity, "notes": notes, "planned": planned, "best": b0,
                         "final": (p["evals"][-1] if p["evals"] and n_blocks <= 1 else None), "job": (rec or {}).get("slurm_job_id") or out_job,
                         "stated_job": stated[0] if stated else None, "device": st.get("device"), "multi": len(distinct) > 1, "var": distinct.index(grp)})

    # ---- logs and GPU CSVs -------------------------------------------------------------------------------------
    gpus, seen = [], {}
    cands = [(d, p) for d in logdirs for p in sorted(glob.glob(os.path.join(raw, d, "gpu_*.csv")))]
    cands += [("(top level)", p) for p in sorted(glob.glob(os.path.join(raw, "gpu_*.csv")))]
    for d, p in cands:
        m = re.match(r"gpu_(\d+)_(.+)\.csv", os.path.basename(p))
        h = md5(p)
        if h in seen:
            seen[h]["dups"].append(f"{d}/{os.path.basename(p)}")
            continue
        g = {"path": p, "dir": d, "job": m.group(1), "node": m.group(2), "dups": [], **gpu_stats(p)}
        seen[h] = g
        gpus.append(g)

    # ---- sessions ---------------------------------------------------------------------------------------------
    jobs = defaultdict(lambda: {"recs": [], "nodelist": None, "qos": None, "csv": [], "out": None})
    for r in runs:
        for rec in r["recs"]:
            j = jobs[rec["slurm_job_id"]]
            j["recs"].append(r["id"])
            j["nodelist"], j["qos"] = rec["slurm_nodelist"], rec["slurm_qos"]
            j["nodes"] = rec["nodes"]
    for r in runs:
        if not r["recs"] and r["job"]:
            jobs[r["job"]]["recs"].append(r["id"])
    for g in gpus:
        jobs[g["job"]]["csv"].append(g)
    for jid, o in outs.items():
        jobs[jid]["out"] = o

    def sdir(jid):
        j = jobs[jid]
        kind = "batch" if j["out"] or (j["qos"] and "interactive" not in j["qos"]) else "interactive"
        n = j.get("nodes") or (len(j["csv"]) or "?")
        return f"job{jid}_{n}node_{kind}"

    os.makedirs(org)
    placed = []
    for r in runs:
        if r["validity"] == "suspect collision":
            d = os.path.join(org, "_suspect_collision", r["id"] + (f"__v{r['var'] + 1}_{'+'.join(r['copies'])}" if r["multi"] else ""))
        elif r["job"]:
            d = os.path.join(org, sdir(r["job"]), r["id"])
        else:
            d = os.path.join(org, "_unattributed", r["id"])
        os.makedirs(d)
        for f in RUN_FILES:
            src = os.path.join(r["dir"], f)
            if os.path.exists(src):
                shutil.copy2(src, d)
        json.dump({"run_id": r["id"], "source_copies": r["copies"], "source_dir": r["dir"], "file_md5": r["p"]["hash"],
                   "runs_jsonl_records": r["recs"], "validity": r["validity"], "notes": r["notes"]}, open(os.path.join(d, "_provenance.json"), "w"), indent=1)
        r["placed"] = os.path.relpath(d, out)
    for g in gpus:
        d = os.path.join(org, sdir(g["job"]) if g["job"] in jobs else "_unattributed")
        os.makedirs(d, exist_ok=True)
        shutil.copy2(g["path"], d)
        g["placed"] = os.path.relpath(os.path.join(d, os.path.basename(g["path"])), out)
    for jid, o in outs.items():
        d = os.path.join(org, sdir(jid))
        os.makedirs(d, exist_ok=True)
        shutil.copy2(o["path"], d)

    # ---- MANIFEST.md ------------------------------------------------------------------------------------------
    L = ["# Results manifest", "",
         f"_Generated by `tools/organize_results.py` from `{os.path.relpath(raw, os.path.dirname(out))}` (raw copies: {', '.join(copies)}; logs: {', '.join(logdirs) or 'none'}). "
         "Everything below is computed from the copied files, not from the run names. Raw files are untouched; `results/organized/` holds copies._", "",
         "## Sessions", "", "| session folder | job | nodes (from runs.jsonl) | qos | GPU CSV | batch .out log | runs |", "|---|---|---|---|---|---|---|"]
    for jid in sorted(jobs):
        j = jobs[jid]
        csvs = ", ".join(f"`{os.path.basename(g['path'])}`" for g in j["csv"]) or "none"
        o = "–" if not j["out"] else f"`{os.path.basename(j['out']['path'])}` ({'complete' if j['out']['finished'] else 'PARTIAL: no sacct footer'})"
        L.append(f"| `{sdir(jid)}` | {jid} | {j['nodelist'] or (', '.join(g['node'] for g in j['csv']) + ' (from CSV names; no runs.jsonl record)' if j['csv'] else 'unknown')} | {j['qos'] or '–'} | {csvs} | {o} | {', '.join(f'`{x}`' for x in j['recs']) or '–'} |")
    L += ["", "## Runs", "",
          "Metrics (best val loss and step, final val loss) are computed from `metrics.jsonl`; step time, comm and GPU-hours come from the `runs.jsonl` record "
          "(blank where there is no single clean record). `validity`: valid, smoke test, partial, or suspect collision.", "",
          "| run_id | session | nodes | world | seed | global batch | steps (planned) | final val | best val (step) | median step | comm/step | GPU-h | GPU | validity |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"valid": 0, "smoke test": 1, "partial": 2, "suspect collision": 3}
    for r in sorted(runs, key=lambda r: (order[r["validity"]], r["job"] or "", r["id"])):
        p, rec, st = r["p"], r["rec"], (r["p"]["starts"][0] if r["p"]["starts"] else {})
        nodes = rec["nodes"] if rec else "–"
        clean = rec is not None and r["validity"] != "suspect collision"
        ms = f"{rec['median_step_time_s'] * 1000:.1f} ms" if clean else "–"
        mb = f"{rec['comm_bytes_per_step'] / 1e6:.3f} MB" if clean else "–"
        gh = f"{rec['gpu_hours']:.3f}" if clean else "–"
        steps = (rec["steps"] if rec else p["step_rows"]) if p["evals"] else None
        steps = "–" if steps is None else steps
        best_s = f"{r['best']['val_loss']:.3f} ({r['best']['step']})" if r["best"] else "–"
        if r["validity"] == "partial" and r["best"]:
            best_s += " so far"
        tag = f" (v{r['var'] + 1})" if r["multi"] else ""
        sess = r["placed"].split("/")[1]
        L.append(f"| `{r['id']}`{tag} | `{sess}` | {nodes} | {st.get('world_size', '–')} | {p['cfg'].get('seed', '–')} | "
                 f"{st.get('global_batch', '–')} | {steps} ({r['planned']}) | {f3(r['final']['val_loss']) if r['final'] and r['validity'] != 'partial' else '–'} | {best_s} | "
                 f"{ms} | {mb} | {gh} | {(r['device'] or '–').replace('NVIDIA ', '')} | **{r['validity']}** |")
    L += ["", "### Notes per run", ""]
    for r in sorted(runs, key=lambda r: (order[r["validity"]], r["job"] or "", r["id"])):
        if r["notes"]:
            L.append(f"- `{r['id']}`{' (v' + str(r['var'] + 1) + ' of ' + '+'.join(r['copies']) + ')' if r['multi'] else ''} [{r['validity']}], in `{r['placed']}`:")
            L += [f"  - {n}" for n in r["notes"]]
    for r in runs:
        if r.get("attr"):
            L += ["", f"## `{r['id']}`: two runs in one file, attributed", "",
                  "Each block of `metrics.jsonl` (one per `start` line) is matched to its source by exact equality: to the `runs.jsonl` record's final val loss, "
                  "and to the eval lines a batch job printed in its `.out` log. Not merged; `step_times.csv` cannot be split.", "",
                  "| block | attributed to | evals | final val | best val (step) | wall (done line) | median step (done line) |", "|---|---|---|---|---|---|---|"]
            for x in r["attr"]:
                d = x["done"] or {}
                L.append(f"| {x['i']} | {'; '.join(x['why'])} | {x['n']} | {x['final']:.4f} | {x['best']['val_loss']:.4f} ({x['best']['step']}) | "
                         f"{d.get('wall_time_s', float('nan')):.1f} s | {d.get('median_step_time_s', float('nan')) * 1000:.2f} ms |")
            if len(r["attr"]) == 2 and all(len(x["why"]) == 1 and not x["why"][0].startswith("no match") for x in r["attr"]):
                a, b = r["attr"]
                L += ["", f"Both attributed to different jobs on the same hardware ({r['device']}), same seed and config: final val differs by {abs(a['final'] - b['final']):.4f} "
                      f"({100 * abs(a['final'] - b['final']) / a['final']:.2f}%), best val by {abs(a['best']['val_loss'] - b['best']['val_loss']):.4f}. One pair of runs, so this is a single reproducibility data point."]
    L += ["", "## GPU utilization", "",
          f"Active = a 1-second GPU sample with utilization **above {BUSY_PCT}%**. `mean (all)` averages every sample in the file (including idle time); "
          "`mean (active)` averages only the active samples. All times are the node's local (Pacific) clock.", "",
          "| csv | job | node | GPUs | span | seconds | mean (all) | mean (active) | active share | completeness |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in sorted(gpus, key=lambda g: (g["job"], g["node"])):
        j = jobs[g["job"]]
        comp = ("PARTIAL (batch job still running when copied)" if j["out"] and not j["out"]["finished"] else
                "complete batch job" if j["out"] else "interactive allocation (end of file not independently verified)")
        mb = f"{g['mean_busy']:.1f}%" if g["mean_busy"] is not None else "–"
        dup = " (identical copy also at top level)" if g["dups"] else ""
        L.append(f"| `{os.path.basename(g['path'])}`{dup} | {g['job']} | {g['node']} | {g['gpus']} | {g['first']} to {g['last']} | "
                 f"{g['seconds']:.0f} | {g['mean_all']:.1f}% | {mb} | {100 * g['busy_frac']:.0f}% | {comp} |")
    nocsv = [jid for jid in jobs if not jobs[jid]["csv"] and jobs[jid]["recs"]]
    for jid in nocsv:
        L.append(f"\nNo GPU CSV in the copies for job {jid} ({', '.join(jobs[jid]['recs'])}).")
    L += ["", "## Provenance check against what Aaron stated", ""]
    L += [f"- {f}" for f in flags] or ["- No run disagrees with the stated (job, nodes, world size, seed, global batch, steps)."]
    L += [f"- Stated: {STATED_BATCH_NAME_REUSE}. Run names `gpt_ddp_2node`, `gpt_r2_2node`, `gpt_r4_2node` are reused by the interactive job 60627319 and the 2-node sbatch 60623329."
          + (" Confirmed for `gpt_ddp_2node`: one metrics block matches the 60623329 .out log exactly." if any(r.get("attr") for r in runs) else "")]
    grp = defaultdict(list)
    for r in runs:
        if r["validity"] == "valid" and r["rec"]:
            c, st = r["p"]["cfg"], r["p"]["starts"][0]
            grp[(c["mode"], c["rank"] if c["mode"] == "powersgd" else None, st["world_size"], r["planned"], st["global_batch"], c["seed"])].append(r)
    reps = {k: v for k, v in grp.items() if len(v) > 1}
    L += ["", "## Repeat runs of the same configuration", ""]
    if reps:
        L += ["Same mode, rank, world size, steps, global batch and seed, different job. Shows how much two identical runs differ.", "",
              "| config | run | job | final val | best val (step) | median step |", "|---|---|---|---|---|---|"]
        for k, v in sorted(reps.items(), key=lambda kv: str(kv[0])):
            for r in v:
                L.append(f"| {k[0]}{' r' + str(k[1]) if k[1] else ''}, world {k[2]}, {k[3]} steps, batch {k[4]}, seed {k[5]} | `{r['id']}` | {r['job']} | "
                         f"{r['final']['val_loss']:.4f} | {r['best']['val_loss']:.4f} ({r['best']['step']}) | {r['rec']['median_step_time_s'] * 1000:.2f} ms |")
    else:
        L.append("None in these copies.")
    lrs = defaultdict(set)
    for r in runs:
        if r["validity"] == "valid" and r["p"]["starts"]:
            lrs[r["p"]["starts"][0]["world_size"]].add(r["p"]["starts"][0]["lr_eff"])
    L += ["", "## Effective learning rate by world size (from `start` lines)", ""] + [f"- world size {w}: lr_eff {sorted(v)}" for w, v in sorted(lrs.items())]
    if len(lrs) > 1 and len({tuple(sorted(v)) for v in lrs.values()}) == 1:
        L += ["", f"- **DISAGREES with stated provenance:** 2-node runs were stated to use a scaled LR, but `lr_eff` is identical ({sorted(next(iter(lrs.values())))}) at every world size "
              "(the trainer scales lr with global batch for ResNet-18 only, not for NanoGPT). Batch size still differs (512 vs 256)."]
    L += ["", "## Not run (no files exist)", "", "- ResNet-18/CIFAR-10 with `train_ddp.py`, FP8 (A100 has no native FP8), tensor parallelism, LSTM in `train_ddp.py`.", ""]
    open(os.path.join(out, "MANIFEST.md"), "w").write("\n".join(L))
    print(f"wrote {os.path.relpath(os.path.join(out, 'MANIFEST.md'))}; {len(runs)} run variants, {len(gpus)} GPU CSVs, {len(jobs)} jobs")
    for r in runs:
        print(f"  {r['validity']:18s} {r['id']:26s} -> {r['placed']}")


if __name__ == "__main__":
    main()
