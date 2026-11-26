#!/bin/bash
# Correctness and smoke test for train_tensor.py, on one GPU node. Run it before scripts/nanogpt_tp.sbatch:
#   salloc -N 1 -C gpu -G 4 -q interactive -t 00:20:00 -A <your project>
#   bash scripts/smoke_tp.sh
# 8 short runs, about 5 minutes. Prints PASS / FAIL for:
#   1. --tp-size 1 reproduces train_ddp.py (same seed and data): final validation loss within 1e-3
#   2. the split forward pass: validation loss before training matches the unsplit model (within 1e-4)
#   3. the split backward pass: with dropout off, validation loss after 40 steps matches the unsplit model (within 1e-2;
#      float summation order differs, so not bit-identical), and the loss really dropped (so the match means something)
#   4. tensor-parallel bytes per step = 4 x layers x batch x sequence x width x 4 bytes, exactly
#   5. data-parallel bytes per step (DDP, 2+ replicas) = 4 bytes x parameters held per GPU, exactly
# Checks 1-3 use the real Shakespeare data: on random tokens every model ends at the same loss floor (ln 65 = 4.174)
# after a few steps, so a match proves nothing. Each split run is compared with an unsplit run that has the SAME
# data-parallel layout and per-replica batch, so both see the same training and validation batches:
#   tp 2 on 4 GPUs (2 replicas)  vs  tp 1 on 2 GPUs (2 replicas)
#   tp 4 on 4 GPUs (1 replica)   vs  tp 1 on 1 GPU  (1 replica)
# tp 4 needs a head count divisible by 4, so that pair uses --n-head 12 (head width 32; the parameter count does not
# depend on the head count). The default 6-head model can only be split 2 ways.
# Writes to results/smoke_tp/ and results/smoke_tp_runs.jsonl, never to the real results.
set -u
export PROJ=${PROJ:-$SCRATCH/powersgd}
source "$PROJ/scripts/common.sh"
export RUNS_JSONL=$PROJ/results/smoke_tp_runs.jsonl
# keep an earlier smoke log, but start a fresh one, so a run that fails now cannot PASS on an old record
[[ -f $RUNS_JSONL ]] && mv "$RUNS_JSONL" "${RUNS_JSONL%.jsonl}_before_$(date +%m%d-%H%M%S).jsonl"
OUT=$PROJ/results/smoke_tp
REAL="--max-steps 40 --eval-interval 40 --start-iter 5 --timing-warmup 15 --out-dir $OUT"
SYN="--synthetic $REAL"
EQ="$REAL --dropout 0 --batch-size 64"   # equivalence pairs: no dropout, same per-replica batch

run() {  # [GPUS_PER_NODE=n] run <trainer> <name> <args...>
  local script=$1 name=$2; shift 2
  echo "=== $name (${GPUS_PER_NODE} GPUs)"
  launch_with "$script" "$@" --run-name "$name" 2>&1 | grep -E '"type": "done"|Error|error' | tail -2
}

run train_ddp.py    smoketp_ddp_ref   --model nanogpt $REAL
run train_tensor.py smoketp_tp1       --tp-size 1 $REAL
GPUS_PER_NODE=2 run train_tensor.py smoketp_eq2_tp1 --tp-size 1 $EQ
run train_tensor.py smoketp_eq2_tp2   --tp-size 2 $EQ
GPUS_PER_NODE=1 run train_tensor.py smoketp_eq4_tp1 --tp-size 1 --n-head 12 $EQ
run train_tensor.py smoketp_eq4_tp4   --tp-size 4 --n-head 12 $EQ
run train_tensor.py smoketp_tp2_r4    --tp-size 2 --mode powersgd --rank 4 $SYN
run train_tensor.py smoketp_tp2_large --tp-size 2 --n-layer 12 --n-head 12 --n-embd 768 $SYN

python - <<'EOF'
import json, os
rows = {}
for l in open(os.environ["RUNS_JSONL"]):
    r = json.loads(l)
    rows[r["run_id"]] = r  # the latest record of each name wins

def val(n):
    return ((rows.get(n) or {}).get("final_metric") or {}).get("val_loss")

def init(n):
    return (rows.get(n) or {}).get("init_val_loss")

def check(ok, what, detail):
    print(f'{"PASS" if ok else "FAIL"}  {what}: {detail}')

print(f'{"run":20s} {"status":10s} {"tp":>3s} {"dp":>3s} {"params/GPU":>11s} {"step ms":>8s} {"DP bytes":>11s} '
      f'{"TP bytes":>11s} {"init val":>9s} {"val":>7s}')
for k, r in rows.items():
    if not k.startswith("smoketp_"):
        continue
    step = r["median_step_time_s"] * 1000 if r.get("median_step_time_s") else float("nan")
    print(f'{k:20s} {r["status"][:10]:10s} {r.get("tp_size") or 1:>3} {r.get("dp_size") or r.get("world_size") or 0:>3} '
          f'{r.get("params_per_gpu") or r["params"]:>11,d} {step:8.1f} '
          f'{r.get("comm_bytes_per_step") or 0:11.4e} {r.get("tp_bytes_per_step") or 0:11.4e} '
          f'{init(k) or float("nan"):9.4f} {val(k) or float("nan"):7.4f}')

core = ("smoketp_ddp_ref", "smoketp_tp1", "smoketp_eq2_tp1", "smoketp_eq2_tp2", "smoketp_eq4_tp1", "smoketp_eq4_tp4",
        "smoketp_tp2_r4", "smoketp_tp2_large")
missing = [n for n in core if n not in rows or rows[n]["status"] != "ok"]
check(not missing, "all runs finished", "ok" if not missing else f"missing or failed: {missing}")
if val("smoketp_ddp_ref") is not None and val("smoketp_tp1") is not None:
    d = abs(val("smoketp_ddp_ref") - val("smoketp_tp1"))
    check(d < 1e-3, "tp 1 reproduces train_ddp.py", f"|delta val loss| = {d:.2e}")
for ref, split in (("smoketp_eq2_tp1", "smoketp_eq2_tp2"), ("smoketp_eq4_tp1", "smoketp_eq4_tp4")):
    if init(ref) is None or init(split) is None or val(ref) is None or val(split) is None:
        check(False, f"{split} vs {ref}", "a run is missing")
        continue
    d = abs(init(split) - init(ref))
    check(d < 1e-4, f"split forward pass ({split})", f"|delta init val loss| vs {ref} = {d:.2e}")
    d = abs(val(split) - val(ref))
    check(d < 1e-2, f"split backward pass ({split})", f"|delta val loss after 40 steps| vs {ref} = {d:.2e}")
    drop = init(ref) - val(ref)
    check(drop > 0.3, f"training moved ({ref})", f"val loss {init(ref):.3f} -> {val(ref):.3f} (needs a drop > 0.3)")
for n, r in rows.items():
    if not n.startswith("smoketp_") or (r.get("tp_size") or 1) < 2 or r["status"] != "ok":
        continue
    with open(os.path.join(r["run_dir"], "config.json")) as f:
        c = json.load(f)
    expect = 4 * c["n_layer"] * r["batch_size_per_replica"] * c["block_size"] * c["n_embd"] * 4
    check(r["tp_bytes_per_step"] == expect, f"TP bytes ({n})", f'{r["tp_bytes_per_step"]:.0f} vs expected {expect}')
    if r["mode"] == "ddp" and r["dp_size"] > 1:  # with one replica there is no data-parallel traffic to check
        check(r["comm_bytes_per_step"] == 4 * r["params_per_gpu"], f"DP bytes ({n})",
              f'{r["comm_bytes_per_step"]:.0f} vs 4 x {r["params_per_gpu"]} params')
EOF
