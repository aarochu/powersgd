#!/bin/bash
# Correctness and smoke test for train_tensor.py, on one GPU node. Run it before scripts/nanogpt_tp.sbatch:
#   salloc -N 1 -C gpu -G 4 -q interactive -t 00:20:00 -A <your project>
#   bash scripts/smoke_tp.sh
# Synthetic data, 40 steps per run, about 5-10 minutes. Prints PASS / FAIL for:
#   1. --tp-size 1 reproduces train_ddp.py (same seed and data): final validation loss within 1e-3
#   2. the split forward pass: validation loss before training is the same at tp 1, 2 and 4 (within 1e-4)
#   3. the split backward pass: with dropout off, final validation loss at tp 2 and 4 matches tp 1 (within 1e-2;
#      float summation order differs, so not bit-identical)
#   4. tensor-parallel bytes per step = 4 x layers x batch x sequence x width x 4 bytes, exactly
#   5. data-parallel bytes per step (DDP, 2+ replicas) = 4 bytes x parameters held per GPU, exactly
# Writes to results/smoke_tp/ and results/smoke_tp_runs.jsonl, never to the real results.
set -u
export PROJ=${PROJ:-$SCRATCH/powersgd}
source "$PROJ/scripts/common.sh"
export RUNS_JSONL=$PROJ/results/smoke_tp_runs.jsonl
OUT=$PROJ/results/smoke_tp
S="--synthetic --max-steps 40 --eval-interval 40 --start-iter 5 --timing-warmup 15 --out-dir $OUT"

run() {  # run <trainer> <name> <args...>
  local script=$1 name=$2; shift 2
  echo "=== $name"
  launch_with "$script" $S "$@" --run-name "$name" 2>&1 | grep -E '"type": "done"|Error|error' | tail -2
}

run train_ddp.py    smoketp_ddp_ref        --model nanogpt
run train_tensor.py smoketp_tp1            --tp-size 1
run train_tensor.py smoketp_tp1_nodrop     --tp-size 1 --dropout 0
run train_tensor.py smoketp_tp2_nodrop     --tp-size 2 --dropout 0
run train_tensor.py smoketp_tp4_nodrop     --tp-size 4 --dropout 0
run train_tensor.py smoketp_tp2_r4         --tp-size 2 --mode powersgd --rank 4
run train_tensor.py smoketp_tp2_large      --tp-size 2 --n-layer 12 --n-head 12 --n-embd 768

python - <<'EOF'
import json, os
rows = {}
for l in open(os.environ["RUNS_JSONL"]):
    r = json.loads(l)
    rows[r["run_id"]] = r  # the latest record of each name wins

def val(n):
    return ((rows.get(n) or {}).get("final_metric") or {}).get("val_loss")

def check(ok, what, detail):
    print(f'{"PASS" if ok else "FAIL"}  {what}: {detail}')

print(f'{"run":22s} {"status":10s} {"tp":>3s} {"params/GPU":>11s} {"step ms":>8s} {"DP bytes":>11s} {"TP bytes":>11s} {"init val":>9s} {"val":>7s}')
for k, r in rows.items():
    if not k.startswith("smoketp_"):
        continue
    step = r["median_step_time_s"] * 1000 if r.get("median_step_time_s") else float("nan")
    print(f'{k:22s} {r["status"][:10]:10s} {r.get("tp_size") or 1:>3} {r.get("params_per_gpu") or r["params"]:>11,d} {step:8.1f} '
          f'{r.get("comm_bytes_per_step") or 0:11.4e} {r.get("tp_bytes_per_step") or 0:11.4e} '
          f'{r.get("init_val_loss") or float("nan"):9.4f} {val(k) or float("nan"):7.4f}')

missing = [n for n in ("smoketp_ddp_ref", "smoketp_tp1", "smoketp_tp1_nodrop", "smoketp_tp2_nodrop", "smoketp_tp4_nodrop")
           if n not in rows or rows[n]["status"] != "ok"]
check(not missing, "all core runs finished", "ok" if not missing else f"missing or failed: {missing}")
if val("smoketp_ddp_ref") is not None and val("smoketp_tp1") is not None:
    d = abs(val("smoketp_ddp_ref") - val("smoketp_tp1"))
    check(d < 1e-3, "tp 1 reproduces train_ddp.py", f"|delta val loss| = {d:.2e}")
inits = [rows[n]["init_val_loss"] for n in ("smoketp_tp1_nodrop", "smoketp_tp2_nodrop", "smoketp_tp4_nodrop") if n in rows]
if len(inits) == 3:
    d = max(inits) - min(inits)
    check(d < 1e-4, "split forward pass", f"init val loss spread over tp 1/2/4 = {d:.2e}")
for n in ("smoketp_tp2_nodrop", "smoketp_tp4_nodrop"):
    if val(n) is not None and val("smoketp_tp1_nodrop") is not None:
        d = abs(val(n) - val("smoketp_tp1_nodrop"))
        check(d < 1e-2, f"split backward pass ({n})", f"|delta val loss| vs tp 1 = {d:.2e}")
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
