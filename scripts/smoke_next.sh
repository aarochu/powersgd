#!/bin/bash
# GPU smoke test for everything in docs/NEXT_EXPERIMENTS.md. Run it before any of those jobs:
#   salloc -N 1 -C gpu -G 4 -q interactive -t 00:20:00 -A <your project>
#   bash scripts/smoke_next.sh
# Synthetic data, 40 steps per config, about 5-10 minutes. Checks the trainer patch on a GPU (best_step / lr_eff
# logging has only run on CPU so far), the fp16/bf16/fp8 hooks, the modeled slow link, and that the larger NanoGPT
# sizes fit in memory. Writes to results/smoke_next/ and results/smoke_next_runs.jsonl, never to the real results.
set -u
export PROJ=${PROJ:-$SCRATCH/powersgd}
source "$PROJ/scripts/common.sh"
export RUNS_JSONL=$PROJ/results/smoke_next_runs.jsonl
OUT=$PROJ/results/smoke_next
S="--model nanogpt --synthetic --max-steps 40 --eval-interval 40 --start-iter 5 --timing-warmup 15"

run() {  # run <name> <args...>
  local name=$1; shift
  echo "=== $name"
  launch $S "$@" --run-name "$name" --out-dir "$OUT" 2>&1 | grep -E '"type": "done"|Error|error' | tail -2
}

run smoke_ddp_fp32
run smoke_r2_fp32      --mode powersgd --rank 2
run smoke_ddp_fp16     --precision fp16
run smoke_ddp_bf16     --precision bf16
run smoke_ddp_fp8e4m3  --precision fp8_e4m3
run smoke_ddp_fp8e5m2  --precision fp8_e5m2
run smoke_r4_fp16      --mode powersgd --rank 4 --precision fp16
run smoke_ddp_bw10     --sim-bandwidth-gbps 10
run smoke_r2_bw10      --mode powersgd --rank 2 --sim-bandwidth-gbps 10
run smoke_medium_ddp   --n-layer 8 --n-head 8 --n-embd 512
run smoke_large_ddp    --n-layer 12 --n-head 12 --n-embd 768
run smoke_large_r4     --n-layer 12 --n-head 12 --n-embd 768 --mode powersgd --rank 4

python - <<'EOF'
import json, os
rows = {}
for l in open(os.environ["RUNS_JSONL"]):
    r = json.loads(l)
    rows[r["run_id"]] = r
print(f'{"run":20s} {"status":10s} {"params":>11s} {"step ms":>8s} {"bytes/step":>11s} {"modeled delay s":>15s}')
for k, r in rows.items():
    sim = (r.get("sim_link") or {}).get("modeled_delay_s_total")
    step = r["median_step_time_s"] * 1000 if r.get("median_step_time_s") else float("nan")
    print(f'{k:20s} {r["status"][:10]:10s} {r["params"]:>11,d} {step:8.1f} {r["comm_bytes_per_step"] or 0:11.3e} {sim if sim is not None else "-":>15}')
# the modeled delay should add about 2(N-1)/N * bytes / (10 Gbit/s) per step to DDP
if "smoke_ddp_fp32" in rows and "smoke_ddp_bw10" in rows:
    a, b = rows["smoke_ddp_fp32"], rows["smoke_ddp_bw10"]
    n = a["world_size"]
    expect = 2 * (n - 1) / n * a["comm_bytes_per_step"] / (10e9 / 8) * 1000
    got = (b["median_step_time_s"] - a["median_step_time_s"]) * 1000
    # at least the modeled delay (the wait before it also removes some overlap, so a little more is normal)
    ok = got >= 0.9 * expect
    print(f'{"PASS" if ok else "FAIL"}  modeled 10 Gbit/s link on DDP: expected at least about +{expect:.1f} ms/step, '
          f"measured +{got:.1f} ms/step")
EOF
