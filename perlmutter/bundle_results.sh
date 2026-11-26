#!/bin/bash
# Package finished runs for copying off Perlmutter: copies (never moves) the given paths, adds the Slurm job summary
# and Slurm GPU accounting for the given jobs, redacts the scratch path, user name and Slurm account in the copies,
# and writes $SCRATCH/<name>.tgz.
#
#   cd $SCRATCH/powersgd
#   bash perlmutter/bundle_results.sh gloo_more 61000001,61000002 results/scaling logs/gpt_slow_61000003.out
#   # then, on your laptop:  scp <user>@perlmutter.nersc.gov:<scratch>/gloo_more.tgz ~/Downloads/
#
# Paths are relative to $PROJ (default $SCRATCH/powersgd); shell globs work. Job IDs are comma-separated.
set -eo pipefail
[[ $# -ge 3 ]] || { sed -n '2,10p' "$0"; exit 1; }
name=$1
jobs=$2
shift 2
PROJ=${PROJ:-$SCRATCH/powersgd}
cd "$PROJ"
stage=$(mktemp -d "$SCRATCH/bundle_${name}.XXXX")
dest=$stage/$name
mkdir -p "$dest"
for p in "$@"; do
  [[ -e $p ]] || { echo "missing: $p" >&2; exit 1; }
  mkdir -p "$dest/$(dirname "$p")"
  cp -R "$p" "$dest/$(dirname "$p")/"
done

sacct -j "$jobs" -X -P --format=JobID,JobName%60,Elapsed,State,NNodes,ExitCode,Start,End > "$dest/jobs.txt"
{
  echo "# GPU usage per job step from Slurm accounting (TRESUsageInAve). gpuutil = average % of time a GPU kernel ran;"
  echo "# gpumem = average GPU memory used. The .batch step is the launcher and reports 0. Jobs started by"
  echo "# perlmutter/run.sbatch with GPU_MONITOR=1 have the monitor as step .0 and training as step .1."
  echo "# Coarse sampling makes this unreliable for jobs of a minute or two; prefer the nvidia-smi CSVs when present."
  echo "# Command: sacct -j $jobs -P -n --format=JobID,JobName%60,TRESUsageInAve%500"
  sacct -j "$jobs" -P -n --format=JobID,JobName%60,TRESUsageInAve%500
} > "$dest/gpu_usage_slurm.txt"

acct=$(sacct -j "${jobs%%,*}" -X -n -P --format=Account | head -n1)
grep -rIl '' "$dest" | while read -r f; do
  # the GPU account (e.g. m1234_g) first, then the project name without the _g suffix
  R_SCRATCH="$SCRATCH" R_USER="$USER" R_ACCT="$acct" R_PROJ="${acct%_g}" perl -pi -e '
    s/\Q$ENV{R_SCRATCH}\E/\$SCRATCH/g;
    s/\Q$ENV{R_ACCT}\E/<account>/g if length $ENV{R_ACCT};
    s/\b\Q$ENV{R_PROJ}\E\b/<account>/g if length $ENV{R_PROJ};
    s/\b\Q$ENV{R_USER}\E\b/<user>/g;' "$f"
done

tar czf "$SCRATCH/$name.tgz" -C "$stage" "$name"
echo "wrote $SCRATCH/$name.tgz ($(du -h "$SCRATCH/$name.tgz" | cut -f1)); staging copy in $stage"
echo "check before sharing:  grep -r -c -E \"$USER|${acct:-NO_ACCOUNT}\" $dest | grep -v ':0$' || echo clean"
