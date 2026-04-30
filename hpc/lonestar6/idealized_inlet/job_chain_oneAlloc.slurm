#!/bin/bash
# Single SLURM allocation that runs an N-window cycling-DA chain by issuing
# one fresh `srun` per window inside the same allocation. Each window runs as
# a fresh MPI process so the OS reclaims PETSc/glibc allocator pools between
# windows. Equivalent memory semantics to the multi-job afterok chain, but
# only one sbatch + one queue wait.
#
# Walltime budget: ~25 min/window observed at vmax=10, np=4, nt_da=6, max-funcs=15.
# 4 windows × 25 min ≈ 1h40m → request 2h to be safe.
# --------------------------------------------------------
#SBATCH -J chain_alloc
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o chain_alloc.%j.out
#SBATCH -e chain_alloc.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# --- Canonical operational adjoint defaults (post-fix) -----------------
# These match the new in-process baseline: persistent sweep-KSP + glibc trim.
# run_chain_in_alloc.sh also sets these but we make them explicit at SLURM
# scope so they survive whatever the helper does.
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

# --- Eval-memory diagnostics for production validation -----------------
# Light per-eval CSV per window per rank. Cost is negligible (~10 lines/eval)
# and gives us peak/window-final memory evidence in the run log.
RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_EVAL_MEM_DIAG=1

# --- Background watchdog: peak-RSS-per-rank tracker --------------------
# Sample free / per-process RSS every 60s. Useful for confirming the run
# stays comfortably below the LS6 256 GB node budget.
WATCH_LOG="$RES_DIR/watchdog_chain_alloc_${SLURM_JOB_ID}.log"
( while sleep 60; do
    echo "=== $(date +%T) ==="
    free -h | awk '/Mem:/ {print "  total="$2" used="$3" free="$4" avail="$7}'
    # Top-5 RSS by python procs (rough proxy for our ranks)
    ps -eo pid,rss,comm --sort=-rss | grep -E "python|swe" | head -5
  done > "$WATCH_LOG" 2>&1 ) &
WPID=$!
trap "kill $WPID 2>/dev/null || true" EXIT

# Pick the launcher: `ibrun` is TACC's wrapper that uses the allocation's
# task-count by default. `srun -n N` works inside an allocation.
bash hpc/lonestar6/idealized_inlet/run_chain_in_alloc.sh \
  --nwin 4 \
  --np 4 \
  --vmax 10 \
  --nt-ramp 24 --nt-da 6 \
  --method 4dvar \
  --launcher ibrun \
  --extra "--obs-fraction 0.05 --obs-frequency 5 --obs-noise-level 0.01 --background-error-std 0.02 --max-iterations 15 --max-funcs 15 --mem-limit-gb 240"

echo "=== chain_alloc SLURM job done ==="
ls -la $WORK/SWEMniCS/results/idealized_inlet_da/result_*_cycling_w*.json 2>/dev/null || true
echo
echo "=== peak RSS observed by watchdog (rank-0-side, host-level) ==="
if [[ -f "$WATCH_LOG" ]]; then
  awk '/python|swe/ {if ($2+0 > peak) {peak=$2+0; line=$0}} END {if (line) print "  peak rss "peak" KB  ("line")"}' "$WATCH_LOG"
fi
echo
echo "=== per-rank end-of-run memory (eval_mem CSVs, last after_cleanup row per window) ==="
for csv in $RES_DIR/eval_mem_chain_alloc_w*_rank0.csv; do
  [[ -f "$csv" ]] || continue
  echo "  $(basename "$csv")"
  awk -F, '$4=="after_cleanup"' "$csv" | tail -1 | awk -F, '{printf "    eval=%s rss=%s MB trim_freed=%s MB\n", $2, $5, $8}'
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
