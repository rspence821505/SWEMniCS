#!/bin/bash
# 3-window cycling DA, 2-hour windows, recompute-Jacobians-in-adjoint.
# Development-queue companion to job_recompute_6w_2h.slurm — runs the
# first half of the same chain so we can validate the multi-window
# recompute path quickly while the 6w job waits in `normal`.
#
# Walltime budget: ~30 min/window expected (nt_da=12, 15 evals
# at ~95s/eval + ~5 min per-window overhead) -> ~90 min for 3 windows.
# Development cap is 2h, leaves ~30 min margin.
#
# Acceptance (matches 6w job for the first 3 windows):
#   - W0 analysis RMSE within ~1% of single-window baseline 0.1079
#   - W1, W2 improvement_pct consistent with prior 1h-window cycling
#   - All 3 windows complete (no OOM, no Newton failure)
# --------------------------------------------------------
#SBATCH -J recJ_3w_2h
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o recJ_3w_2h.%j.out
#SBATCH -e recJ_3w_2h.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# --- Recompute mode -----------------------------------------------------
# With Bug 5 fix (copy=False default), recompute returns the persistent
# form Mat A in-place. Sweep-KSP MUST be on so the LU factor is reused
# across backward steps — without it, per-step LU factor leaks ~190 MB
# (observed in 3136139: 52 GB after 15 evals at nt_da=12).
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

# --- Diagnostics -------------------------------------------------------
RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_recJ_3w_2h_${SLURM_JOB_ID}_rank{rank}.csv"

# --- Background watchdog -----------------------------------------------
WATCH_LOG="$RES_DIR/watchdog_recJ_3w_2h_${SLURM_JOB_ID}.log"
( while sleep 60; do
    echo "=== $(date +%T) ==="
    free -h | awk '/Mem:/ {print "  total="$2" used="$3" free="$4" avail="$7}'
    ps -eo pid,rss,comm --sort=-rss | grep -E "python|swe" | head -5
  done > "$WATCH_LOG" 2>&1 ) &
WPID=$!
trap "kill $WPID 2>/dev/null || true" EXIT

# Wipe stale chain bg files for a true cold-start.
rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

# --- Run ---------------------------------------------------------------
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 12 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 5 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window result JSONs ==="
for j in $RES_DIR/result_4dvar_N_A_cycling_w*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j')); w=d['windows'][0]; print(f\"    bg={w['bg_rmse']:.6f}  an={w['analysis_rmse']:.6f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}  conv={w.get('converged','?')}\")" 2>/dev/null || cat "$j" | head -20
done

echo
echo "=== peak RSS observed by watchdog ==="
if [[ -f "$WATCH_LOG" ]]; then
  awk '/python|swe/ {if ($2+0 > peak) {peak=$2+0; line=$0}} END {if (line) print "  peak rss "peak" KB  ("line")"}' "$WATCH_LOG"
fi

echo
echo "=== per-window after_cleanup RSS (rank 0) ==="
for csv in $RES_DIR/eval_mem_recJ_3w_2h_${SLURM_JOB_ID}_rank0.csv; do
  [[ -f "$csv" ]] || continue
  awk -F, '$4=="after_cleanup"' "$csv" | tail -10 | awk -F, '{printf "  eval=%s rss=%s MB\n", $2, $5}'
done

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

exit $RET
