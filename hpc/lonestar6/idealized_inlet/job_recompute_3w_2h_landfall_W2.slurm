#!/bin/bash
# 3-window cycling DA, 2-hour windows, recompute-J adjoint, with the
# storm track redesigned so LANDFALL (storm crossing y=0, the open-ocean
# boundary that represents the coast) lands at the MIDDLE OF W2.
#
# Track geometry (NEW, vs default):
#   start: (40, -20) km  -- 20 km out at sea, far east  (unchanged)
#   end:   (10,  20) km  -- over the shallow shelf, NW of the inlet
#                          (default was (25, 50) km — well past coast)
#   The line passes through (25, 0) km exactly at frac=0.5 of the
#   track, so the storm center crosses the coast at the midpoint of
#   the track duration.
#
# Timing (with nt_ramp=24 → DA starts at t=4h, 2h windows):
#   ramp:  0 → 4h   (storm at sea, building intensity)
#   W0:    4 → 6h   (storm 30→26 km offshore, approaching)
#   W1:    6 → 8h   (storm 26→22 km offshore, intensifying at coast)
#   W2:    8 → 10h  (LANDFALL at t=9h; mid-W2)
#   track ends at t=18h, but DA finishes at t=10h so storm still has
#   plenty of "active" wind during the entire DA window.
#
#   track_duration_s = 64800 (18h) → frac at t=9h is 0.5 → landfall.
#
# Other parameters identical to the validated 3w_2h_dev job, including
# the recompute-J memory-leak fix (sweep-KSP=1, copy=False).
# --------------------------------------------------------
#SBATCH -J recJ_lf_W2
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o recJ_lf_W2.%j.out
#SBATCH -e recJ_lf_W2.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# --- Recompute mode (Bug 5 fix) ----------------------------------------
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

# --- Diagnostics -------------------------------------------------------
RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_recJ_lf_W2_${SLURM_JOB_ID}_rank{rank}.csv"

WATCH_LOG="$RES_DIR/watchdog_recJ_lf_W2_${SLURM_JOB_ID}.log"
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
# track endpoints in METERS:
#   --track-start-x 40000   --track-start-y -20000   (40, -20) km
#   --track-end-x   10000   --track-end-y    20000   (10,  20) km
# track-duration-s 64800 = 18h → landfall at t=9h (mid-W2).
# vmax kept at 10 m/s for parity with the validated runs; bump to 25-30
# in a follow-up if a stronger surge signal is desired.
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --track-start-x 40000 --track-start-y -20000 \
  --track-end-x   10000 --track-end-y    20000 \
  --track-duration-s 64800 \
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
echo "=== peak RSS (watchdog) ==="
if [[ -f "$WATCH_LOG" ]]; then
  awk '/python|swe/ {if ($2+0 > peak) {peak=$2+0; line=$0}} END {if (line) print "  peak rss "peak" KB  ("line")"}' "$WATCH_LOG"
fi

echo
echo "=== per-eval after_cleanup RSS (rank 0) ==="
csv="$RES_DIR/eval_mem_recJ_lf_W2_${SLURM_JOB_ID}_rank0.csv"
[[ -f "$csv" ]] && awk -F, '$4=="after_cleanup"' "$csv" | tail -10 | awk -F, '{printf "  eval=%s rss=%s MB\n", $2, $5}'

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

exit $RET
