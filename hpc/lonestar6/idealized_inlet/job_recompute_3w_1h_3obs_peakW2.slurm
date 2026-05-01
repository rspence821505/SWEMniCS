#!/bin/bash
# 3-window cycling DA, 1-hour windows, 3 observations per window,
# storm slowed so PEAK SURGE lands in W2.
#
# Geometry: default track (40, -20) -> (25, 50) km, but
# track_duration_s = 60000 s (16.7 h) instead of the default 8 h.
# This shifts the storm timeline by 2.08x so the closest-approach
# point lands at t=6.5h (middle of W2 in the 1h-window cycling).
#
# Storm timeline (default geometry, td=60000):
#   t = 4.77 h  -> storm crosses coast (y=0)               [W0 starts at 4h, ends 5h]
#   t = 6.50 h  -> closest approach to inlet (frac=0.39)  [W2 covers 6-7h]
#   t = 16.70 h -> end of track (storm held at (25,50) thereafter)
#
# DA window timeline:
#   ramp 0-4h  : storm at sea, building intensity, far from coast
#   W0   4-5h  : storm crosses coast within window (landfall in W0)
#   W1   5-6h  : storm moving over shelf, intensifying at inlet
#   W2   6-7h  : CLOSEST APPROACH at t=6.5h (peak forcing + surge)
#
# All bug fixes on (Bug 5 sweep-KSP). Bug 6 reverted (default off).
# 3 obs/window via obs-frequency=3 (slots [0, 3, 6]).
#
# Acceptance:
#   - W0 imp competitive with the validated 1h baseline (~25%) given
#     storm dynamics now in W0 instead of decay phase
#   - W2 shows MEANINGFUL improvement_pct (>10%), proving DA can latch
#     onto peak-surge dynamics with adequate observation density
#   - W1 not stuck at ~0% (the previous cycling-divergence pattern)
# --------------------------------------------------------
#SBATCH -J recJ_pkW2
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o recJ_pkW2.%j.out
#SBATCH -e recJ_pkW2.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_recJ_pkW2_${SLURM_JOB_ID}_rank{rank}.csv"

WATCH_LOG="$RES_DIR/watchdog_recJ_pkW2_${SLURM_JOB_ID}.log"
( while sleep 60; do
    echo "=== $(date +%T) ==="
    free -h | awk '/Mem:/ {print "  total="$2" used="$3" free="$4" avail="$7}'
    ps -eo pid,rss,comm --sort=-rss | grep -E "python|swe" | head -5
  done > "$WATCH_LOG" 2>&1 ) &
WPID=$!
trap "kill $WPID 2>/dev/null || true" EXIT

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

# Default track endpoints; only track_duration_s changed (8h -> 16.7h)
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --track-duration-s 60000 \
  --nt-ramp 24 --nt-da 6 \
  --n-windows 3 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 3 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --max-iterations 15 --max-funcs 15 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window result JSONs ==="
for j in $RES_DIR/result_4dvar_N_A_cycling_w*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j')); w=d['windows'][0]; print(f\"    bg={w['bg_rmse']:.6f}  an={w['analysis_rmse']:.6f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\")" 2>/dev/null || cat "$j" | head -20
done

echo
echo "=== peak RSS (watchdog) ==="
[[ -f "$WATCH_LOG" ]] && awk '/python|swe/ {if ($2+0 > peak) {peak=$2+0; line=$0}} END {if (line) print "  peak rss "peak" KB  ("line")"}' "$WATCH_LOG"

echo
echo "=== per-eval after_cleanup RSS (rank 0) ==="
csv="$RES_DIR/eval_mem_recJ_pkW2_${SLURM_JOB_ID}_rank0.csv"
[[ -f "$csv" ]] && awk -F, '$4=="after_cleanup"' "$csv" | tail -15 | awk -F, '{printf "  eval=%s rss=%s MB\n", $2, $5}'

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

exit $RET
