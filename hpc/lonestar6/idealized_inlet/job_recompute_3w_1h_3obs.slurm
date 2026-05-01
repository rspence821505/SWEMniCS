#!/bin/bash
# 3-window cycling DA, 1-hour windows, 3 observations per window.
# Recompute-J adjoint, all bug fixes (Bug 5 sweep-KSP + Bug 6 forward
# Newton KSP reset).
#
# Why this configuration:
#   - 1h windows (nt_da=6) avoid the W2 Newton-divergence we hit on the
#     2h cycling runs (3136354 landfall-W2, 3136299 default).
#   - obs-frequency=3 gives 3 obs slots per window (k=0, 3, 6) instead
#     of the 2 slots used in the validated 1h baseline (3135966 had
#     obs_frequency=5 → only 2 slots: k=0, 5).
#   - Default storm geometry (no track-* flags overridden), matching the
#     validated 1h baseline (0.1454 → 0.1079, -25.79%).
#
# Acceptance:
#   - W0 imp >= -25% (matches or beats 1h baseline)
#   - W1, W2 don't go to inf-cost / Newton-divergence regime
#   - Per-eval growth < 30 MB (Bug 6 fix)
# --------------------------------------------------------
#SBATCH -J recJ_3w1h3o
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 01:30:00
#SBATCH -o recJ_3w1h3o.%j.out
#SBATCH -e recJ_3w1h3o.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
# SWE4DVAR_FORWARD_KSP_RESET intentionally NOT set — Bug 6 attempt
# was a regression (3136376 showed ~200 MB/eval vs 3136299's ~165).
# Default in code is now off.
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_recJ_3w1h3o_${SLURM_JOB_ID}_rank{rank}.csv"

WATCH_LOG="$RES_DIR/watchdog_recJ_3w1h3o_${SLURM_JOB_ID}.log"
( while sleep 60; do
    echo "=== $(date +%T) ==="
    free -h | awk '/Mem:/ {print "  total="$2" used="$3" free="$4" avail="$7}'
    ps -eo pid,rss,comm --sort=-rss | grep -E "python|swe" | head -5
  done > "$WATCH_LOG" 2>&1 ) &
WPID=$!
trap "kill $WPID 2>/dev/null || true" EXIT

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

# 1h windows (nt_da=6), 3 obs/window (obs-frequency=3 → slots [0, 3, 6])
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
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
echo "=== per-eval after_cleanup RSS (rank 0) — Bug 6 acceptance: <30 MB/eval growth ==="
csv="$RES_DIR/eval_mem_recJ_3w1h3o_${SLURM_JOB_ID}_rank0.csv"
[[ -f "$csv" ]] && awk -F, '$4=="after_cleanup"' "$csv" | tail -15 | awk -F, '{printf "  eval=%s rss=%s MB\n", $2, $5}'

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

exit $RET
