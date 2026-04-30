#!/bin/bash
# 6-window cycling DA, 2-hour windows, recompute-Jacobians-in-adjoint.
#
# Validates that the recompute path scales beyond the 1-window 1-hour
# smoke test (recJ_w0). 2h windows = nt_da=12 doubles the per-eval
# adjoint cost vs the single-window job. Recompute mode saves ~1.4 GB
# of stored-Jacobian memory per eval at nt_da=12, which is what makes
# 6 windows fit comfortably in a single MPI process on one LS6 node.
#
# Per-window budget estimate:
#   nt_da=6 single window: ~25 min wall at np=4, max-funcs=15 (legacy)
#   nt_da=12 (~2x):        ~50 min wall expected
#   6 windows:             ~5 hours
#   + truth re-solves per window in chain mode + slurm overhead
#   Budget: 8 hours.
#
# Acceptance:
#   - All 6 windows complete (no Newton failure, no OOM)
#   - W0 analysis RMSE within ~1% of single-process W0 baseline
#   - W1+ improvements consistent with prior 1h-window cycling runs
#   - Per-eval RSS stays below ~12 GB/rank (recompute saves ~1.4 GB
#     vs stored-J at nt_da=12)
# --------------------------------------------------------
#SBATCH -J recJ_6w_2h
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 08:00:00
#SBATCH -o recJ_6w_2h.%j.out
#SBATCH -e recJ_6w_2h.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

# --- Recompute mode -----------------------------------------------------
# Forces SWE4DVAR_CAPTURE_REPLAY_META=1 inside cost_functions and skips
# legacy stored-J. Also forces SWE4DVAR_ADJOINT_SWEEP_KSP=0 and
# SWE4DVAR_ADJOINT_TRANSPOSE_CACHE=0 (Bug 2 fix) — per-step LU is the
# correct path under recompute.
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_MALLOC_TRIM=1

# --- Diagnostics -------------------------------------------------------
RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_EVAL_MEM_DIAG=1
export SWE4DVAR_EVAL_MEM_DIAG_CSV="$RES_DIR/eval_mem_recJ_6w_2h_${SLURM_JOB_ID}_rank{rank}.csv"

# --- Background watchdog: peak-RSS-per-rank tracker --------------------
WATCH_LOG="$RES_DIR/watchdog_recJ_6w_2h_${SLURM_JOB_ID}.log"
( while sleep 120; do
    echo "=== $(date +%T) ==="
    free -h | awk '/Mem:/ {print "  total="$2" used="$3" free="$4" avail="$7}'
    ps -eo pid,rss,comm --sort=-rss | grep -E "python|swe" | head -5
  done > "$WATCH_LOG" 2>&1 ) &
WPID=$!
trap "kill $WPID 2>/dev/null || true" EXIT

# Wipe stale chain bg files so this is a true cold-start cycling run.
rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

# --- Run --------------------------------------------------------------
# nt_ramp=24 ramp, nt_da=12 = 2-hour DA window (dt=600s).
# 6 windows in chain-mode, 1 sbatch, single ibrun (recompute mode is
# safe in-process; Phase C-1 Newton reuse + Bug 2 fix means no
# operator-handle leaks across windows).
ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 10 --track-shift 0 \
  --nt-ramp 24 --nt-da 12 \
  --n-windows 6 --start-window 0 --chain-mode \
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
for csv in $RES_DIR/eval_mem_recJ_6w_2h_${SLURM_JOB_ID}_rank0.csv; do
  [[ -f "$csv" ]] || continue
  awk -F, '$4=="after_cleanup"' "$csv" | tail -10 | awk -F, '{printf "  eval=%s rss=%s MB\n", $2, $5}'
done

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true

exit $RET
