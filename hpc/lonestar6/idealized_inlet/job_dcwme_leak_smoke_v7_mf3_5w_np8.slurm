#!/bin/bash
# DC-WME L_wme cleanup leak smoke test (commit 965ad96 validation).
# Same config as 3176096 (mf3) but n-windows=5 on dev queue.
# 3176096 hit OOM in W5 forward solve (rank 2: 28.5 GB / 32 GB)
# because the cleanup code at idealized_inlet_da.py:1217 looped over
# DiagonalCovariance attrs while the iterative L_wme path actually uses
# DenseCovariance (ksp+mat). Fix 965ad96 extends the attribute list.
#
# Pass criteria: _check_memory output stays bounded across 2-3 windows.
# If memory grows ~5-6 GB/rank/window we still have a leak; if it
# plateaus the fix works and we can launch the 30h normal-queue run.
#
# Walltime cap (2h dev) likely truncates at W2 or W3 — that's plenty
# of windows to evaluate per-window memory behavior.
# --------------------------------------------------------
#SBATCH -J dcw_leak_smoke
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o dcw_leak_smoke_v7_mf3.%j.out
#SBATCH -e dcw_leak_smoke_v7_mf3.%j.err

set -uxo pipefail
shopt -s nullglob
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
unset SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS
unset SWE4DVAR_CAPTURE_REPLAY_META
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
TAG="dcw_leak_smoke_v7_mf3_5w"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_dcwme_static_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 7 --track-shift 0 \
  --track-duration-s 144000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 5 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --fixed-sigma-b-sq-h 0.01 \
  --fixed-sigma-b-sq-uv 0.15 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window results ==="
for j in $RES_DIR/result_dcwme_static_cycling_w*-*.json; do
  [[ -f "$j" ]] || continue
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f\"    W{w['window']-1}: bg={w['bg_rmse']:.4f}  an={w['analysis_rmse']:.4f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\") for w in d['windows']]" 2>/dev/null
done

sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET
