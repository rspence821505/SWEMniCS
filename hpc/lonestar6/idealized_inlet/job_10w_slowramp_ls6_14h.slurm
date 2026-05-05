#!/bin/bash
# 10w slow-ramp peak-reach follow-up (14h walltime).
#
# Prior 10w_slow (3142250) walltime-capped at 8h after 7 of 10
# windows. Confirmed slow ramp delays the W3 cycling ceiling by 3
# windows (W3 bg 0.174 vs iter14 baseline 0.285 = -39%) but did
# not reach storm peak. With track-duration-s=157000 peak hits
# W8-W9, which 3142250 did not see.
#
# 3142250 measured pace: 8h / 7 windows = ~68 min/window. For 10
# windows ~11.5h. 14h walltime gives 22% margin to absorb any
# slowdown at peak from harder Newton convergence.
#
# Same physics + DA config as 3142250; only change is walltime.
# Goal: capture peak-window (W8-W9) bg_rmse to test whether slow
# ramp keeps DA viable through the storm peak, or whether the
# cycling ceiling re-asserts at peak.
# --------------------------------------------------------
#SBATCH -J 10w_slow
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 14:00:00
#SBATCH -o 10w_slow_ls6.%j.out
#SBATCH -e 10w_slow_ls6.%j.err

set -euxo pipefail
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
source $WORK/SWEMniCS/hpc/lonestar6/idealized_inlet/env.ls6.sh
git -C $WORK/SWEMniCS rev-parse HEAD
cd $WORK/SWEMniCS

export SWE4DVAR_NEWTON_RELAX=0.3
export SWE4DVAR_ADJOINT_RECOMPUTE_JACOBIANS=1
export SWE4DVAR_FORWARD_NEWTON_REUSE=1
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

RES_DIR="$WORK/SWEMniCS/results/idealized_inlet_da"
mkdir -p "$RES_DIR"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_10w_slow_14h_ls6_${SLURM_JOB_ID}.csv"
rm -f "$SWE4DVAR_FORWARD_DIAG_CSV"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 7 --track-shift 0 \
  --track-duration-s 157000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 10 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 5 --max-funcs 5 \
  --mem-limit-gb 240
RET=$?

echo
echo "=== Per-window results ==="
for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f\"    W{w['window']-1}: bg={w['bg_rmse']:.4f}  an={w['analysis_rmse']:.4f}  imp={w['improvement_pct']:+.2f}%  evals={w['n_func_evals']}\") for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET
