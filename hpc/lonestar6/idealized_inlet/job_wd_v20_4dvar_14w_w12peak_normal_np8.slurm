#!/bin/bash
# Production scale-up: 4D-Var v=20 + WD, 14-window cycling DA with
# storm peak in W12. Storm-surge-forecasting framing: 14h of cycling
# DA assimilating gauge observations every hour as a tropical cyclone
# intensifies through landfall.
#
# Window structure (nt-ramp=96 at dt=150 = 4h warm-up):
#   W0:  t=4-5h    W1:  t=5-6h    ...    W11: t=15-16h
#   W12: t=16-17h (PEAK, track-dur=118800 → peak at t=16.5h)
#   W13: t=17-18h (post-peak)
#
# Pace at v=20 + WD np=8 dt=150: ~50-60 min/window pre-peak, ~80 min
# approaching peak, peak windows fast-fail (MUMPS ceiling). 14w estimate:
# 11×55min + 2×80min + 1×5min(fast-fail) + 1×60min ≈ 14h. Budget 40h
# to absorb peak-window slowdown and any cycling cleanup overhead.
#
# Per finding_v20_wd_baseline.md: peak windows (W11/W12) expected to
# produce evals=0, an=bg due to MUMPS singular-Jacobian ceiling.
# Meaningful 4D-Var-vs-DC-WME comparison is W0-W10 (pre-peak).
# --------------------------------------------------------
#SBATCH -J 14w_4dv_v20
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 40:00:00
#SBATCH -o 14w_w12peak_v20_wd_4dvar.%j.out
#SBATCH -e 14w_w12peak_v20_wd_4dvar.%j.err

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

# Isolated output subdir so the parallel DC-WME companion doesn't clobber
# this run's chain_bg / result JSONs.
export SWE4DVAR_OUTPUT_SUBDIR=14w_v20_4dvar
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="14w_w12peak_v20_wd_4dvar"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 0 \
  --track-duration-s 118800 \
  --dt 150 \
  --nt-ramp 96 --nt-da 24 \
  --n-windows 14 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 3 --max-funcs 3 \
  --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 \
  --wd --wd-alpha 1.5 \
  --mem-limit-gb 240
RET=$?

for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== v=20 + WD 4D-Var 14w peak-W12 results ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  echo "  $(basename "$j")"
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}') for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET
