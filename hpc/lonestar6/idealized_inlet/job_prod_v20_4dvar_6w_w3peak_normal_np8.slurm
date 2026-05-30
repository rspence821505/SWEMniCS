#!/bin/bash
# Production: 4D-Var v=20 + WD 6-window cycling, peak W3, ALL FIXES.
# Companion to the DC-WME 6w production run.
#
# Fix set (commits up to 065db23):
#  - REUSE=0                         (W2-hang fix)
#  - in-process cycling              (4D-Var has no L_wme memory accumulation)
#  - forward-evolve NaN+magnitude guard (chain_bg corruption prevention)
#  - bg-feasibility preflight + seed rescue (no fake "0-iter TAO converged")
#
# Walltime: at v=20 + WD + REUSE=0 in-process, ~75-90 min/window.
# 6 windows × 90 min = 9h. 14h walltime gives ~50% peak-window margin.
# --------------------------------------------------------
#SBATCH -J prod_4dv_v20_6w
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 14:00:00
#SBATCH -o prod_v20_4dvar_6w.%j.out
#SBATCH -e prod_v20_4dvar_6w.%j.err

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
export SWE4DVAR_FORWARD_NEWTON_REUSE=0
export SWE4DVAR_ADJOINT_SWEEP_KSP=1
export SWE4DVAR_MALLOC_TRIM=1

export SWE4DVAR_OUTPUT_SUBDIR=prod_v20_4dvar_6w
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="prod_v20_4dvar_6w_w3peak"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 0 \
  --track-duration-s 54000 \
  --dt 150 \
  --nt-ramp 96 --nt-da 24 \
  --n-windows 6 --start-window 0 --chain-mode \
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
echo "=== PRODUCTION: 4D-Var v=20 + WD 6w peak-W3 ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}  bg_status={w.get(\"bg_status\",\"?\")}') for w in d['windows']]" 2>/dev/null
done
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET
