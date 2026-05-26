#!/bin/bash
# v=20 + WD + 4D-Var probe: cycling-stall breaker (mf10 + ls-step-max 1.0).
# Targets the W3-W8 imp≈0% regime observed in 3182228 by:
#   (A) max-iter=10 max-funcs=10 (vs mf3): gives TAO more line-search budget,
#       so a single rejection doesn't fall back to iter-0 = bg.
#   (B) --ls-step-max 1.0: caps step length to keep trial states inside
#       Newton's basin of attraction.
#
# 6 windows (W0-W5) is enough to see whether the stall opens up:
#   3182228 mf3 trajectory: W0 +5.97%, W1 +1.10%, W2 +0.33%, W3-W5 ~0%.
#   Target: W3-W5 lifting from 0% to ~1-3%.
#
# Pace projection: mf3 was ~95 min/window. mf10 worst case ~3h/window
# (7 extra forward solves × ~12 min). 6w × 3h = 18h; 30h walltime gives
# ~67% margin.
# --------------------------------------------------------
#SBATCH -J mf10_v20_4dv
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 30:00:00
#SBATCH -o mf10_v20_wd_4dvar.%j.out
#SBATCH -e mf10_v20_wd_4dvar.%j.err

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

# Isolated output subdir
export SWE4DVAR_OUTPUT_SUBDIR=mf10_v20_4dvar
RES_DIR="$WORK/SWEMniCS/results/${SWE4DVAR_OUTPUT_SUBDIR}"
mkdir -p "$RES_DIR"
TAG="mf10_v20_wd_4dvar_6w"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 20 --track-shift 0 \
  --track-duration-s 118800 \
  --dt 150 \
  --nt-ramp 96 --nt-da 24 \
  --n-windows 6 --start-window 0 --chain-mode \
  --obs-fraction 0.05 --obs-frequency 6 \
  --obs-noise-level 0.01 --background-error-std 0.02 \
  --smooth-length 1500 \
  --max-iterations 10 --max-funcs 10 \
  --ls-step-max 1.0 \
  --fixed-sigma-b-sq-h 0.01 --fixed-sigma-b-sq-uv 0.15 \
  --wd --wd-alpha 1.5 \
  --mem-limit-gb 240
RET=$?

for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
  cp "$j" "$RES_DIR/${TAG}_$(basename $j)" 2>/dev/null || true
done
echo
echo "=== mf10 + ls-step-max 1.0 — v=20 + WD 4D-Var 6w ==="
for j in $RES_DIR/${TAG}_result_*.json; do
  [[ -f "$j" ]] || continue
  python3 -c "import json; d=json.load(open('$j'));
[print(f'    W{w[\"window\"]-1}: bg={w[\"bg_rmse\"]:.4f}  an={w[\"analysis_rmse\"]:.4f}  imp={w[\"improvement_pct\"]:+.2f}%  evals={w[\"n_func_evals\"]}') for w in d['windows']]" 2>/dev/null
done
echo "=== Comparison vs mf3 baseline (3182228) ==="
echo "  W0: mf3=+5.97%   W1: mf3=+1.10%   W2: mf3=+0.33%   W3: mf3=+0.03%   W4: mf3=+0.03%   W5: mf3=+0.00%"
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,Elapsed 2>&1 || true
exit $RET
