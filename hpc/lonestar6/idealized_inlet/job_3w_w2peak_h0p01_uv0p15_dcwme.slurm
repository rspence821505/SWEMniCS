#!/bin/bash
# 3-window cycling DA with storm peak in W2.
# sigma_b^2_h=0.01, sigma_b^2_uv=0.15
# track-duration-s=43200 places peak at simulation t=6h (start of W2)
# --------------------------------------------------------
#SBATCH -J 3w_dcw_np8
#SBATCH -A ADCIRC
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00
#SBATCH -o 3w_w2peak_h0p01_uv0p15_dcwme.%j.out
#SBATCH -e 3w_w2peak_h0p01_uv0p15_dcwme.%j.err

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
TAG="3w_w2peak_dcwme_h0p01_uv0p15"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null
rm -f $RES_DIR/result_dcwme_*cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method dcwme_static \
  --vmax 7 --track-shift 0 \
  --track-duration-s 43200 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 3 --start-window 0 --chain-mode \
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
for j in $RES_DIR/result_*_cycling_w*-*.json; do
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
