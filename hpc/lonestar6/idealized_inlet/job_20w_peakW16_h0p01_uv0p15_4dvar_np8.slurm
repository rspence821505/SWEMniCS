#!/bin/bash
# 24h cycling DA: 20 windows × 1h DA + 4h ramp; storm peak at simulation t=20h.
# h=0.01, uv=0.15 (deep dip), 4D-Var, np=8.
#
# Window structure:
#   - nt-ramp=48 (4h spin-up)
#   - nt-da=12 (1h per window)
#   - n-windows=20 → W0 (t=4-5h) through W19 (t=23-24h)
#   - track-duration-s=144000 → peak at t=20h (in W16)
#
# Walltime: at np=8 deep-dip h=0.01, ~32 min/window. 20 windows ~ 10.7h.
# 14h walltime gives ~30% margin to absorb peak-window slowdown.
#
# With incremental cycling JSON save (978a88f), any walltime cap still
# preserves partial results through last completed window.
# --------------------------------------------------------
#SBATCH -J 24h_4dv_h01
#SBATCH -A ADCIRC
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 14:00:00
#SBATCH -o 24h_peakW16_h0p01_uv0p15_4dvar.%j.out
#SBATCH -e 24h_peakW16_h0p01_uv0p15_4dvar.%j.err

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
TAG="24h_peakW16_4dvar_h0p01_uv0p15"
export SWE4DVAR_FORWARD_DIAG_CSV="$RES_DIR/forward_diag_${TAG}_${SLURM_JOB_ID}.csv"

rm -f $RES_DIR/chain_bg_after_w*_rank*.npy
rm -f $RES_DIR/result_4dvar_N_A_cycling_w*-*.json 2>/dev/null

ibrun python -u experiments/idealized_inlet_da.py \
  --method 4dvar \
  --vmax 7 --track-shift 0 \
  --track-duration-s 144000 \
  --dt 300 \
  --nt-ramp 48 --nt-da 12 \
  --n-windows 20 --start-window 0 --chain-mode \
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
for j in $RES_DIR/result_4dvar_N_A_cycling_w*-*.json; do
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
